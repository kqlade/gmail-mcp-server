/**
 * Gmail API client wrapper with token refresh support.
 *
 * This module handles:
 * - Creating authenticated Gmail client from stored credentials
 * - Automatic token refresh
 * - Gmail API method wrappers
 * - Response transformation to spec-defined formats
 */

import { google, gmail_v1 } from 'googleapis';
import type { TokenStore, GmailCredentials, AccountInfo } from '../store/interface.js';
import { encrypt, decrypt } from '../utils/crypto.js';
import { NotAuthorizedError, GmailApiError, InsufficientScopeError } from '../utils/errors.js';

export interface GmailClientDependencies {
  tokenStore: TokenStore;
  encryptionKey: string;
  googleClientId: string;
  googleClientSecret: string;
}

export interface MessageHeader {
  name: string;
  value: string;
}

export interface MessageMetadata {
  id: string;
  threadId: string;
  snippet: string;
  headers: {
    from?: string;
    to?: string;
    subject?: string;
    date?: string;
  };
  attachments: AttachmentMetadata[];
}

export interface MessageFull extends MessageMetadata {
  body: {
    text?: string;
    html?: string;
  };
}

export interface AttachmentMetadata {
  attachmentId: string;
  filename: string;
  mimeType: string;
  size: number;
}

export interface SearchResultMessage {
  id: string;
  threadId: string;
  snippet?: string;
}

export interface SearchResult {
  messages: SearchResultMessage[];
  nextPageToken?: string;
}

export interface BatchSearchResult {
  query: string;
  result?: SearchResult;
  error?: string;
}

export interface ThreadListItem {
  id: string;
  snippet?: string;
  historyId?: string;
  subject?: string;
  from?: string;
  date?: string;
  messageCount?: number;
}

export interface ThreadResult {
  threads: ThreadListItem[];
  nextPageToken?: string;
}

export interface ModifyResult {
  id: string;
  success: boolean;
  error?: string;
}

export interface BatchModifyResult {
  results: ModifyResult[];
  successCount: number;
  failureCount: number;
}

export interface LabelInfo {
  id: string;
  name: string;
  type?: string;
}

export interface LabelDetailedInfo extends LabelInfo {
  messagesTotal: number;
  messagesUnread: number;
  threadsTotal: number;
  threadsUnread: number;
}

export interface DraftInfo {
  id: string;
  messageId: string;
}

export interface DraftContent {
  id: string;
  messageId: string;
  snippet: string;
  subject?: string;
  to?: string;
  body?: {
    text?: string;
    html?: string;
  };
}

export interface DraftListResult {
  drafts: DraftInfo[];
  nextPageToken?: string;
}

// Gmail scopes
const GMAIL_LABELS_SCOPE = 'https://www.googleapis.com/auth/gmail.labels';
const GMAIL_MODIFY_SCOPE = 'https://www.googleapis.com/auth/gmail.modify';
const GMAIL_COMPOSE_SCOPE = 'https://www.googleapis.com/auth/gmail.compose';

// Token refresh threshold (5 minutes before expiry)
const REFRESH_THRESHOLD_MS = 5 * 60 * 1000;

// Default body truncation limits
const DEFAULT_MAX_BODY_LENGTH = 50000; // 50KB default for full format
const SUMMARY_MAX_BODY_LENGTH = 2000;  // 2KB for summary format

// Client cache for reusing Gmail clients
interface CachedClient {
  client: gmail_v1.Gmail;
  expiresAt: number;
}
const clientCache = new Map<string, CachedClient>();

/**
 * Simple async semaphore to limit concurrent Gmail API calls.
 * Prevents overwhelming the Gmail API rate limits when many
 * requests arrive in parallel (e.g., agent issues 20+ email_get calls).
 */
const MAX_CONCURRENT_GMAIL_CALLS = 5;
let activeGmailCalls = 0;
const waitQueue: Array<() => void> = [];

async function withGmailConcurrency<T>(fn: () => Promise<T>): Promise<T> {
  if (activeGmailCalls >= MAX_CONCURRENT_GMAIL_CALLS) {
    await new Promise<void>((resolve) => waitQueue.push(resolve));
  }
  activeGmailCalls++;
  try {
    return await fn();
  } finally {
    activeGmailCalls--;
    const next = waitQueue.shift();
    if (next) next();
  }
}

// ---------------------------------------------------------------------------
// Retry for transient Google API failures
// ---------------------------------------------------------------------------

const MAX_GMAIL_RETRIES = 2;
const RETRY_BASE_MS = 500;
const MAX_CONCURRENT_BATCH_SEARCHES = 2;

function isTransientGmailError(error: unknown): boolean {
  if (!(error instanceof Error)) return false;
  const msg = error.message.toLowerCase();
  if (
    msg.includes('unexpected end of json') ||
    msg.includes('unterminated string in json') ||
    msg.includes('econnreset') ||
    msg.includes('etimedout') ||
    msg.includes('enotfound') ||
    msg.includes('enetunreach') ||
    msg.includes('eai_again') ||
    msg.includes('fetch failed') ||
    msg.includes('socket hang up') ||
    msg.includes('aborted')
  ) {
    return true;
  }
  const status = (error as { code?: number }).code;
  if (status === 429 || status === 500 || status === 502 || status === 503 || status === 504) {
    return true;
  }
  return false;
}

async function withRetry<T>(fn: () => Promise<T>): Promise<T> {
  for (let attempt = 0; ; attempt++) {
    try {
      return await fn();
    } catch (error) {
      if (attempt < MAX_GMAIL_RETRIES && isTransientGmailError(error)) {
        const delay = RETRY_BASE_MS * 2 ** attempt + Math.floor(Math.random() * 250);
        await new Promise<void>((r) => setTimeout(r, delay));
        continue;
      }
      throw error;
    }
  }
}

async function mapWithConcurrency<T, R>(
  items: T[],
  limit: number,
  mapper: (item: T, index: number) => Promise<R>
): Promise<R[]> {
  if (items.length === 0) return [];

  const concurrency = Math.max(1, Math.min(limit, items.length));
  const results = new Array<R>(items.length);
  let nextIndex = 0;

  async function worker(): Promise<void> {
    while (true) {
      const currentIndex = nextIndex;
      nextIndex += 1;
      if (currentIndex >= items.length) return;
      results[currentIndex] = await mapper(items[currentIndex]!, currentIndex);
    }
  }

  await Promise.all(Array.from({ length: concurrency }, () => worker()));
  return results;
}

/**
 * Create a Gmail client factory.
 */
export function createGmailClientFactory(deps: GmailClientDependencies) {
  const { tokenStore, encryptionKey, googleClientId, googleClientSecret } = deps;

  /**
   * Get or refresh credentials for a user.
   * If email is specified, gets that specific account.
   * If email is omitted, gets the default account.
   */
  async function getValidCredentials(mcpUserId: string, email?: string): Promise<GmailCredentials> {
    const credentials = await tokenStore.getCredentials(mcpUserId, email);

    if (!credentials) {
      if (email) {
        throw new NotAuthorizedError(`Gmail account ${email} not connected. Use gmail.listAccounts to see connected accounts.`);
      }
      throw new NotAuthorizedError('Gmail not connected. Use gmail.authorize to link your Gmail account.');
    }

    // Check if token needs refresh
    const now = Date.now();
    if (credentials.expiryDate - now < REFRESH_THRESHOLD_MS) {
      return await refreshCredentials(credentials);
    }

    return credentials;
  }

  /**
   * Refresh access token using stored refresh token.
   */
  async function refreshCredentials(credentials: GmailCredentials): Promise<GmailCredentials> {
    const oauth2Client = new google.auth.OAuth2(
      googleClientId,
      googleClientSecret
    );

    // Decrypt refresh token
    const refreshToken = decrypt(credentials.refreshToken, encryptionKey);

    oauth2Client.setCredentials({
      refresh_token: refreshToken,
    });

    try {
      const { credentials: newTokens } = await oauth2Client.refreshAccessToken();

      if (!newTokens.access_token) {
        throw new Error('No access token returned');
      }

      // Update stored credentials
      const newRefreshToken = newTokens.refresh_token
        ? encrypt(newTokens.refresh_token, encryptionKey)
        : undefined;

      await tokenStore.updateAccessToken(
        credentials.mcpUserId,
        credentials.email,
        newTokens.access_token,
        newTokens.expiry_date ?? Date.now() + 3600000,
        newRefreshToken
      );

      return {
        ...credentials,
        accessToken: newTokens.access_token,
        expiryDate: newTokens.expiry_date ?? Date.now() + 3600000,
        refreshToken: newRefreshToken ?? credentials.refreshToken,
      };
    } catch (error: unknown) {
      const errorMessage = error instanceof Error ? error.message : String(error);

      // Check for revocation - delete only this specific account
      if (errorMessage.includes('invalid_grant')) {
        await tokenStore.deleteCredentials(credentials.mcpUserId, credentials.email);
        throw new NotAuthorizedError(`Gmail access for ${credentials.email} revoked. Please re-authorize.`);
      }

      throw new GmailApiError(`Token refresh failed: ${errorMessage}`);
    }
  }

  /**
   * Create an authenticated Gmail client for a user.
   * Clients are cached and reused within their token lifetime.
   */
  async function getGmailClient(mcpUserId: string, email?: string): Promise<gmail_v1.Gmail> {
    const credentials = await getValidCredentials(mcpUserId, email);
    const cacheKey = `${mcpUserId}:${credentials.email}`;

    // Check cache
    const cached = clientCache.get(cacheKey);
    const now = Date.now();
    if (cached && cached.expiresAt > now) {
      return cached.client;
    }

    const oauth2Client = new google.auth.OAuth2(
      googleClientId,
      googleClientSecret
    );

    oauth2Client.setCredentials({
      access_token: credentials.accessToken,
    });

    const client = google.gmail({
      version: 'v1',
      auth: oauth2Client,
      timeout: 60_000,
    });

    // Cache client until token expires (with 1 minute buffer)
    clientCache.set(cacheKey, {
      client,
      expiresAt: credentials.expiryDate - 60000,
    });

    return client;
  }

  /**
   * Search messages using Gmail query syntax.
   * Returns thread IDs + snippets from a single threads.list call -- no fan-out.
   */
  async function searchMessages(
    mcpUserId: string,
    query: string,
    maxResults: number = 20,
    pageToken?: string,
    email?: string
  ): Promise<SearchResult> {
    const result = await listThreads(mcpUserId, query, maxResults, pageToken, email);
    return {
      messages: result.threads.map((thread) => ({
        id: thread.id,
        threadId: thread.id,
        snippet: thread.snippet,
      })),
      nextPageToken: result.nextPageToken,
    };
  }

  /**
   * Batch search: run multiple queries in parallel.
   * Returns results for each query.
   */
  async function batchSearchMessages(
    mcpUserId: string,
    queries: Array<{ query: string; maxResults?: number }>,
    email?: string
  ): Promise<BatchSearchResult[]> {
    return mapWithConcurrency(queries, MAX_CONCURRENT_BATCH_SEARCHES, async ({ query, maxResults }) => {
      try {
        const result = await searchMessages(mcpUserId, query, maxResults ?? 20, undefined, email);
        return { query, result };
      } catch (error) {
        return {
          query,
          error: error instanceof Error ? error.message : String(error),
        };
      }
    });
  }

  /**
   * Get a single message.
   *
   * Formats:
   * - 'metadata': Headers and snippet only (fastest)
   * - 'summary': Headers, snippet, and first 2KB of text body
   * - 'full': Complete message with body (supports truncation options)
   *
   * Options for 'full' and 'summary' formats:
   * - maxBodyLength: Truncate body to this many characters (default: 50KB for full, 2KB for summary)
   * - includeHtml: Include HTML body (default: true for full, false for summary)
   */
  async function getMessage(
    mcpUserId: string,
    messageId: string,
    format: 'metadata' | 'summary' | 'full' = 'metadata',
    email?: string,
    options?: {
      maxBodyLength?: number;
      includeHtml?: boolean;
    }
  ): Promise<MessageMetadata | MessageFull> {
    const gmail = await getGmailClient(mcpUserId, email);

    try {
      // For metadata-only, use metadata format; otherwise use full to get body
      const apiFormat = format === 'metadata' ? 'metadata' : 'full';

      const response = await gmail.users.messages.get({
        userId: 'me',
        id: messageId,
        format: apiFormat,
        metadataHeaders: ['From', 'To', 'Subject', 'Date'],
      });

      const message = response.data;
      const headers = extractHeaders(message.payload?.headers ?? []);
      const attachments = extractAttachments(message.payload);

      const metadata: MessageMetadata = {
        id: message.id!,
        threadId: message.threadId!,
        snippet: message.snippet ?? '',
        headers,
        attachments,
      };

      if (format === 'metadata') {
        return metadata;
      }

      // Determine truncation settings based on format
      const isSummary = format === 'summary';
      const maxBodyLength = options?.maxBodyLength ??
        (isSummary ? SUMMARY_MAX_BODY_LENGTH : DEFAULT_MAX_BODY_LENGTH);
      const includeHtml = options?.includeHtml ?? !isSummary;

      const body = extractBody(message.payload, { maxBodyLength, includeHtml });

      // Add truncation info to response
      const result: MessageFull & { truncated?: boolean; originalSize?: number } = {
        ...metadata,
        body,
      };

      // Check if content was truncated
      const rawBody = extractBody(message.payload, { maxBodyLength: Infinity, includeHtml: true });
      const textLength = rawBody.text?.length ?? 0;
      const htmlLength = rawBody.html?.length ?? 0;
      const totalOriginalLength = textLength + htmlLength;

      if ((body.text?.length ?? 0) < textLength || (includeHtml && (body.html?.length ?? 0) < htmlLength)) {
        result.truncated = true;
        result.originalSize = totalOriginalLength;
      }

      return result;
    } catch (error: unknown) {
      throw wrapGmailError(error);
    }
  }

  /**
   * List threads. Returns only what threads.list provides (id, snippet, historyId)
   * -- one HTTP call, no per-thread enrichment fan-out.
   * Callers that need Subject/From/Date should call getThread on specific items.
   */
  async function listThreads(
    mcpUserId: string,
    query?: string,
    maxResults: number = 20,
    pageToken?: string,
    email?: string
  ): Promise<ThreadResult> {
    const gmail = await getGmailClient(mcpUserId, email);

    try {
      const response = await gmail.users.threads.list({
        userId: 'me',
        q: query,
        maxResults,
        pageToken,
      });

      const threads: ThreadListItem[] = (response.data.threads ?? [])
        .filter(t => t.id)
        .map(t => ({
          id: t.id!,
          snippet: t.snippet ?? undefined,
          historyId: t.historyId ?? undefined,
        }));

      return {
        threads,
        nextPageToken: response.data.nextPageToken ?? undefined,
      };
    } catch (error: unknown) {
      throw wrapGmailError(error);
    }
  }

  /**
   * List threads with lightweight header enrichment (Subject, From, Date).
   *
   * Calls threads.list for IDs, then threads.get with format=metadata on each
   * thread to pull the first message's Subject and the last message's From/Date.
   * Enrichment failures are swallowed per-thread so the overall list still returns.
   */
  const TRIAGE_ENRICHMENT_CONCURRENCY = 3;

  async function listThreadsEnriched(
    mcpUserId: string,
    query?: string,
    maxResults: number = 20,
    pageToken?: string,
    email?: string
  ): Promise<ThreadResult> {
    const result = await listThreads(mcpUserId, query, maxResults, pageToken, email);
    if (result.threads.length === 0) return result;

    const enriched = await mapWithConcurrency(
      result.threads,
      TRIAGE_ENRICHMENT_CONCURRENCY,
      async (thread) => {
        try {
          const messages = await withGmailConcurrency(() =>
            withRetry(() => getThread(mcpUserId, thread.id, 'metadata', email)),
          );
          if (messages.length === 0) return thread;

          const first = messages[0]!;
          const last = messages[messages.length - 1]!;

          return {
            ...thread,
            subject: first.headers.subject,
            from: last.headers.from,
            date: last.headers.date,
            messageCount: messages.length,
          };
        } catch {
          return thread;
        }
      },
    );

    return { threads: enriched, nextPageToken: result.nextPageToken };
  }

  /**
   * Get a thread with all messages.
   */
  async function getThread(
    mcpUserId: string,
    threadId: string,
    format: 'metadata' | 'full' = 'metadata',
    email?: string
  ): Promise<Array<MessageMetadata | MessageFull>> {
    const gmail = await getGmailClient(mcpUserId, email);

    try {
      const response = await gmail.users.threads.get({
        userId: 'me',
        id: threadId,
        format: format === 'full' ? 'full' : 'metadata',
        metadataHeaders: ['From', 'To', 'Subject', 'Date'],
      });

      return (response.data.messages ?? []).map(message => {
        const headers = extractHeaders(message.payload?.headers ?? []);
        const attachments = extractAttachments(message.payload);

        const metadata: MessageMetadata = {
          id: message.id!,
          threadId: message.threadId!,
          snippet: message.snippet ?? '',
          headers,
          attachments,
        };

        if (format === 'full') {
          const body = extractBody(message.payload);
          return { ...metadata, body } as MessageFull;
        }

        return metadata;
      });
    } catch (error: unknown) {
      throw wrapGmailError(error);
    }
  }

  /**
   * Get attachment metadata.
   */
  async function getAttachmentMetadata(
    mcpUserId: string,
    messageId: string,
    attachmentId: string,
    email?: string
  ): Promise<AttachmentMetadata | null> {
    const gmail = await getGmailClient(mcpUserId, email);

    try {
      const response = await gmail.users.messages.attachments.get({
        userId: 'me',
        messageId,
        id: attachmentId,
      });

      // Get message to find attachment details
      const message = await gmail.users.messages.get({
        userId: 'me',
        id: messageId,
        format: 'metadata',
      });

      const attachments = extractAttachments(message.data.payload);
      const attachment = attachments.find(a => a.attachmentId === attachmentId);

      if (!attachment) {
        return null;
      }

      return {
        ...attachment,
        size: response.data.size ?? attachment.size,
      };
    } catch (error: unknown) {
      throw wrapGmailError(error);
    }
  }

  /**
   * Get threadIds for a list of messageIds.
   * Used by archive operations to convert message-level to thread-level operations.
   * Uses parallel fetching for improved performance.
   */
  async function getThreadIdsForMessages(
    mcpUserId: string,
    messageIds: string[],
    email?: string
  ): Promise<string[]> {
    const gmail = await getGmailClient(mcpUserId, email);

    const results = await mapWithConcurrency(messageIds, MAX_CONCURRENT_GMAIL_CALLS, async (messageId) => {
        try {
          const response = await gmail.users.messages.get({
            userId: 'me',
            id: messageId,
            format: 'minimal',
            fields: 'threadId',
          });
          return response.data.threadId ?? null;
        } catch (error) {
          // If message not found, skip it (it may have been deleted)
          const gmailError = error as { code?: number };
          if (gmailError.code !== 404) {
            throw wrapGmailError(error);
          }
          return null;
        }
      });

    // Collect unique threadIds, filtering out nulls
    const threadIds = new Set<string>();
    for (const threadId of results) {
      if (threadId) {
        threadIds.add(threadId);
      }
    }

    return Array.from(threadIds);
  }

  /**
   * Check if user has the required scope.
   */
  async function checkScope(mcpUserId: string, requiredScope: string, email?: string): Promise<void> {
    const credentials = await getValidCredentials(mcpUserId, email);
    const grantedScopes = new Set(credentials.scope.split(' '));

    if (grantedScopes.has(requiredScope)) return;

    // gmail.modify is a superset that includes labels, compose, and readonly
    if (grantedScopes.has(GMAIL_MODIFY_SCOPE)) return;

    throw new InsufficientScopeError(requiredScope);
  }

  /**
   * Modify labels on a single message.
   */
  async function modifyMessage(
    mcpUserId: string,
    messageId: string,
    addLabels: string[],
    removeLabels: string[],
    email?: string
  ): Promise<ModifyResult> {
    await checkScope(mcpUserId, GMAIL_LABELS_SCOPE, email);
    const gmail = await getGmailClient(mcpUserId, email);

    try {
      await gmail.users.messages.modify({
        userId: 'me',
        id: messageId,
        requestBody: {
          addLabelIds: addLabels,
          removeLabelIds: removeLabels,
        },
      });
      return { id: messageId, success: true };
    } catch (error: unknown) {
      const wrapped = wrapGmailError(error);
      return { id: messageId, success: false, error: wrapped.message };
    }
  }

  /**
   * Modify labels on a thread (affects all messages in thread).
   */
  async function modifyThread(
    mcpUserId: string,
    threadId: string,
    addLabels: string[],
    removeLabels: string[],
    email?: string
  ): Promise<ModifyResult> {
    await checkScope(mcpUserId, GMAIL_LABELS_SCOPE, email);
    const gmail = await getGmailClient(mcpUserId, email);

    try {
      await gmail.users.threads.modify({
        userId: 'me',
        id: threadId,
        requestBody: {
          addLabelIds: addLabels,
          removeLabelIds: removeLabels,
        },
      });
      return { id: threadId, success: true };
    } catch (error: unknown) {
      const wrapped = wrapGmailError(error);
      return { id: threadId, success: false, error: wrapped.message };
    }
  }

  /**
   * Batch modify labels on messages and/or threads.
   *
   * Thread vs Message semantics:
   * - messageIds: Modifies only the specified messages
   * - threadIds: Modifies ALL messages in the specified threads
   *
   * For inbox operations (archive/unarchive), prefer threadIds because Gmail's
   * inbox view is thread-based. A thread appears in inbox if ANY message has the INBOX label.
   */
  async function batchModify(
    mcpUserId: string,
    messageIds: string[] | undefined,
    threadIds: string[] | undefined,
    addLabels: string[],
    removeLabels: string[],
    email?: string
  ): Promise<BatchModifyResult> {
    await checkScope(mcpUserId, GMAIL_MODIFY_SCOPE, email);
    const gmail = await getGmailClient(mcpUserId, email);

    const promises: Promise<ModifyResult>[] = [];

    if (messageIds) {
      for (const id of messageIds) {
        promises.push(
          gmail.users.messages.modify({
            userId: 'me',
            id,
            requestBody: { addLabelIds: addLabels, removeLabelIds: removeLabels },
          }).then(() => ({ id, success: true as const }))
            .catch((error: unknown) => ({ id, success: false as const, error: wrapGmailError(error).message }))
        );
      }
    }

    if (threadIds) {
      for (const id of threadIds) {
        promises.push(
          gmail.users.threads.modify({
            userId: 'me',
            id,
            requestBody: { addLabelIds: addLabels, removeLabelIds: removeLabels },
          }).then(() => ({ id, success: true as const }))
            .catch((error: unknown) => ({ id, success: false as const, error: wrapGmailError(error).message }))
        );
      }
    }

    const results = await mapWithConcurrency(promises, MAX_CONCURRENT_GMAIL_CALLS, async (promise) => promise);

    return {
      results,
      successCount: results.filter(r => r.success).length,
      failureCount: results.filter(r => !r.success).length,
    };
  }

  // Convenience methods for common operations

  /**
   * Archive messages and/or threads (remove INBOX label).
   *
   * IMPORTANT: Gmail's inbox is thread-based. A thread remains in the inbox if ANY
   * message in that thread has the INBOX label. For reliable archiving, prefer using
   * threadIds or use the MCP tool with archiveEntireThread=true (default).
   */
  async function archiveMessages(
    mcpUserId: string,
    messageIds?: string[],
    threadIds?: string[],
    email?: string
  ): Promise<BatchModifyResult> {
    return batchModify(mcpUserId, messageIds, threadIds, [], ['INBOX'], email);
  }

  /**
   * Unarchive messages and/or threads (add INBOX label).
   *
   * For consistency, prefer using threadIds or use the MCP tool with
   * archiveEntireThread=true (default) to restore entire conversations.
   */
  async function unarchiveMessages(
    mcpUserId: string,
    messageIds?: string[],
    threadIds?: string[],
    email?: string
  ): Promise<BatchModifyResult> {
    return batchModify(mcpUserId, messageIds, threadIds, ['INBOX'], [], email);
  }

  async function markAsRead(
    mcpUserId: string,
    messageIds?: string[],
    threadIds?: string[],
    email?: string
  ): Promise<BatchModifyResult> {
    return batchModify(mcpUserId, messageIds, threadIds, [], ['UNREAD'], email);
  }

  async function markAsUnread(
    mcpUserId: string,
    messageIds?: string[],
    threadIds?: string[],
    email?: string
  ): Promise<BatchModifyResult> {
    return batchModify(mcpUserId, messageIds, threadIds, ['UNREAD'], [], email);
  }

  async function starMessages(
    mcpUserId: string,
    messageIds?: string[],
    threadIds?: string[],
    email?: string
  ): Promise<BatchModifyResult> {
    return batchModify(mcpUserId, messageIds, threadIds, ['STARRED'], [], email);
  }

  async function unstarMessages(
    mcpUserId: string,
    messageIds?: string[],
    threadIds?: string[],
    email?: string
  ): Promise<BatchModifyResult> {
    return batchModify(mcpUserId, messageIds, threadIds, [], ['STARRED'], email);
  }

  // ============== LABEL METHODS ==============

  /**
   * Get detailed information about a single label including message counts.
   * One HTTP call per invocation -- use this for targeted label stats.
   */
  async function getLabelInfo(mcpUserId: string, labelId: string, email?: string): Promise<LabelDetailedInfo> {
    const gmail = await getGmailClient(mcpUserId, email);

    try {
      const response = await gmail.users.labels.get({
        userId: 'me',
        id: labelId,
      });

      const label = response.data;
      return {
        id: label.id!,
        name: label.name!,
        type: label.type ?? undefined,
        messagesTotal: label.messagesTotal ?? 0,
        messagesUnread: label.messagesUnread ?? 0,
        threadsTotal: label.threadsTotal ?? 0,
        threadsUnread: label.threadsUnread ?? 0,
      };
    } catch (error: unknown) {
      throw wrapGmailError(error);
    }
  }

  /**
   * List all labels. Returns only what labels.list provides (id, name, type)
   * -- one HTTP call, no per-label fan-out.
   * Use getLabelInfo for message counts on a specific label.
   */
  async function listLabels(mcpUserId: string, email?: string): Promise<LabelInfo[]> {
    const gmail = await getGmailClient(mcpUserId, email);

    try {
      const response = await gmail.users.labels.list({
        userId: 'me',
      });

      return (response.data.labels ?? [])
        .filter(label => label.id)
        .map(label => ({
          id: label.id!,
          name: label.name ?? label.id!,
          type: label.type ?? undefined,
        }));
    } catch (error: unknown) {
      throw wrapGmailError(error);
    }
  }

  /**
   * Add labels to messages/threads.
   */
  async function addLabels(
    mcpUserId: string,
    messageIds: string[] | undefined,
    threadIds: string[] | undefined,
    labelIds: string[],
    email?: string
  ): Promise<BatchModifyResult> {
    return batchModify(mcpUserId, messageIds, threadIds, labelIds, [], email);
  }

  /**
   * Remove labels from messages/threads.
   */
  async function removeLabels(
    mcpUserId: string,
    messageIds: string[] | undefined,
    threadIds: string[] | undefined,
    labelIds: string[],
    email?: string
  ): Promise<BatchModifyResult> {
    return batchModify(mcpUserId, messageIds, threadIds, [], labelIds, email);
  }

  /**
   * Create a new label.
   */
  async function createLabel(
    mcpUserId: string,
    name: string,
    email?: string
  ): Promise<{ id: string; name: string }> {
    await checkScope(mcpUserId, GMAIL_LABELS_SCOPE, email);
    const gmail = await getGmailClient(mcpUserId, email);

    try {
      const response = await gmail.users.labels.create({
        userId: 'me',
        requestBody: {
          name,
          labelListVisibility: 'labelShow',
          messageListVisibility: 'show',
        },
      });

      return {
        id: response.data.id!,
        name: response.data.name!,
      };
    } catch (error: unknown) {
      throw wrapGmailError(error);
    }
  }

  // ============== DRAFT METHODS ==============

  /**
   * Create a new draft email.
   * When replyToMessageId is provided, the draft will be threaded as a reply
   * with proper In-Reply-To and References headers.
   */
  async function createDraft(
    mcpUserId: string,
    to: string | string[],
    subject: string,
    body: string,
    options?: {
      cc?: string | string[];
      bcc?: string | string[];
      isHtml?: boolean;
      replyToMessageId?: string;
    },
    email?: string
  ): Promise<{ draftId: string; messageId: string }> {
    await checkScope(mcpUserId, GMAIL_COMPOSE_SCOPE, email);
    const gmail = await getGmailClient(mcpUserId, email);

    // Threading headers for replies
    let inReplyTo = '';
    let references = '';
    let threadId: string | undefined;

    if (options?.replyToMessageId) {
      // Fetch the original message to get threading headers
      const originalMsg = await gmail.users.messages.get({
        userId: 'me',
        id: options.replyToMessageId,
        format: 'metadata',
        metadataHeaders: ['Message-ID', 'References'],
      });

      threadId = originalMsg.data.threadId ?? undefined;

      // Extract Message-ID from original
      const messageIdHeader = originalMsg.data.payload?.headers?.find(
        (h) => h.name?.toLowerCase() === 'message-id'
      );
      if (messageIdHeader?.value) {
        inReplyTo = messageIdHeader.value;

        // Build References: existing references + original message-id
        const referencesHeader = originalMsg.data.payload?.headers?.find(
          (h) => h.name?.toLowerCase() === 'references'
        );
        if (referencesHeader?.value) {
          references = `${referencesHeader.value} ${inReplyTo}`;
        } else {
          references = inReplyTo;
        }
      }
    }

    // Build email headers
    const toAddrs = Array.isArray(to) ? to.join(', ') : to;
    const ccAddrs = options?.cc ? (Array.isArray(options.cc) ? options.cc.join(', ') : options.cc) : '';
    const bccAddrs = options?.bcc ? (Array.isArray(options.bcc) ? options.bcc.join(', ') : options.bcc) : '';

    const contentType = options?.isHtml ? 'text/html' : 'text/plain';

    // Build raw email
    let rawEmail = `To: ${toAddrs}\r\n`;
    if (ccAddrs) rawEmail += `Cc: ${ccAddrs}\r\n`;
    if (bccAddrs) rawEmail += `Bcc: ${bccAddrs}\r\n`;
    if (inReplyTo) rawEmail += `In-Reply-To: ${inReplyTo}\r\n`;
    if (references) rawEmail += `References: ${references}\r\n`;
    rawEmail += `Subject: ${subject}\r\n`;
    rawEmail += `Content-Type: ${contentType}; charset=utf-8\r\n\r\n`;
    rawEmail += body;

    // Base64url encode
    const encodedEmail = Buffer.from(rawEmail).toString('base64url');

    try {
      const response = await gmail.users.drafts.create({
        userId: 'me',
        requestBody: {
          message: {
            raw: encodedEmail,
            threadId,
          },
        },
      });

      return {
        draftId: response.data.id!,
        messageId: response.data.message?.id ?? '',
      };
    } catch (error: unknown) {
      throw wrapGmailError(error);
    }
  }

  /**
   * Send an email message directly.
   * Supports replies via replyToMessageId (preserves threading).
   */
  async function sendMessage(
    mcpUserId: string,
    to: string | string[],
    subject: string,
    body: string,
    options?: {
      cc?: string | string[];
      bcc?: string | string[];
      isHtml?: boolean;
      replyToMessageId?: string;
    },
    email?: string
  ): Promise<{ messageId: string; threadId: string }> {
    await checkScope(mcpUserId, GMAIL_COMPOSE_SCOPE, email);
    const gmail = await getGmailClient(mcpUserId, email);

    let inReplyTo = '';
    let references = '';
    let threadId: string | undefined;

    if (options?.replyToMessageId) {
      const originalMsg = await gmail.users.messages.get({
        userId: 'me',
        id: options.replyToMessageId,
        format: 'metadata',
        metadataHeaders: ['Message-ID', 'References'],
      });

      threadId = originalMsg.data.threadId ?? undefined;

      const messageIdHeader = originalMsg.data.payload?.headers?.find(
        (h) => h.name?.toLowerCase() === 'message-id'
      );
      if (messageIdHeader?.value) {
        inReplyTo = messageIdHeader.value;
        const referencesHeader = originalMsg.data.payload?.headers?.find(
          (h) => h.name?.toLowerCase() === 'references'
        );
        references = referencesHeader?.value
          ? `${referencesHeader.value} ${inReplyTo}`
          : inReplyTo;
      }
    }

    const toAddrs = Array.isArray(to) ? to.join(', ') : to;
    const ccAddrs = options?.cc ? (Array.isArray(options.cc) ? options.cc.join(', ') : options.cc) : '';
    const bccAddrs = options?.bcc ? (Array.isArray(options.bcc) ? options.bcc.join(', ') : options.bcc) : '';
    const contentType = options?.isHtml ? 'text/html' : 'text/plain';

    let rawEmail = `To: ${toAddrs}\r\n`;
    if (ccAddrs) rawEmail += `Cc: ${ccAddrs}\r\n`;
    if (bccAddrs) rawEmail += `Bcc: ${bccAddrs}\r\n`;
    if (inReplyTo) rawEmail += `In-Reply-To: ${inReplyTo}\r\n`;
    if (references) rawEmail += `References: ${references}\r\n`;
    rawEmail += `Subject: ${subject}\r\n`;
    rawEmail += `Content-Type: ${contentType}; charset=utf-8\r\n\r\n`;
    rawEmail += body;

    const encodedEmail = Buffer.from(rawEmail).toString('base64url');

    try {
      const response = await gmail.users.messages.send({
        userId: 'me',
        requestBody: {
          raw: encodedEmail,
          threadId,
        },
      });

      return {
        messageId: response.data.id!,
        threadId: response.data.threadId!,
      };
    } catch (error: unknown) {
      throw wrapGmailError(error);
    }
  }

  /**
   * List drafts. Returns only what drafts.list provides (id, message.id)
   * -- one HTTP call, no per-draft fan-out.
   * Use getDraft for full content on a specific draft.
   */
  async function listDrafts(
    mcpUserId: string,
    maxResults: number = 20,
    pageToken?: string,
    email?: string
  ): Promise<DraftListResult> {
    const gmail = await getGmailClient(mcpUserId, email);

    try {
      const response = await gmail.users.drafts.list({
        userId: 'me',
        maxResults,
        pageToken,
      });

      const drafts: DraftInfo[] = (response.data.drafts ?? [])
        .filter(draft => draft.id)
        .map(draft => ({
          id: draft.id!,
          messageId: draft.message?.id ?? '',
        }));

      return {
        drafts,
        nextPageToken: response.data.nextPageToken ?? undefined,
      };
    } catch (error: unknown) {
      throw wrapGmailError(error);
    }
  }

  /**
   * Get a draft with full content.
   */
  async function getDraft(mcpUserId: string, draftId: string, email?: string): Promise<DraftContent> {
    const gmail = await getGmailClient(mcpUserId, email);

    try {
      const response = await gmail.users.drafts.get({
        userId: 'me',
        id: draftId,
        format: 'full',
      });

      const message = response.data.message;
      const headers = message?.payload?.headers ?? [];
      const subjectHeader = headers.find((h: gmail_v1.Schema$MessagePartHeader) => h.name?.toLowerCase() === 'subject');
      const toHeader = headers.find((h: gmail_v1.Schema$MessagePartHeader) => h.name?.toLowerCase() === 'to');

      return {
        id: response.data.id!,
        messageId: message?.id ?? '',
        snippet: message?.snippet ?? '',
        subject: subjectHeader?.value ?? undefined,
        to: toHeader?.value ?? undefined,
        body: extractBody(message?.payload),
      };
    } catch (error: unknown) {
      throw wrapGmailError(error);
    }
  }

  /**
   * Update a draft.
   * When replyToMessageId is provided, the draft will be threaded as a reply
   * with proper In-Reply-To and References headers.
   */
  async function updateDraft(
    mcpUserId: string,
    draftId: string,
    to: string | string[],
    subject: string,
    body: string,
    options?: {
      cc?: string | string[];
      bcc?: string | string[];
      isHtml?: boolean;
      replyToMessageId?: string;
    },
    email?: string
  ): Promise<{ draftId: string; messageId: string }> {
    await checkScope(mcpUserId, GMAIL_COMPOSE_SCOPE, email);
    const gmail = await getGmailClient(mcpUserId, email);

    // Threading headers for replies
    let inReplyTo = '';
    let references = '';
    let threadId: string | undefined;

    if (options?.replyToMessageId) {
      // Fetch the original message to get threading headers
      const originalMsg = await gmail.users.messages.get({
        userId: 'me',
        id: options.replyToMessageId,
        format: 'metadata',
        metadataHeaders: ['Message-ID', 'References'],
      });

      threadId = originalMsg.data.threadId ?? undefined;

      // Extract Message-ID from original
      const messageIdHeader = originalMsg.data.payload?.headers?.find(
        (h) => h.name?.toLowerCase() === 'message-id'
      );
      if (messageIdHeader?.value) {
        inReplyTo = messageIdHeader.value;

        // Build References: existing references + original message-id
        const referencesHeader = originalMsg.data.payload?.headers?.find(
          (h) => h.name?.toLowerCase() === 'references'
        );
        if (referencesHeader?.value) {
          references = `${referencesHeader.value} ${inReplyTo}`;
        } else {
          references = inReplyTo;
        }
      }
    }

    const toAddrs = Array.isArray(to) ? to.join(', ') : to;
    const ccAddrs = options?.cc ? (Array.isArray(options.cc) ? options.cc.join(', ') : options.cc) : '';
    const bccAddrs = options?.bcc ? (Array.isArray(options.bcc) ? options.bcc.join(', ') : options.bcc) : '';
    const contentType = options?.isHtml ? 'text/html' : 'text/plain';

    let rawEmail = `To: ${toAddrs}\r\n`;
    if (ccAddrs) rawEmail += `Cc: ${ccAddrs}\r\n`;
    if (bccAddrs) rawEmail += `Bcc: ${bccAddrs}\r\n`;
    if (inReplyTo) rawEmail += `In-Reply-To: ${inReplyTo}\r\n`;
    if (references) rawEmail += `References: ${references}\r\n`;
    rawEmail += `Subject: ${subject}\r\n`;
    rawEmail += `Content-Type: ${contentType}; charset=utf-8\r\n\r\n`;
    rawEmail += body;

    const encodedEmail = Buffer.from(rawEmail).toString('base64url');

    try {
      const response = await gmail.users.drafts.update({
        userId: 'me',
        id: draftId,
        requestBody: {
          message: {
            raw: encodedEmail,
            threadId,
          },
        },
      });

      return {
        draftId: response.data.id!,
        messageId: response.data.message?.id ?? '',
      };
    } catch (error: unknown) {
      throw wrapGmailError(error);
    }
  }

  /**
   * Send an existing draft. Removes the draft and sends its content.
   */
  async function sendDraft(
    mcpUserId: string,
    draftId: string,
    email?: string
  ): Promise<{ messageId: string; threadId: string }> {
    await checkScope(mcpUserId, GMAIL_COMPOSE_SCOPE, email);
    const gmail = await getGmailClient(mcpUserId, email);

    try {
      const response = await gmail.users.drafts.send({
        userId: 'me',
        requestBody: { id: draftId },
      });

      return {
        messageId: response.data.id!,
        threadId: response.data.threadId!,
      };
    } catch (error: unknown) {
      throw wrapGmailError(error);
    }
  }

  /**
   * Trash messages or threads (move to Trash).
   * Returns per-item status so callers can distinguish success, already_trashed, not_found, and errors.
   */
  async function trashMessages(
    mcpUserId: string,
    messageIds?: string[],
    threadIds?: string[],
    email?: string
  ): Promise<BatchModifyResult> {
    await checkScope(mcpUserId, GMAIL_MODIFY_SCOPE, email);
    const gmail = await getGmailClient(mcpUserId, email);

    const promises: Promise<ModifyResult>[] = [];

    if (messageIds?.length) {
      for (const id of messageIds) {
        promises.push(
          gmail.users.messages.trash({ userId: 'me', id })
            .then(() => ({ id, success: true as const }))
            .catch((error: unknown) => classifyOrganizeError(id, error))
        );
      }
    }
    if (threadIds?.length) {
      for (const id of threadIds) {
        promises.push(
          gmail.users.threads.trash({ userId: 'me', id })
            .then(() => ({ id, success: true as const }))
            .catch((error: unknown) => classifyOrganizeError(id, error))
        );
      }
    }

    const results = await mapWithConcurrency(promises, MAX_CONCURRENT_GMAIL_CALLS, async (promise) => promise);
    return {
      results,
      successCount: results.filter(r => r.success).length,
      failureCount: results.filter(r => !r.success).length,
    };
  }

  /**
   * Untrash messages or threads (move out of Trash back to inbox).
   * Returns per-item status so callers can distinguish success, not_found, and errors.
   */
  async function untrashMessages(
    mcpUserId: string,
    messageIds?: string[],
    threadIds?: string[],
    email?: string
  ): Promise<BatchModifyResult> {
    await checkScope(mcpUserId, GMAIL_MODIFY_SCOPE, email);
    const gmail = await getGmailClient(mcpUserId, email);

    const promises: Promise<ModifyResult>[] = [];

    if (messageIds?.length) {
      for (const id of messageIds) {
        promises.push(
          gmail.users.messages.untrash({ userId: 'me', id })
            .then(() => ({ id, success: true as const }))
            .catch((error: unknown) => classifyOrganizeError(id, error))
        );
      }
    }
    if (threadIds?.length) {
      for (const id of threadIds) {
        promises.push(
          gmail.users.threads.untrash({ userId: 'me', id })
            .then(() => ({ id, success: true as const }))
            .catch((error: unknown) => classifyOrganizeError(id, error))
        );
      }
    }

    const results = await mapWithConcurrency(promises, MAX_CONCURRENT_GMAIL_CALLS, async (promise) => promise);
    return {
      results,
      successCount: results.filter(r => r.success).length,
      failureCount: results.filter(r => !r.success).length,
    };
  }

  /**
   * Delete a draft.
   */
  async function deleteDraft(mcpUserId: string, draftId: string, email?: string): Promise<{ success: boolean }> {
    await checkScope(mcpUserId, GMAIL_COMPOSE_SCOPE, email);
    const gmail = await getGmailClient(mcpUserId, email);

    try {
      await gmail.users.drafts.delete({
        userId: 'me',
        id: draftId,
      });

      return { success: true };
    } catch (error: unknown) {
      throw wrapGmailError(error);
    }
  }

  // ============== ACCOUNT MANAGEMENT METHODS ==============

  /**
   * List all connected Gmail accounts for the user.
   */
  async function listAccounts(mcpUserId: string): Promise<AccountInfo[]> {
    return tokenStore.listAccounts(mcpUserId);
  }

  /**
   * Set which account is the default for the user.
   */
  async function setDefaultAccount(mcpUserId: string, accountEmail: string): Promise<void> {
    await tokenStore.setDefaultAccount(mcpUserId, accountEmail);
  }

  /**
   * Remove a specific Gmail account.
   */
  async function removeAccount(mcpUserId: string, accountEmail: string): Promise<void> {
    await tokenStore.deleteCredentials(mcpUserId, accountEmail);
  }

  // Wrap Gmail API methods with concurrency limiter + transient-error retry
  function limited<TArgs extends unknown[], TResult>(
    fn: (...args: TArgs) => Promise<TResult>
  ): (...args: TArgs) => Promise<TResult> {
    return (...args: TArgs) => withGmailConcurrency(() => withRetry(() => fn(...args)));
  }

  return {
    getValidCredentials,
    searchMessages: limited(searchMessages),
    batchSearchMessages: limited(batchSearchMessages),
    getMessage: limited(getMessage),
    listThreads: limited(listThreads),
    listThreadsEnriched,
    getThread: limited(getThread),
    getAttachmentMetadata: limited(getAttachmentMetadata),
    getThreadIdsForMessages: limited(getThreadIdsForMessages),
    // Modification methods
    checkScope,
    modifyMessage: limited(modifyMessage),
    modifyThread: limited(modifyThread),
    batchModify: limited(batchModify),
    archiveMessages: limited(archiveMessages),
    unarchiveMessages: limited(unarchiveMessages),
    markAsRead: limited(markAsRead),
    markAsUnread: limited(markAsUnread),
    starMessages: limited(starMessages),
    unstarMessages: limited(unstarMessages),
    // Label methods
    getLabelInfo: limited(getLabelInfo),
    listLabels: limited(listLabels),
    addLabels: limited(addLabels),
    removeLabels: limited(removeLabels),
    createLabel: limited(createLabel),
    // Send methods
    sendMessage: limited(sendMessage),
    // Trash methods
    trashMessages: limited(trashMessages),
    untrashMessages: limited(untrashMessages),
    // Draft methods
    sendDraft: limited(sendDraft),
    createDraft: limited(createDraft),
    listDrafts: limited(listDrafts),
    getDraft: limited(getDraft),
    updateDraft: limited(updateDraft),
    deleteDraft: limited(deleteDraft),
    // Account management methods (no rate limiting — SQLite only)
    listAccounts,
    setDefaultAccount,
    removeAccount,
  };
}

// Helper functions

function extractHeaders(headers: gmail_v1.Schema$MessagePartHeader[]): MessageMetadata['headers'] {
  const result: MessageMetadata['headers'] = {};

  for (const header of headers) {
    const name = header.name?.toLowerCase();
    const value = header.value;

    if (name === 'from') result.from = value ?? undefined;
    if (name === 'to') result.to = value ?? undefined;
    if (name === 'subject') result.subject = value ?? undefined;
    if (name === 'date') result.date = value ?? undefined;
  }

  return result;
}

function extractAttachments(payload: gmail_v1.Schema$MessagePart | undefined): AttachmentMetadata[] {
  const attachments: AttachmentMetadata[] = [];

  if (!payload) return attachments;

  function processPartRecursive(part: gmail_v1.Schema$MessagePart) {
    if (part.body?.attachmentId && part.filename) {
      attachments.push({
        attachmentId: part.body.attachmentId,
        filename: part.filename,
        mimeType: part.mimeType ?? 'application/octet-stream',
        size: part.body.size ?? 0,
      });
    }

    if (part.parts) {
      for (const subPart of part.parts) {
        processPartRecursive(subPart);
      }
    }
  }

  processPartRecursive(payload);
  return attachments;
}

interface ExtractBodyOptions {
  maxBodyLength?: number;
  includeHtml?: boolean;
}

function extractBody(
  payload: gmail_v1.Schema$MessagePart | undefined,
  options?: ExtractBodyOptions
): { text?: string; html?: string } {
  const result: { text?: string; html?: string } = {};
  const maxLength = options?.maxBodyLength ?? Infinity;
  const includeHtml = options?.includeHtml ?? true;

  if (!payload) return result;

  function processPartRecursive(part: gmail_v1.Schema$MessagePart) {
    const mimeType = part.mimeType ?? '';
    const data = part.body?.data;

    if (data) {
      const decoded = Buffer.from(data, 'base64').toString('utf-8');

      if (mimeType === 'text/plain' && !result.text) {
        result.text = decoded.length > maxLength
          ? decoded.slice(0, maxLength) + '... [truncated]'
          : decoded;
      } else if (mimeType === 'text/html' && !result.html && includeHtml) {
        result.html = decoded.length > maxLength
          ? decoded.slice(0, maxLength) + '... [truncated]'
          : decoded;
      }
    }

    if (part.parts) {
      for (const subPart of part.parts) {
        processPartRecursive(subPart);
      }
    }
  }

  processPartRecursive(payload);
  return result;
}

function classifyOrganizeError(id: string, error: unknown): ModifyResult {
  const gmailError = error as { code?: number; message?: string };
  const status = gmailError.code ?? 500;
  if (status === 404) {
    return { id, success: false, error: 'not_found' };
  }
  if (status === 400) {
    const msg = (gmailError.message ?? '').toLowerCase();
    if (msg.includes('already') || msg.includes('invalid')) {
      return { id, success: false, error: 'already_in_state' };
    }
  }
  return { id, success: false, error: gmailError.message ?? 'unknown_error' };
}

function wrapGmailError(error: unknown): GmailApiError {
  if (error instanceof GmailApiError || error instanceof NotAuthorizedError) {
    throw error;
  }

  const gmailError = error as { code?: number; message?: string };
  const statusCode = gmailError.code ?? 500;
  const message = gmailError.message ?? 'Gmail API error';

  return new GmailApiError(message, statusCode);
}

export type GmailClient = ReturnType<typeof createGmailClientFactory>;
