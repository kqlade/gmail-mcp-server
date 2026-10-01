/**
 * MCP Server setup using @modelcontextprotocol/sdk
 *
 * Gmail MCP tools:
 * - gmail.status / gmail.authorize — auth and connection
 * - gmail.searchMessages — single or batch message search
 * - gmail.getMessage / gmail.getThread / gmail.listThreads — read
 * - gmail.getAttachmentMetadata / gmail.listLabels / gmail.getLabelInfo — metadata
 * - gmail.organizeMessages — archive, star, trash, label, read/unread (batched)
 * - gmail.sendMessage — send email
 * - gmail.manageDraft — create, get, update, delete, send, list drafts
 * - gmail.createLabel — create custom labels
 * - gmail.listAccounts — account visibility (read-only)
 *
 * Every tool operates on the account of the caller the bearer token named
 * (see Caller in config.ts; a caller may be pinned to one Google account).
 * There is deliberately no per-call account parameter and no
 * account-mutation tool: agent callers proved they will hallucinate email
 * addresses, and a mutation tool would let a confused agent disconnect the
 * real account.
 */

import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { StreamableHTTPServerTransport } from '@modelcontextprotocol/sdk/server/streamableHttp.js';
import { z } from 'zod';
import type { IncomingMessage, ServerResponse } from 'node:http';
import { callerById, callerCan, type Caller, type Config } from '../config.js';
import type { TokenStore } from '../store/interface.js';
import { createGmailClientFactory } from '../gmail/client.js';
import { createGoogleOAuth } from '../auth/googleOAuth.js';
import { NotAuthorizedError, GmailApiError, InsufficientScopeError } from '../utils/errors.js';

export interface McpServerDependencies {
  config: Config;
  tokenStore: TokenStore;
}

// Structured search fields shared by searchMessages, listThreads, and batch items.
// The server compiles these into a Gmail query string so the caller doesn't need
// to know Gmail search syntax.
const structuredSearchFields = {
  from: z.string().optional().describe('Sender name or email address (e.g. "github", "notifications@github.com")'),
  to: z.string().optional().describe('Recipient email address'),
  subject: z.string().optional().describe('Words in the subject line'),
  after: z.string().optional().describe('Messages after this date (YYYY-MM-DD)'),
  before: z.string().optional().describe('Messages before this date (YYYY-MM-DD)'),
  label: z.string().optional().describe('Gmail label name to filter by'),
  hasAttachment: z.boolean().optional().describe('Only messages with attachments'),
  scope: z.enum(['default', 'inbox', 'trash', 'spam', 'anywhere']).optional()
    .describe('Where to search. "default" (the default) = everything except trash and spam. "inbox" = only inbox. "trash"/"spam" = only that location. "anywhere" = everything including trash and spam.'),
};

type StructuredSearchParams = {
  from?: string;
  to?: string;
  subject?: string;
  after?: string;
  before?: string;
  label?: string;
  hasAttachment?: boolean;
  scope?: string;
  query?: string;
};

function buildSearchQuery(params: StructuredSearchParams): string {
  const parts: string[] = [];
  if (params.from) parts.push(`from:${params.from}`);
  if (params.to) parts.push(`to:${params.to}`);
  if (params.subject) parts.push(`subject:(${params.subject})`);
  if (params.after) parts.push(`after:${params.after}`);
  if (params.before) parts.push(`before:${params.before}`);
  if (params.label) parts.push(`label:${params.label}`);
  if (params.hasAttachment) parts.push('has:attachment');

  switch (params.scope) {
    case 'inbox': parts.push('in:inbox'); break;
    case 'trash': parts.push('in:trash'); break;
    case 'spam': parts.push('in:spam'); break;
    case 'anywhere': parts.push('in:anywhere'); break;
    // 'default' and undefined: no qualifier — Gmail excludes trash/spam automatically
  }

  if (params.query) parts.push(params.query);
  return parts.join(' ');
}

function formatToolResult<T extends object>(payload: T): {
  content: Array<{ type: 'text'; text: string }>;
  structuredContent: Record<string, unknown>;
} {
  return {
    content: [
      {
        type: 'text' as const,
        text: JSON.stringify(payload),
      },
    ],
    structuredContent: payload as Record<string, unknown>,
  };
}

// Helper to format error responses
function formatError(error: unknown): {
  content: Array<{ type: 'text'; text: string }>;
  structuredContent: Record<string, unknown>;
  isError: true;
} {
  let code = -32603; // Internal error default
  let message = 'An unexpected error occurred';
  let data: Record<string, unknown> | undefined;

  if (error instanceof InsufficientScopeError) {
    code = -32001; // NOT_AUTHORIZED
    message = error.message;
    data = { requiredScope: error.requiredScope };
  } else if (error instanceof NotAuthorizedError) {
    code = -32001; // NOT_AUTHORIZED
    message = error.message;
  } else if (error instanceof GmailApiError) {
    code = -32000; // GMAIL_API_ERROR / RATE_LIMITED
    message = error.httpStatus === 429 ? `Rate limited by Gmail API: ${error.message}` : error.message;
    if (typeof error.httpStatus === 'number') {
      data = { ...data, httpStatus: error.httpStatus };
    }
  } else if (error instanceof Error) {
    message = error.message;
  }

  const payload = { error: message, code, ...(data && { data }) };
  return {
    ...formatToolResult(payload),
    isError: true,
  };
}

export interface McpServerInstance {
  handleRequest: (req: IncomingMessage, res: ServerResponse, body: unknown, caller: Caller) => Promise<void>;
}

function containsKey(value: unknown, key: string): boolean {
  if (!value || typeof value !== 'object') return false;
  if (Array.isArray(value)) return value.some((item) => containsKey(item, key));
  const record = value as Record<string, unknown>;
  return Object.prototype.hasOwnProperty.call(record, key) ||
    Object.values(record).some((item) => containsKey(item, key));
}

function retiredArgumentError(body: unknown): { id: unknown; message: string } | null {
  if (!body || typeof body !== 'object' || Array.isArray(body)) return null;
  const request = body as {
    id?: unknown;
    method?: unknown;
    params?: { arguments?: unknown };
  };
  if (request.method !== 'tools/call') return null;
  const args = request.params?.arguments;
  if (!args || typeof args !== 'object' || Array.isArray(args)) return null;
  if (containsKey(args, 'email')) {
    return {
      id: request.id ?? null,
      message: 'The email argument is retired. This bearer credential is pinned to one mailbox; remove email and retry.',
    };
  }
  if (containsKey(args, 'replyToMessageId')) {
    return {
      id: request.id ?? null,
      message: 'replyToMessageId is retired because write credentials cannot read mailbox metadata. Get the message with the read credential and pass its opaque replyContext.',
    };
  }
  if (containsKey(args, 'archiveEntireThread')) {
    return {
      id: request.id ?? null,
      message: 'archiveEntireThread is retired because write credentials cannot resolve message IDs by reading Gmail. Pass threadIds from the read result, or pass messageIds to modify only those messages.',
    };
  }
  return null;
}

function isJsonRpcNotification(body: unknown): boolean {
  if (!body || typeof body !== 'object' || Array.isArray(body)) return false;
  const request = body as {
    id?: unknown;
    jsonrpc?: unknown;
    method?: unknown;
    params?: unknown;
  };
  const paramsValid =
    request.params === undefined ||
    (request.params !== null && typeof request.params === 'object');
  return (
    !Object.prototype.hasOwnProperty.call(request, 'id') &&
    request.jsonrpc === '2.0' &&
    typeof request.method === 'string' &&
    paramsValid
  );
}

function unsupportedBatchResponses(body: unknown[]): Array<Record<string, unknown>> {
  const responses: Array<Record<string, unknown>> = [];
  for (const item of body) {
    const retired = retiredArgumentError(item);
    if (!item || typeof item !== 'object' || Array.isArray(item)) {
      responses.push({
        jsonrpc: '2.0',
        id: null,
        error: { code: -32600, message: 'Invalid JSON-RPC request' },
      });
      continue;
    }
    const request = item as {
      id?: unknown;
      jsonrpc?: unknown;
      method?: unknown;
      params?: unknown;
    };
    const paramsValid =
      request.params === undefined ||
      (request.params !== null && typeof request.params === 'object');
    const structurallyValid =
      request.jsonrpc === '2.0' &&
      typeof request.method === 'string' &&
      paramsValid;
    // Only a structurally valid notification suppresses its response.
    if (isJsonRpcNotification(request)) {
      continue;
    }
    responses.push({
      jsonrpc: '2.0',
      id: request.id ?? null,
      error: !structurallyValid
        ? { code: -32600, message: 'Invalid JSON-RPC request' }
        : retired
        ? { code: -32602, message: retired.message }
        : {
            code: -32600,
            message: 'JSON-RPC batch requests are not supported; send each request separately',
          },
    });
  }
  return responses;
}

export async function createMcpServer(deps: McpServerDependencies): Promise<McpServerInstance> {
  const { config, tokenStore } = deps;
  const googleOAuth = createGoogleOAuth({ config, tokenStore });

  // Create Gmail client factory (shared across requests; holds the client cache)
  const gmailClient = createGmailClientFactory({
    tokenStore,
    encryptionKey: config.tokenEncryptionKey,
    googleClientId: config.googleClientId,
    googleClientSecret: config.googleClientSecret,
    pinnedAccountFor: (mcpUserId) => callerById(config.callers, mcpUserId)?.account,
  });

  // Builds a fresh McpServer with all tools registered for one caller. In
  // stateless mode a server+transport pair must NOT be shared across
  // concurrent requests (responses cross-wire and hang), so this runs once
  // per request, and the caller the bearer token named is closed over by
  // every tool: no argument can point a call at someone else's mailbox.
  function buildServer(caller: Caller): McpServer {
  const mcpUserId = caller.id;
  const canRead = callerCan(caller, 'read');
  const canWrite = callerCan(caller, 'write');
  // What to call the account in tool text: the pin when there is one.
  const accountLabel = caller.account ?? 'your Google account';
  const server = new McpServer(
    {
      name: 'gmail-mcp',
      version: '1.0.0',
    },
    {
      capabilities: {
        tools: {},
      },
    }
  );

  // Register gmail.status tool
  if (canRead) server.registerTool(
    'gmail.status',
    {
      description: 'Returns whether Gmail is connected for this compartment and which account it operates on',
    },
    async () => {

      try {
        const accounts = await tokenStore.listAccounts(mcpUserId);
        const operating = caller.account
          ? accounts.find(a => a.email.toLowerCase() === caller.account)
          : (accounts.find(a => a.isDefault) ?? accounts[0]);

        if (operating) {
          return formatToolResult({
            authorized: true,
            account: operating.email,
            scopes: operating.scopes,
            connectedAt: operating.connectedAt.toISOString(),
          });
        }

        return formatToolResult({
          authorized: false,
          account: caller.account ?? null,
          message: caller.account
            ? `Gmail account ${caller.account} is not connected. Use gmail.authorize and ` +
              `complete the OAuth flow as ${caller.account}.`
            : 'Gmail is not connected for this compartment. Use gmail.authorize to connect it.',
        });
      } catch (error) {
        return formatError(error);
      }
    }
  );

  // Register gmail.authorize tool
  if (canWrite) server.registerTool(
    'gmail.authorize',
    {
      description: `Initiates the OAuth consent flow to connect ${accountLabel} for this compartment`,
      inputSchema: {
        scopes: z.array(z.enum(['gmail.readonly', 'gmail.labels', 'gmail.modify', 'gmail.compose'])).optional(),
      },
    },
    async (args) => {
      const scopes = args?.scopes ?? ['gmail.readonly'];

      // Create one-time PKCE state from this authenticated write request and
      // return Google's URL directly. There is no browser-accessible start hop.
      const authUrl = await googleOAuth.createAuthorizationUrl(caller, scopes);

      return {
        content: [
          {
            type: 'text' as const,
            text: caller.account
              ? `To connect Gmail, open the following URL in your browser and sign in as ${caller.account} — other accounts are rejected (link valid for 10 minutes):\n\n${authUrl}\n\nAfter authorizing, return here and try your Gmail operation again.`
              : `To connect Gmail, open the following URL in your browser and sign in with the Google account this compartment should use (link valid for 10 minutes):\n\n${authUrl}\n\nAfter authorizing, return here and try your Gmail operation again.`,
          },
        ],
      };
    }
  );

  // Register gmail.searchMessages tool (also handles batch searches via queries array)
  if (canRead) server.registerTool(
    'gmail.searchMessages',
    {
      description:
        'Search messages. Prefer structured fields (from, to, subject, after, before, scope) over raw query.\n' +
        'Two modes:\n' +
        '- Single search: pass structured fields and/or "query" with optional maxResults/pageToken\n' +
        '- Batch search: pass "queries" array to run multiple searches in parallel\n' +
        'For single search, structured fields and "query" are merged. For batch, each item can use structured fields and/or "query".',
      inputSchema: {
        ...structuredSearchFields,
        query: z.string().optional().describe('Raw Gmail query to merge with structured fields (for single search)'),
        queries: z.array(z.object({
          ...structuredSearchFields,
          query: z.string().optional().describe('Raw Gmail query to merge with structured fields'),
          maxResults: z.number().int().min(1).max(100).optional().describe('Max results for this query (default 20)'),
        })).min(1).max(10).optional().describe('Array of search queries for batch search (max 10)'),
        maxResults: z.number().int().min(1).max(100).optional().describe('Maximum results for single search (1-100, default 20)'),
        pageToken: z.string().optional().describe('Token for pagination (single search only)'),
      },
    },
    async (args) => {
      const hasQueries = Array.isArray(args.queries) && args.queries.length > 0;

      try {
        if (hasQueries) {
          const builtQueries = args.queries!.map((q: StructuredSearchParams & { maxResults?: number }) => ({
            query: buildSearchQuery(q),
            maxResults: q.maxResults,
          }));
          const results = await gmailClient.batchSearchMessages(mcpUserId, builtQueries);
          return formatToolResult({
            results,
            queryCount: results.length,
            totalMessages: results.reduce(
              (sum: number, r: { result?: { messages?: unknown[] } }) =>
                sum + (Array.isArray(r.result?.messages) ? r.result.messages.length : 0),
              0,
            ),
            failedQueries: results.filter((r: { error?: string }) => typeof r.error === 'string').length,
          });
        }

        const builtQuery = buildSearchQuery(args as StructuredSearchParams);
        if (!builtQuery.trim()) {
          return {
            ...formatToolResult({
              error: 'Provide at least one search parameter (from, subject, query, etc.)',
              code: -32602,
            }),
            isError: true,
          };
        }

        const result = await gmailClient.searchMessages(
          mcpUserId,
          builtQuery,
          args.maxResults ?? 20,
          args.pageToken
        );
        return formatToolResult(result);
      } catch (error) {
        return formatError(error);
      }
    }
  );

  // Register gmail.getMessage tool
  if (canRead) server.registerTool(
    'gmail.getMessage',
    {
      description:
        'Get a single message by ID. Formats: "metadata" (headers/snippet only, fastest), ' +
        '"summary" (headers + first 2KB of text body, no HTML), "full" (complete message with truncation options). ' +
        'Use "summary" for quick content preview without downloading huge HTML emails.',
      inputSchema: {
        messageId: z.string().describe('The message ID'),
        format: z.enum(['metadata', 'summary', 'full']).optional().describe('Response format: metadata (fastest), summary (2KB text preview), or full (default: metadata)'),
        maxBodyLength: z.number().int().min(100).max(500000).optional().describe('Max characters for body content. Default: 2000 for summary, 50000 for full. Use smaller values for faster responses.'),
        includeHtml: z.boolean().optional().describe('Include HTML body (default: false for summary, true for full). Set false to reduce response size.'),
      },
    },
    async (args) => {

      try {
        const message = await gmailClient.getMessage(
          mcpUserId,
          args.messageId,
          args.format ?? 'metadata',
          {
            maxBodyLength: args.maxBodyLength,
            includeHtml: args.includeHtml,
          }
        );
        return formatToolResult(message);
      } catch (error) {
        return formatError(error);
      }
    }
  );

  // Register gmail.listThreads tool
  if (canRead) server.registerTool(
    'gmail.listThreads',
    {
      description: 'List conversation threads (id, snippet). Use gmail.getThread for full metadata on specific threads. Prefer structured fields (from, subject, scope, etc.) over raw query.',
      inputSchema: {
        ...structuredSearchFields,
        query: z.string().optional().describe('Raw Gmail query to merge with structured fields'),
        maxResults: z.number().int().min(1).max(100).optional().describe('Maximum results (1-100, default 20)'),
        pageToken: z.string().optional().describe('Token for pagination'),
      },
    },
    async (args) => {

      try {
        const builtQuery = buildSearchQuery(args as StructuredSearchParams) || undefined;
        const result = await gmailClient.listThreads(
          mcpUserId,
          builtQuery,
          args.maxResults ?? 20,
          args.pageToken
        );
        return formatToolResult(result);
      } catch (error) {
        return formatError(error);
      }
    }
  );

  // Register gmail.getThread tool
  if (canRead) server.registerTool(
    'gmail.getThread',
    {
      description: 'Get a thread with all its messages',
      inputSchema: {
        threadId: z.string().describe('The thread ID'),
        format: z.enum(['metadata', 'full']).optional().describe('Response format (default: metadata)'),
      },
    },
    async (args) => {

      try {
        const messages = await gmailClient.getThread(
          mcpUserId,
          args.threadId,
          args.format ?? 'metadata'
        );
        return formatToolResult({ messages });
      } catch (error) {
        return formatError(error);
      }
    }
  );

  // Register gmail.getAttachmentMetadata tool
  if (canRead) server.registerTool(
    'gmail.getAttachmentMetadata',
    {
      description: 'Get metadata about an attachment (filename, size, MIME type)',
      inputSchema: {
        messageId: z.string().describe('The message ID'),
        attachmentId: z.string().describe('The attachment ID'),
      },
    },
    async (args) => {

      try {
        const attachment = await gmailClient.getAttachmentMetadata(
          mcpUserId,
          args.messageId,
          args.attachmentId
        );

        if (!attachment) {
          return {
            content: [
              {
                type: 'text' as const,
                text: JSON.stringify({ error: 'Attachment not found', code: -32602 }),
              },
            ],
            isError: true,
          };
        }

        return {
          content: [
            {
              type: 'text' as const,
              text: JSON.stringify(attachment),
            },
          ],
        };
      } catch (error) {
        return formatError(error);
      }
    }
  );

  // ============== TRIAGE SNAPSHOT TOOL ==============

  if (canRead) server.registerTool(
    'gmail.triageSnapshot',
    {
      description:
        'Get a triage-ready inbox snapshot in one call. Returns thread IDs with subject, ' +
        'sender, date, snippet, and message count for up to maxResults inbox threads. ' +
        'Use pageToken to paginate through large inboxes.',
      inputSchema: {
        ...structuredSearchFields,
        query: z.string().optional().describe('Raw Gmail query to merge with structured fields'),
        maxResults: z.number().int().min(1).max(50).optional().describe('Max threads to return (default 25, max 50)'),
        pageToken: z.string().optional().describe('Pagination token from a previous triageSnapshot call'),
      },
    },
    async (args) => {

      try {
        const builtQuery = buildSearchQuery({
          ...(args as StructuredSearchParams),
          scope: (args as StructuredSearchParams).scope ?? 'inbox',
        }) || 'in:inbox';

        const result = await gmailClient.listThreadsEnriched(
          mcpUserId,
          builtQuery,
          args.maxResults ?? 25,
          args.pageToken
        );

        return formatToolResult({
          threadCount: result.threads.length,
          hasMore: Boolean(result.nextPageToken),
          nextPageToken: result.nextPageToken ?? undefined,
          threads: result.threads,
        });
      } catch (error) {
        return formatError(error);
      }
    }
  );

  // ============== ORGANIZE TOOL ==============

  const ORGANIZE_ACTIONS = [
    'archive', 'unarchive',
    'mark_read', 'mark_unread',
    'star', 'unstar',
    'trash', 'untrash',
    'add_labels', 'remove_labels',
  ] as const;

  type OrganizeAction = (typeof ORGANIZE_ACTIONS)[number];

  if (canWrite) server.registerTool(
    'gmail.organizeMessages',
    {
      description:
        'Apply one or more organize actions to messages/threads in a single call. ' +
        'Supported actions: archive, unarchive, mark_read, mark_unread, star, unstar, trash, untrash, add_labels, remove_labels. ' +
        'Each action item specifies its own messageIds/threadIds. Pass threadIds from a read result to archive or unarchive a whole conversation.',
      inputSchema: {
        actions: z.array(z.object({
          action: z.enum(ORGANIZE_ACTIONS).describe('The organize operation to perform'),
          messageIds: z.array(z.string()).optional().describe('Message IDs to act on'),
          threadIds: z.array(z.string()).optional().describe('Thread IDs to act on'),
          labelIds: z.array(z.string()).optional().describe('Label IDs (required for add_labels/remove_labels)'),
        })).min(1).describe('Array of organize actions to apply'),
      },
    },
    async (args) => {
      const results: Array<{ action: OrganizeAction; ok: boolean; result?: unknown; error?: string }> = [];

      for (const item of args.actions) {
        if (!item.messageIds?.length && !item.threadIds?.length) {
          results.push({ action: item.action, ok: false, error: 'At least one messageId or threadId must be provided' });
          continue;
        }

        try {
          const { messageIds, threadIds } = item;

          let result: unknown;
          switch (item.action) {
            case 'archive':
              result = await gmailClient.archiveMessages(mcpUserId, messageIds, threadIds);
              break;
            case 'unarchive':
              result = await gmailClient.unarchiveMessages(mcpUserId, messageIds, threadIds);
              break;
            case 'mark_read':
              result = await gmailClient.markAsRead(mcpUserId, messageIds, threadIds);
              break;
            case 'mark_unread':
              result = await gmailClient.markAsUnread(mcpUserId, messageIds, threadIds);
              break;
            case 'star':
              result = await gmailClient.starMessages(mcpUserId, messageIds, threadIds);
              break;
            case 'unstar':
              result = await gmailClient.unstarMessages(mcpUserId, messageIds, threadIds);
              break;
            case 'trash':
              result = await gmailClient.trashMessages(mcpUserId, messageIds, threadIds);
              break;
            case 'untrash':
              result = await gmailClient.untrashMessages(mcpUserId, messageIds, threadIds);
              break;
            case 'add_labels':
              if (!item.labelIds?.length) {
                results.push({ action: item.action, ok: false, error: 'labelIds required for add_labels' });
                continue;
              }
              result = await gmailClient.addLabels(mcpUserId, messageIds, threadIds, item.labelIds);
              break;
            case 'remove_labels':
              if (!item.labelIds?.length) {
                results.push({ action: item.action, ok: false, error: 'labelIds required for remove_labels' });
                continue;
              }
              result = await gmailClient.removeLabels(mcpUserId, messageIds, threadIds, item.labelIds);
              break;
          }
          const batchResult = result as { failureCount?: number } | undefined;
          const hasFailures = typeof batchResult?.failureCount === 'number' && batchResult.failureCount > 0;
          results.push({ action: item.action, ok: !hasFailures, result });
        } catch (error) {
          const message = error instanceof Error ? error.message : 'Unknown error';
          results.push({ action: item.action, ok: false, error: message });
        }
      }

      return {
        content: [{ type: 'text' as const, text: JSON.stringify({ results, actionCount: results.length }) }],
      };
    }
  );

  // Register gmail.getLabelInfo tool
  if (canRead) server.registerTool(
    'gmail.getLabelInfo',
    {
      description: 'Get information about a label including message counts. Common labels: INBOX, UNREAD, STARRED, SENT, DRAFT, TRASH, SPAM.',
      inputSchema: {
        labelId: z.string().describe('The label ID (e.g., "INBOX", "UNREAD", "STARRED", or custom label ID)'),
      },
    },
    async (args) => {

      try {
        const result = await gmailClient.getLabelInfo(mcpUserId, args.labelId);
        return formatToolResult(result);
      } catch (error) {
        return formatError(error);
      }
    }
  );

  // Register gmail.listLabels tool
  if (canRead) server.registerTool(
    'gmail.listLabels',
    {
      description: 'List all labels (id, name, type). Use gmail.getLabelInfo for message counts on a specific label.',
    },
    async () => {

      try {
        const result = await gmailClient.listLabels(mcpUserId);
        return formatToolResult({ labels: result });
      } catch (error) {
        return formatError(error);
      }
    }
  );

  // Register gmail.createLabel tool
  if (canWrite) server.registerTool(
    'gmail.createLabel',
    {
      description: 'Create a new custom label. Requires gmail.labels scope.',
      inputSchema: {
        name: z.string().describe('Name for the new label'),
      },
    },
    async (args) => {

      try {
        const result = await gmailClient.createLabel(mcpUserId, args.name);
        return { content: [{ type: 'text' as const, text: JSON.stringify(result) }] };
      } catch (error) {
        return formatError(error);
      }
    }
  );

  // Register gmail.sendMessage tool
  if (canWrite) server.registerTool(
    'gmail.sendMessage',
    {
      description: 'Send an email message. For replies, pass the opaque replyContext returned by gmail.getMessage on the read surface. Requires gmail.compose scope.',
      inputSchema: {
        to: z.union([z.string(), z.array(z.string())]).describe('Recipient email address(es)'),
        subject: z.string().describe('Email subject'),
        body: z.string().describe('Email body content'),
        cc: z.union([z.string(), z.array(z.string())]).optional().describe('CC recipient(s)'),
        bcc: z.union([z.string(), z.array(z.string())]).optional().describe('BCC recipient(s)'),
        isHtml: z.boolean().optional().describe('Whether body is HTML (default: false, plain text)'),
        replyContext: z.string().optional().describe('Opaque context returned by gmail.getMessage. Preserves threading without giving this write surface mailbox-read access.'),
      },
    },
    async (args) => {

      try {
        const result = await gmailClient.sendMessage(mcpUserId, args.to, args.subject, args.body, {
          cc: args.cc,
          bcc: args.bcc,
          isHtml: args.isHtml,
          replyContext: args.replyContext,
        });
        return { content: [{ type: 'text' as const, text: JSON.stringify(result) }] };
      } catch (error) {
        return formatError(error);
      }
    }
  );

  // ============== DRAFT LIFECYCLE TOOL ==============

  const DRAFT_ACTIONS = ['create', 'get', 'update', 'delete', 'send', 'list'] as const;
  const READ_ONLY_DRAFT_ACTIONS = ['get', 'list'] as const;
  const WRITE_ONLY_DRAFT_ACTIONS = ['create', 'update', 'delete', 'send'] as const;
  const availableDraftActions = canRead && canWrite
    ? DRAFT_ACTIONS
    : canRead
      ? READ_ONLY_DRAFT_ACTIONS
      : WRITE_ONLY_DRAFT_ACTIONS;

  const recipientSchema = z.union([z.string(), z.array(z.string())]);

  if (canRead || canWrite) server.registerTool(
    'gmail.manageDraft',
    {
      description: canRead && canWrite
        ? 'Manage draft emails. Actions: create, get, update, delete, send, or list.'
        : canRead
          ? 'Read draft emails. Actions: get or list. Mutations require a write-capability token.'
          : 'Mutate draft emails. Actions: create, update, delete, or send. Reading and listing drafts require a read-capability token.',
      inputSchema: {
        action: z.enum(availableDraftActions).describe('Draft operation to perform'),
        draftId: z.string().optional().describe('Draft ID (required for get/update/delete/send)'),
        to: recipientSchema.optional().describe('Recipient email address(es) (for create/update)'),
        subject: z.string().optional().describe('Email subject (for create/update)'),
        body: z.string().optional().describe('Email body content (for create/update)'),
        cc: recipientSchema.optional().describe('CC recipient(s)'),
        bcc: recipientSchema.optional().describe('BCC recipient(s)'),
        isHtml: z.boolean().optional().describe('Whether body is HTML (default: false)'),
        draftContext: z.string().optional().describe('Required for update: opaque context from gmail.getDraft that preserves omitted recipients, content type, and threading'),
        replyContext: z.string().optional().describe('Opaque context returned by gmail.getMessage (preserves threading without a mailbox read on this surface)'),
        maxResults: z.number().int().min(1).max(100).optional().describe('Max results for list (default 20)'),
        pageToken: z.string().optional().describe('Pagination token for list'),
      },
    },
    async (args) => {
      const composeOpts = { cc: args.cc, bcc: args.bcc, isHtml: args.isHtml, replyContext: args.replyContext };

      try {
        let result: unknown;

        switch (args.action) {
          case 'create':
            if (!args.to || !args.subject || !args.body) {
              return { content: [{ type: 'text' as const, text: JSON.stringify({ error: 'to, subject, and body are required for action=create', code: -32602 }) }], isError: true as const };
            }
            result = await gmailClient.createDraft(mcpUserId, args.to, args.subject, args.body, composeOpts);
            break;
          case 'get':
            if (!args.draftId) {
              return { content: [{ type: 'text' as const, text: JSON.stringify({ error: 'draftId is required for action=get', code: -32602 }) }], isError: true as const };
            }
            result = await gmailClient.getDraft(mcpUserId, args.draftId);
            break;
          case 'update':
            if (!args.draftId || !args.body || !args.draftContext) {
              return { content: [{ type: 'text' as const, text: JSON.stringify({ error: 'draftId, body, and draftContext from gmail.getDraft are required for action=update', code: -32602 }) }], isError: true as const };
            }
            result = await gmailClient.updateDraft(
              mcpUserId,
              args.draftId,
              args.to,
              args.subject,
              args.body,
              { ...composeOpts, draftContext: args.draftContext }
            );
            break;
          case 'delete':
            if (!args.draftId) {
              return { content: [{ type: 'text' as const, text: JSON.stringify({ error: 'draftId is required for action=delete', code: -32602 }) }], isError: true as const };
            }
            result = await gmailClient.deleteDraft(mcpUserId, args.draftId);
            break;
          case 'send':
            if (!args.draftId) {
              return { content: [{ type: 'text' as const, text: JSON.stringify({ error: 'draftId is required for action=send', code: -32602 }) }], isError: true as const };
            }
            result = await gmailClient.sendDraft(mcpUserId, args.draftId);
            break;
          case 'list':
            result = await gmailClient.listDrafts(mcpUserId, args.maxResults ?? 20, args.pageToken);
            break;
        }

        return { content: [{ type: 'text' as const, text: JSON.stringify(result) }] };
      } catch (error) {
        return formatError(error);
      }
    }
  );

  // ============== ACCOUNT VISIBILITY TOOL ==============
  // Read-only. setDefaultAccount/removeAccount were removed on purpose: the
  // operating account is resolved from the caller, so switching is meaningless
  // and disconnecting would let a confused agent brick email access.

  if (canRead) server.registerTool(
    'gmail.listAccounts',
    {
      description:
        `List connected Gmail accounts for this compartment. All tools operate on ` +
        `${accountLabel}; there is no way to target another account.`,
    },
    async () => {

      try {
        const accounts = await gmailClient.listAccounts(mcpUserId);
        const pinnedConnected = caller.account
          ? accounts.some(a => a.email.toLowerCase() === caller.account)
          : accounts.length > 0;

        return formatToolResult({
          pinnedAccount: caller.account ?? null,
          pinnedAccountConnected: pinnedConnected,
          accounts: accounts.map(a => ({
            email: a.email,
            scopes: a.scopes,
            connectedAt: a.connectedAt.toISOString(),
          })),
          count: accounts.length,
        });
      } catch (error) {
        return formatError(error);
      }
    }
  );

  return server;
  }

  // Per-request server + transport, mirroring the SDK's stateless example.
  const handleRequest = async (req: IncomingMessage, res: ServerResponse, body: unknown, caller: Caller) => {
    if (Array.isArray(body)) {
      if (body.length === 0) {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({
          jsonrpc: '2.0',
          id: null,
          error: { code: -32600, message: 'Invalid JSON-RPC request: an empty batch is not allowed' },
        }));
        return;
      }
      const responses = unsupportedBatchResponses(body);
      if (responses.length === 0) {
        res.writeHead(202);
        res.end();
      } else {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify(responses));
      }
      return;
    }
    const retired = retiredArgumentError(body);
    if (retired) {
      if (isJsonRpcNotification(body)) {
        res.writeHead(202);
        res.end();
        return;
      }
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({
        jsonrpc: '2.0',
        id: retired.id,
        error: { code: -32602, message: retired.message },
      }));
      return;
    }
    const server = buildServer(caller);
    const transport = new StreamableHTTPServerTransport({
      sessionIdGenerator: undefined, // Stateless mode
    });

    res.on('close', () => {
      void transport.close();
      void server.close();
    });

    try {
      await server.connect(transport);
      await transport.handleRequest(req, res, body);
    } catch (error) {
      console.error('Error handling MCP request:', error);
      if (!res.headersSent) {
        res.writeHead(500, { 'Content-Type': 'application/json' });
        res.end(
          JSON.stringify({
            jsonrpc: '2.0',
            error: { code: -32603, message: 'Internal server error' },
            id: null,
          })
        );
      }
    }
  };

  return { handleRequest };
}
