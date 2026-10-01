import {
  createCipheriv,
  createDecipheriv,
  createHash,
  randomBytes,
} from 'node:crypto';

const PREFIX = 'dc1.';
const ALGORITHM = 'aes-256-gcm';
const IV_BYTES = 12;
const TAG_BYTES = 16;
const MAX_HEADER_CHARS = 16_384;
const AAD = Buffer.from('gmail-mcp:draft-context:v1');

export interface DraftContextData {
  mcpUserId: string;
  draftId: string;
  messageId: string;
  threadId?: string;
  to?: string;
  cc?: string;
  bcc?: string;
  subject?: string;
  inReplyTo?: string;
  references?: string;
  isHtml: boolean;
}

export function deriveDraftContextKey(encryptionKey: string): Buffer {
  return createHash('sha256')
    .update('gmail-mcp:draft-context:v1\0')
    .update(encryptionKey)
    .digest();
}

function normalizeHeader(value: string | undefined): string | undefined {
  if (value === undefined) return undefined;
  const normalized = value.replace(/[\r\n]+/g, ' ').trim();
  return normalized || undefined;
}

function valid(data: DraftContextData, expectedUserId?: string): boolean {
  const bounded = (value: unknown, required = false): value is string =>
    typeof value === 'string' &&
    (required ? value.length > 0 : true) &&
    value.length <= MAX_HEADER_CHARS &&
    !/[\r\n]/.test(value);
  return (
    (!expectedUserId || data.mcpUserId === expectedUserId) &&
    bounded(data.mcpUserId, true) &&
    data.mcpUserId.length <= 64 &&
    bounded(data.draftId, true) &&
    data.draftId.length <= 1024 &&
    bounded(data.messageId) &&
    data.messageId.length <= 1024 &&
    (data.threadId === undefined || bounded(data.threadId, true)) &&
    (data.to === undefined || bounded(data.to)) &&
    (data.cc === undefined || bounded(data.cc)) &&
    (data.bcc === undefined || bounded(data.bcc)) &&
    (data.subject === undefined || bounded(data.subject)) &&
    (data.inReplyTo === undefined || bounded(data.inReplyTo)) &&
    (data.references === undefined || bounded(data.references)) &&
    typeof data.isHtml === 'boolean'
  );
}

export function createDraftContext(data: DraftContextData, key: Buffer): string {
  const normalized: DraftContextData = {
    ...data,
    ...(normalizeHeader(data.threadId) ? { threadId: normalizeHeader(data.threadId) } : {}),
    ...(normalizeHeader(data.to) ? { to: normalizeHeader(data.to) } : {}),
    ...(normalizeHeader(data.cc) ? { cc: normalizeHeader(data.cc) } : {}),
    ...(normalizeHeader(data.bcc) ? { bcc: normalizeHeader(data.bcc) } : {}),
    ...(normalizeHeader(data.subject) ? { subject: normalizeHeader(data.subject) } : {}),
    ...(normalizeHeader(data.inReplyTo) ? { inReplyTo: normalizeHeader(data.inReplyTo) } : {}),
    ...(normalizeHeader(data.references) ? { references: normalizeHeader(data.references) } : {}),
  };
  if (key.length !== 32 || !valid(normalized)) {
    throw new Error('invalid draft context data');
  }
  const iv = randomBytes(IV_BYTES);
  const cipher = createCipheriv(ALGORITHM, key, iv);
  cipher.setAAD(AAD);
  const encrypted = Buffer.concat([
    cipher.update(JSON.stringify({ v: 1, ...normalized }), 'utf8'),
    cipher.final(),
  ]);
  const token = PREFIX + Buffer.concat([iv, cipher.getAuthTag(), encrypted]).toString('base64url');
  if (token.length > 64_000) {
    throw new Error('draft context data is too large');
  }
  return token;
}

export function openDraftContext(
  token: string,
  mcpUserId: string,
  draftId: string,
  key: Buffer
): DraftContextData {
  try {
    if (!token.startsWith(PREFIX) || token.length > 64_000 || key.length !== 32) {
      throw new Error('invalid envelope');
    }
    const envelope = Buffer.from(token.slice(PREFIX.length), 'base64url');
    if (envelope.length <= IV_BYTES + TAG_BYTES) throw new Error('invalid envelope');
    const decipher = createDecipheriv(ALGORITHM, key, envelope.subarray(0, IV_BYTES));
    decipher.setAAD(AAD);
    decipher.setAuthTag(envelope.subarray(IV_BYTES, IV_BYTES + TAG_BYTES));
    const parsed = JSON.parse(
      Buffer.concat([
        decipher.update(envelope.subarray(IV_BYTES + TAG_BYTES)),
        decipher.final(),
      ]).toString('utf8')
    ) as DraftContextData & { v?: unknown };
    if (parsed.v !== 1) throw new Error('invalid payload');
    const data: DraftContextData = {
      mcpUserId: parsed.mcpUserId,
      draftId: parsed.draftId,
      messageId: parsed.messageId,
      ...(parsed.threadId !== undefined ? { threadId: parsed.threadId } : {}),
      ...(parsed.to !== undefined ? { to: parsed.to } : {}),
      ...(parsed.cc !== undefined ? { cc: parsed.cc } : {}),
      ...(parsed.bcc !== undefined ? { bcc: parsed.bcc } : {}),
      ...(parsed.subject !== undefined ? { subject: parsed.subject } : {}),
      ...(parsed.inReplyTo !== undefined ? { inReplyTo: parsed.inReplyTo } : {}),
      ...(parsed.references !== undefined ? { references: parsed.references } : {}),
      isHtml: parsed.isHtml,
    };
    if (!valid(data, mcpUserId) || data.draftId !== draftId) {
      throw new Error('invalid payload');
    }
    return data;
  } catch {
    throw new Error(
      'draftContext is invalid, stale, or belongs to another compartment; fetch the draft again with the read credential'
    );
  }
}
