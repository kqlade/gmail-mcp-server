import {
  createCipheriv,
  createDecipheriv,
  createHash,
  randomBytes,
} from 'node:crypto';

const PREFIX = 'rc2.';
const ALGORITHM = 'aes-256-gcm';
const IV_BYTES = 12;
const TAG_BYTES = 16;
const MAX_REFERENCES_CHARS = 16_384;
const AAD = Buffer.from('gmail-mcp:reply-context:v2');

export interface ReplyContextData {
  mcpUserId: string;
  messageId: string;
  threadId: string;
  inReplyTo: string;
  references?: string;
}

/** Derive once per server instance; individual contexts only do AES-GCM. */
export function deriveReplyContextKey(encryptionKey: string): Buffer {
  return createHash('sha256')
    .update('gmail-mcp:reply-context:v2\0')
    .update(encryptionKey)
    .digest();
}

function normalizeHeader(value: string): string {
  return value.replace(/[\r\n]+/g, ' ').trim();
}

function boundedReferences(value: string | undefined): string | undefined {
  if (!value) return undefined;
  const normalized = normalizeHeader(value);
  if (normalized.length <= MAX_REFERENCES_CHARS) return normalized || undefined;
  // Keep the newest message IDs (the tail of References) when a very long
  // conversation exceeds the envelope bound.
  const parts = normalized.split(/\s+/);
  const kept: string[] = [];
  let length = 0;
  for (let index = parts.length - 1; index >= 0; index -= 1) {
    const part = parts[index]!;
    const nextLength = length + part.length + (kept.length ? 1 : 0);
    if (nextLength > MAX_REFERENCES_CHARS) break;
    kept.unshift(part);
    length = nextLength;
  }
  return kept.join(' ') || normalized.slice(-MAX_REFERENCES_CHARS);
}

function valid(data: ReplyContextData, expectedUserId?: string): boolean {
  return (
    (!expectedUserId || data.mcpUserId === expectedUserId) &&
    typeof data.mcpUserId === 'string' &&
    data.mcpUserId.length > 0 &&
    data.mcpUserId.length <= 64 &&
    typeof data.messageId === 'string' &&
    data.messageId.length > 0 &&
    data.messageId.length <= 1024 &&
    typeof data.threadId === 'string' &&
    data.threadId.length > 0 &&
    data.threadId.length <= 1024 &&
    typeof data.inReplyTo === 'string' &&
    data.inReplyTo.length > 0 &&
    data.inReplyTo.length <= 4096 &&
    !/[\r\n]/.test(data.inReplyTo) &&
    (data.references === undefined ||
      (typeof data.references === 'string' &&
        data.references.length <= MAX_REFERENCES_CHARS &&
        !/[\r\n]/.test(data.references)))
  );
}

/** Encrypt mailbox metadata so the write surface can reply without reading Gmail. */
export function createReplyContext(data: ReplyContextData, key: Buffer): string {
  const references = boundedReferences(data.references);
  const normalized: ReplyContextData = {
    ...data,
    inReplyTo: normalizeHeader(data.inReplyTo),
    ...(references ? { references } : {}),
  };
  if (key.length !== 32 || !valid(normalized)) {
    throw new Error('invalid reply context data');
  }
  const iv = randomBytes(IV_BYTES);
  const cipher = createCipheriv(ALGORITHM, key, iv);
  cipher.setAAD(AAD);
  const encrypted = Buffer.concat([
    cipher.update(JSON.stringify({ v: 2, ...normalized }), 'utf8'),
    cipher.final(),
  ]);
  const envelope = Buffer.concat([iv, cipher.getAuthTag(), encrypted]);
  const token = PREFIX + envelope.toString('base64url');
  if (token.length > 32_000) {
    throw new Error('reply context data is too large');
  }
  return token;
}

/** Open and validate context for exactly one compartment identity. */
export function openReplyContext(
  token: string,
  mcpUserId: string,
  key: Buffer
): ReplyContextData {
  try {
    if (!token.startsWith(PREFIX) || token.length > 32_000 || key.length !== 32) {
      throw new Error('invalid envelope');
    }
    const envelope = Buffer.from(token.slice(PREFIX.length), 'base64url');
    if (envelope.length <= IV_BYTES + TAG_BYTES) throw new Error('invalid envelope');
    const iv = envelope.subarray(0, IV_BYTES);
    const tag = envelope.subarray(IV_BYTES, IV_BYTES + TAG_BYTES);
    const encrypted = envelope.subarray(IV_BYTES + TAG_BYTES);
    const decipher = createDecipheriv(ALGORITHM, key, iv);
    decipher.setAAD(AAD);
    decipher.setAuthTag(tag);
    const plaintext = Buffer.concat([
      decipher.update(encrypted),
      decipher.final(),
    ]).toString('utf8');
    const parsed = JSON.parse(plaintext) as {
      v?: unknown;
      mcpUserId?: unknown;
      messageId?: unknown;
      threadId?: unknown;
      inReplyTo?: unknown;
      references?: unknown;
    };
    if (
      parsed.v !== 2 ||
      typeof parsed.mcpUserId !== 'string' ||
      typeof parsed.messageId !== 'string' ||
      typeof parsed.threadId !== 'string' ||
      typeof parsed.inReplyTo !== 'string' ||
      (parsed.references !== undefined && typeof parsed.references !== 'string')
    ) {
      throw new Error('invalid payload');
    }
    const data: ReplyContextData = {
      mcpUserId: parsed.mcpUserId,
      messageId: parsed.messageId,
      threadId: parsed.threadId,
      inReplyTo: parsed.inReplyTo,
      ...(parsed.references ? { references: parsed.references } : {}),
    };
    if (!valid(data, mcpUserId)) throw new Error('invalid payload');
    return data;
  } catch {
    throw new Error('replyContext is invalid or belongs to another compartment; fetch the message again with the read credential');
  }
}
