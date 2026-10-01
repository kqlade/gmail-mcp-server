import { config as loadEnv } from 'dotenv';
import { createHash, timingSafeEqual } from 'node:crypto';
import { z } from 'zod';

// Load .env file
loadEnv();

/**
 * A caller is one bearer token and the Gmail identity it unlocks.
 *
 * Multi-person deployment: each person's Ripple compartment holds its own
 * token (MCP_GMAIL_API_KEY in that profile's .env) and the token alone decides
 * whose mailbox a request touches. Nothing in the request body can name
 * another person. `account`, when set, pins the Google account that caller
 * may connect and operate on: agents were hallucinating values for a per-tool
 * `email` parameter and burning sessions on "account not connected" retry
 * loops, so the account is resolved server-side and the parameter is gone.
 * Without a pin the caller operates on whichever account they connected.
 */
export interface Caller {
  /** Stable identity rows are keyed by (mcpUserId in the token store). */
  id: string;
  /** Bearer token this caller presents. */
  token: string;
  /** Google account the caller is allowed to connect and operate on, if pinned. */
  account?: string;
  /** Server-enforced tool capabilities carried by this credential. */
  capabilities: Capability[];
}

export type Capability = 'read' | 'write';
export const ALL_CAPABILITIES: Capability[] = ['read', 'write'];
/** Identity the legacy single-token deployment stored its rows under. */
export const LEGACY_CALLER_ID = 'primary';
const MIN_TOKEN_LENGTH = 24;
const CALLER_ID = /^[a-z0-9][a-z0-9_-]{0,63}$/;
const EMAIL_ADDRESS = /^[^@\s]+@[^@\s]+\.[^@\s]+$/;
const TOKEN_PLACEHOLDER = /^(?:generate|replace|change|your)(?:-|_)/i;

/**
 * Parse MCP_AUTH_TOKENS: entries separated by commas or newlines, each exactly
 * `id:token:account:read` or `id:token:account:write`. The same id may appear
 * once per capability so one pinned mailbox can have separate read and write
 * credentials. Anything less explicit fails closed.
 */
export function parseCallers(raw: string | undefined): Caller[] {
  const callers: Caller[] = [];
  const seenTokens = new Set<string>();
  const accountById = new Map<string, string | undefined>();
  const capabilitiesById = new Map<string, Set<Capability>>();
  const entries = (raw ?? '').split(/[\n,]/).map((s) => s.trim()).filter(Boolean);
  for (const [entryIndex, entry] of entries.entries()) {
    const fields = entry.split(':').map((s) => s.trim());
    const [id, token, account, capabilityText] = fields;
    if (fields.length !== 4 || !id || !token || !account || !capabilityText) {
      throw new Error(
        `MCP_AUTH_TOKENS entry must be id:token:account:read or id:token:account:write ` +
        `(invalid entry ${entryIndex + 1})`
      );
    }
    const normalizedId = id.toLowerCase();
    if (!CALLER_ID.test(normalizedId)) {
      throw new Error(`MCP_AUTH_TOKENS caller id ${JSON.stringify(id)} must be a lowercase handle`);
    }
    if (token.length < MIN_TOKEN_LENGTH) {
      throw new Error(`MCP_AUTH_TOKENS token for ${normalizedId} must be at least ${MIN_TOKEN_LENGTH} characters`);
    }
    if (TOKEN_PLACEHOLDER.test(token)) {
      throw new Error(`MCP_AUTH_TOKENS token for ${normalizedId} is a placeholder; generate a random credential`);
    }
    if (seenTokens.has(token)) throw new Error(`MCP_AUTH_TOKENS reuses a token (${normalizedId})`);
    seenTokens.add(token);
    if (capabilityText !== 'read' && capabilityText !== 'write') {
      throw new Error(`MCP_AUTH_TOKENS capability for ${normalizedId} must be exactly read or write`);
    }
    const capability = capabilityText as Capability;

    const normalizedAccount = account.toLowerCase();
    if (!EMAIL_ADDRESS.test(normalizedAccount)) {
      throw new Error(`MCP_AUTH_TOKENS account for ${normalizedId} is not an email address`);
    }
    if (accountById.has(normalizedId) && accountById.get(normalizedId) !== normalizedAccount) {
      throw new Error(`MCP_AUTH_TOKENS gives ${normalizedId} conflicting pinned accounts`);
    }
    accountById.set(normalizedId, normalizedAccount);

    const priorCapabilities = capabilitiesById.get(normalizedId) ?? new Set<Capability>();
    if (priorCapabilities.has(capability)) {
      throw new Error(
        `MCP_AUTH_TOKENS lists ${normalizedId} capability ${capability} more than once`
      );
    }
    priorCapabilities.add(capability);
    capabilitiesById.set(normalizedId, priorCapabilities);

    const caller: Caller = {
      id: normalizedId,
      token,
      account: normalizedAccount,
      capabilities: [capability],
    };
    callers.push(caller);
  }
  return callers;
}

const configSchema = z.object({
  // Server
  port: z.coerce.number().int().positive().default(3000),
  baseUrl: z.string().url(),

  // Google OAuth
  googleClientId: z.string().min(1),
  googleClientSecret: z.string().min(1),
  oauthRedirectUri: z.string().url(),

  // Security
  tokenEncryptionKey: z.string().min(32, 'TOKEN_ENCRYPTION_KEY must be at least 32 characters'),
  // Who may call: MCP_AUTH_TOKENS (per person) and/or the legacy single
  // MCP_AUTH_TOKEN (caller "primary", pinned by required GMAIL_ACCOUNT).
  callers: z.array(z.custom<Caller>()).min(1, 'Set MCP_AUTH_TOKENS (id:token:account:read|write, comma-separated) or MCP_AUTH_TOKEN with GMAIL_ACCOUNT'),

  // Database
  dbUrl: z.string().default('./data/gmail-mcp.db'),

  // CORS
  allowedOrigins: z.string().optional().transform((val) =>
    val ? val.split(',').map(s => s.trim()).filter(Boolean) : []
  ),
});

export type Config = z.infer<typeof configSchema>;

/** Callers from the environment, allowing temporary legacy/scoped cutover overlap. */
export function callersFromEnv(env: NodeJS.ProcessEnv = process.env): Caller[] {
  const callers = parseCallers(env['MCP_AUTH_TOKENS']);
  const legacyToken = (env['MCP_AUTH_TOKEN'] ?? '').trim();
  if (legacyToken) {
    if (legacyToken.length < MIN_TOKEN_LENGTH) {
      throw new Error(`MCP_AUTH_TOKEN must be at least ${MIN_TOKEN_LENGTH} characters`);
    }
    if (TOKEN_PLACEHOLDER.test(legacyToken)) {
      throw new Error('MCP_AUTH_TOKEN is a placeholder; generate a random credential');
    }
    if (callers.some((c) => c.token === legacyToken)) {
      throw new Error('MCP_AUTH_TOKEN reuses a token from MCP_AUTH_TOKENS');
    }
    const primaryCallers = callers.filter((c) => c.id === LEGACY_CALLER_ID);
    const configuredPin = (env['GMAIL_ACCOUNT'] ?? '').trim().toLowerCase() || undefined;
    const capabilityPin = primaryCallers[0]?.account;
    if (
      (configuredPin && !EMAIL_ADDRESS.test(configuredPin)) ||
      (!configuredPin && !capabilityPin)
    ) {
      throw new Error('Legacy MCP_AUTH_TOKEN requires GMAIL_ACCOUNT to pin the only mailbox it may access');
    }
    if (
      primaryCallers.length > 0 &&
      configuredPin !== undefined &&
      configuredPin !== capabilityPin
    ) {
      throw new Error(
        'GMAIL_ACCOUNT conflicts with the pinned account on MCP_AUTH_TOKENS caller "primary"'
      );
    }
    const legacy: Caller = {
      id: LEGACY_CALLER_ID,
      token: legacyToken,
      capabilities: [...ALL_CAPABILITIES],
    };
    // Temporary overlap makes a zero-downtime cutover possible: add scoped
    // `primary` credentials, switch clients, then remove MCP_AUTH_TOKEN. All
    // credentials resolve the existing `primary` database rows.
    legacy.account = capabilityPin ?? configuredPin;
    callers.push(legacy);
  }
  return callers;
}

/** The caller a bearer token belongs to, compared in constant time across all callers. */
export function callerForToken(callers: Caller[], token: string): Caller | null {
  if (!token) return null;
  const presented = Buffer.from(token);
  let found: Caller | null = null;
  for (const caller of callers) {
    const expected = Buffer.from(caller.token);
    if (expected.length === presented.length && timingSafeEqual(expected, presented)) {
      found = caller;
    }
  }
  return found;
}

export function callerById(callers: Caller[], id: string): Caller | null {
  return callers.find((c) => c.id === id) ?? null;
}

/** Stable, non-secret identifier for one exact bearer credential. */
export function callerCredentialId(caller: Caller): string {
  return createHash('sha256').update(caller.token).digest('base64url');
}

export function callerByCredentialId(callers: Caller[], id: string, credentialId: string): Caller | null {
  return callers.find(
    (caller) => caller.id === id && callerCredentialId(caller) === credentialId
  ) ?? null;
}

export function callerCan(caller: Caller, capability: Capability): boolean {
  return caller.capabilities.includes(capability);
}

function loadConfig(): Config {
  let callers: Caller[] = [];
  let callersError: string | null = null;
  try {
    callers = callersFromEnv();
  } catch (error) {
    callersError = (error as Error).message;
  }
  const env = {
    port: process.env['PORT'],
    baseUrl: process.env['BASE_URL'],
    googleClientId: process.env['GOOGLE_CLIENT_ID'],
    googleClientSecret: process.env['GOOGLE_CLIENT_SECRET'],
    oauthRedirectUri: process.env['OAUTH_REDIRECT_URI'],
    tokenEncryptionKey: process.env['TOKEN_ENCRYPTION_KEY'],
    callers,
    dbUrl: process.env['DB_URL'],
    allowedOrigins: process.env['ALLOWED_ORIGINS'],
  };

  const result = configSchema.safeParse(env);
  if (callersError) {
    throw new Error(`Configuration validation failed:\n  callers: ${callersError}`);
  }

  if (!result.success) {
    const errors = result.error.issues.map(
      (issue) => `  ${issue.path.join('.')}: ${issue.message}`
    );
    throw new Error(`Configuration validation failed:\n${errors.join('\n')}`);
  }

  return result.data;
}

// Lazy initialization - config is loaded on first access
let _config: Config | null = null;

export function getConfig(): Config {
  if (!_config) {
    _config = loadConfig();
  }
  return _config;
}

// For testing - allows resetting config
export function resetConfig(): void {
  _config = null;
}
