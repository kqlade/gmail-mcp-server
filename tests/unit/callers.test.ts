import { describe, it, expect } from 'vitest';
import { parseCallers, callersFromEnv, callerForToken, callerById } from '../../src/config.js';
import { resolveCaller, createStartToken, checkStartToken, bearerToken } from '../../src/auth/bearer.js';

const OLA_TOKEN = 'ola-token-0123456789abcdefghij';
const OLA_WRITE_TOKEN = 'ola-write-0123456789abcdefghij';
const SAM_TOKEN = 'sam-token-0123456789abcdefghij';

describe('callers', () => {
  it('parses per-person entries with optional pinned accounts', () => {
    const callers = parseCallers(`ola:${OLA_TOKEN}:Ola@QualifiedIntelligence.com,\n sam:${SAM_TOKEN}`);
    expect(callers).toEqual([
      {
        id: 'ola',
        token: OLA_TOKEN,
        account: 'ola@qualifiedintelligence.com',
        capabilities: ['read', 'write'],
      },
      { id: 'sam', token: SAM_TOKEN, capabilities: ['read', 'write'] },
    ]);
    expect(parseCallers(undefined)).toEqual([]);
  });

  it('rejects malformed, short, duplicate, or reused entries', () => {
    expect(() => parseCallers('ola')).toThrow(/id:token/);
    expect(() => parseCallers('ola:short')).toThrow(/at least 24/);
    expect(() => parseCallers('Ola Kolade:' + OLA_TOKEN)).toThrow(/lowercase handle/);
    expect(() => parseCallers(`ola:${OLA_TOKEN},ola:${SAM_TOKEN}`)).toThrow(/capability .* more than once/);
    expect(() => parseCallers(`ola:${OLA_TOKEN},sam:${OLA_TOKEN}`)).toThrow(/reuses/);
    expect(() => parseCallers(`ola:${OLA_TOKEN}:not-an-email`)).toThrow(/email/);
    expect(() => parseCallers(`ola:${OLA_TOKEN}:a@b.c:read:extra`)).toThrow(/id:token/);
    expect(() => parseCallers(`ola:${OLA_TOKEN}::admin`)).toThrow(/capabilities/);
    expect(() => parseCallers(`ola:${OLA_TOKEN}:a@b.c:read+read`)).toThrow(/repeats/);
    expect(() => parseCallers(
      `ola:${OLA_TOKEN}:a@b.c:read,ola:${OLA_WRITE_TOKEN}:other@b.c:write`
    )).toThrow(/conflicting pinned accounts/);
  });

  it('does not echo token material in malformed-entry errors', () => {
    let message = '';
    try {
      parseCallers(`ola:${OLA_TOKEN}:ola@example.com:read:extra`);
    } catch (error) {
      message = (error as Error).message;
    }
    expect(message).toContain('invalid entry 1');
    expect(message).not.toContain(OLA_TOKEN);
  });

  it('allows separate read and write credentials for one mailbox identity', () => {
    const callers = parseCallers(
      `ola:${OLA_TOKEN}:ola@example.com:read,` +
      `ola:${OLA_WRITE_TOKEN}:ola@example.com:write`
    );
    expect(callers).toEqual([
      {
        id: 'ola',
        token: OLA_TOKEN,
        account: 'ola@example.com',
        capabilities: ['read'],
      },
      {
        id: 'ola',
        token: OLA_WRITE_TOKEN,
        account: 'ola@example.com',
        capabilities: ['write'],
      },
    ]);
    expect(callerForToken(callers, OLA_TOKEN)?.capabilities).toEqual(['read']);
    expect(callerForToken(callers, OLA_WRITE_TOKEN)?.capabilities).toEqual(['write']);
    expect(callerById(callers, 'ola')?.account).toBe('ola@example.com');
  });

  it('keeps the legacy single token as caller "primary", pinned by GMAIL_ACCOUNT', () => {
    const callers = callersFromEnv({ MCP_AUTH_TOKEN: OLA_TOKEN, GMAIL_ACCOUNT: 'Ola@qualifiedintelligence.com' });
    expect(callers).toEqual([{
      id: 'primary',
      token: OLA_TOKEN,
      account: 'ola@qualifiedintelligence.com',
      capabilities: ['read', 'write'],
    }]);
    expect(callersFromEnv({ MCP_AUTH_TOKEN: OLA_TOKEN })).toEqual([{
      id: 'primary',
      token: OLA_TOKEN,
      capabilities: ['read', 'write'],
    }]);
    expect(callersFromEnv({})).toEqual([]);
    expect(() => callersFromEnv({ MCP_AUTH_TOKEN: OLA_TOKEN, MCP_AUTH_TOKENS: `sam:${OLA_TOKEN}` })).toThrow(/reuses/);
    const overlapping = callersFromEnv({
      MCP_AUTH_TOKEN: OLA_TOKEN,
      MCP_AUTH_TOKENS:
        `primary:${SAM_TOKEN}:ola@example.com:read,` +
        `primary:${OLA_WRITE_TOKEN}:ola@example.com:write`,
    });
    expect(overlapping).toHaveLength(3);
    expect(overlapping[2]).toEqual({
      id: 'primary',
      token: OLA_TOKEN,
      account: 'ola@example.com',
      capabilities: ['read', 'write'],
    });
    expect(() => callersFromEnv({
      MCP_AUTH_TOKEN: OLA_TOKEN,
      GMAIL_ACCOUNT: 'other@example.com',
      MCP_AUTH_TOKENS: `primary:${SAM_TOKEN}:ola@example.com:read`,
    })).toThrow(/conflicts/);
  });

  it('resolves the caller from the bearer token alone', () => {
    const callers = parseCallers(`ola:${OLA_TOKEN},sam:${SAM_TOKEN}`);
    expect(callerForToken(callers, OLA_TOKEN)?.id).toBe('ola');
    expect(callerForToken(callers, SAM_TOKEN)?.id).toBe('sam');
    expect(callerForToken(callers, 'nope')).toBeNull();
    expect(callerForToken(callers, '')).toBeNull();
    expect(resolveCaller(`Bearer ${SAM_TOKEN}`, callers)?.id).toBe('sam');
    expect(resolveCaller(`bearer   ${OLA_TOKEN}`, callers)?.id).toBe('ola');
    expect(resolveCaller(`Basic ${OLA_TOKEN}`, callers)).toBeNull();
    expect(resolveCaller(undefined, callers)).toBeNull();
    expect(bearerToken('Bearer x')).toBe('x');
    expect(callerById(callers, 'sam')?.token).toBe(SAM_TOKEN);
    expect(callerById(callers, 'zed')).toBeNull();
  });
});

describe('start tokens', () => {
  const callers = parseCallers(`ola:${OLA_TOKEN},sam:${SAM_TOKEN}`);
  const [ola, sam] = callers;

  it('binds an OAuth start link to the caller that minted it', () => {
    const now = 1_700_000_000_000;
    const link = createStartToken(ola!, now);
    expect(link.caller).toBe('ola');
    expect(checkStartToken(link, callers, now + 1000)?.id).toBe('ola');
    // Expired, tampered, or re-addressed links fail.
    expect(checkStartToken(link, callers, now + 11 * 60 * 1000)).toBeNull();
    expect(checkStartToken({ ...link, caller: 'sam' }, callers, now + 1000)).toBeNull();
    expect(checkStartToken({ ...link, exp: String(now + 60_000) }, callers, now + 1000)).toBeNull();
    expect(checkStartToken({ ...link, sig: createStartToken(sam!, now).sig }, callers, now + 1000)).toBeNull();
    expect(checkStartToken({ exp: link.exp, sig: link.sig }, callers, now + 1000)).toBeNull();
    expect(checkStartToken({ ...link, caller: 'zed' }, callers, now + 1000)).toBeNull();
  });

  it('does not leak the secret into the link', () => {
    const link = createStartToken(sam!);
    expect(JSON.stringify(link)).not.toContain(SAM_TOKEN);
  });

  it('verifies the credential that signed a duplicate-id capability token', () => {
    const callers = parseCallers(
      `ola:${OLA_TOKEN}:ola@example.com:read,` +
      `ola:${OLA_WRITE_TOKEN}:ola@example.com:write`
    );
    const writeCaller = callers[1]!;
    const now = 1_700_000_000_000;
    const readLink = createStartToken(callers[0]!, now);
    const link = createStartToken(writeCaller, now);
    expect(checkStartToken(readLink, callers, now + 1000)).toBeNull();
    expect(checkStartToken(link, callers, now + 1000)?.token).toBe(OLA_WRITE_TOKEN);
  });
});
