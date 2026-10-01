import { describe, it, expect } from 'vitest';
import {
  parseCallers,
  callersFromEnv,
  callerForToken,
  callerById,
  callerByCredentialId,
  callerCredentialId,
} from '../../src/config.js';
import { resolveCaller, bearerToken } from '../../src/auth/bearer.js';

const OLA_TOKEN = 'ola-token-0123456789abcdefghij';
const OLA_WRITE_TOKEN = 'ola-write-0123456789abcdefghij';
const SAM_TOKEN = 'sam-token-0123456789abcdefghij';

describe('callers', () => {
  it('parses exact, pinned, single-capability entries', () => {
    const callers = parseCallers(
      `ola:${OLA_TOKEN}:Ola@QualifiedIntelligence.com:read,\n` +
      `sam:${SAM_TOKEN}:sam@example.com:write`
    );
    expect(callers).toEqual([
      {
        id: 'ola',
        token: OLA_TOKEN,
        account: 'ola@qualifiedintelligence.com',
        capabilities: ['read'],
      },
      {
        id: 'sam',
        token: SAM_TOKEN,
        account: 'sam@example.com',
        capabilities: ['write'],
      },
    ]);
    expect(parseCallers(undefined)).toEqual([]);
  });

  it('rejects malformed, short, duplicate, or reused entries', () => {
    expect(() => parseCallers('ola')).toThrow(/id:token/);
    expect(() => parseCallers('ola:short:ola@example.com:read')).toThrow(/at least 24/);
    expect(() => parseCallers(`Ola Kolade:${OLA_TOKEN}:ola@example.com:read`)).toThrow(/lowercase handle/);
    expect(() => parseCallers(
      `ola:${OLA_TOKEN}:ola@example.com:read,ola:${SAM_TOKEN}:ola@example.com:read`
    )).toThrow(/capability .* more than once/);
    expect(() => parseCallers(
      `ola:${OLA_TOKEN}:ola@example.com:read,sam:${OLA_TOKEN}:sam@example.com:write`
    )).toThrow(/reuses/);
    expect(() => parseCallers(`ola:${OLA_TOKEN}:not-an-email:read`)).toThrow(/email/);
    expect(() => parseCallers(`ola:${OLA_TOKEN}:a@b.c:read:extra`)).toThrow(/id:token/);
    expect(() => parseCallers(`ola:${OLA_TOKEN}::read`)).toThrow(/id:token/);
    expect(() => parseCallers(`ola:${OLA_TOKEN}:a@b.c:read+write`)).toThrow(/exactly read or write/);
    expect(() => parseCallers('ola:generate-a-read-token-here:ola@example.com:read')).toThrow(/placeholder/);
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
    expect(() => callersFromEnv({ MCP_AUTH_TOKEN: OLA_TOKEN })).toThrow(/requires GMAIL_ACCOUNT/);
    expect(callersFromEnv({})).toEqual([]);
    expect(() => callersFromEnv({
      MCP_AUTH_TOKEN: OLA_TOKEN,
      GMAIL_ACCOUNT: 'ola@example.com',
      MCP_AUTH_TOKENS: `sam:${OLA_TOKEN}:sam@example.com:read`,
    })).toThrow(/reuses/);
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
    const callers = parseCallers(
      `ola:${OLA_TOKEN}:ola@example.com:read,sam:${SAM_TOKEN}:sam@example.com:write`
    );
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

  it('identifies the exact credential when an id has read and write siblings', () => {
    const callers = parseCallers(
      `ola:${OLA_TOKEN}:ola@example.com:read,` +
      `ola:${OLA_WRITE_TOKEN}:ola@example.com:write`
    );
    const readId = callerCredentialId(callers[0]!);
    const writeId = callerCredentialId(callers[1]!);
    expect(readId).not.toBe(writeId);
    expect(callerByCredentialId(callers, 'ola', writeId)?.token).toBe(OLA_WRITE_TOKEN);
    expect(callerByCredentialId(callers, 'sam', writeId)).toBeNull();
    expect(callerByCredentialId(callers, 'ola', 'missing')).toBeNull();
  });
});
