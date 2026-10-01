import { describe, expect, it } from 'vitest';
import { createHash } from 'node:crypto';
import {
  createGoogleOAuth,
  grantedScopesExactlyMatch,
} from '../../src/auth/googleOAuth.js';
import { callerCredentialId, parseCallers, type Config } from '../../src/config.js';
import type { OAuthState, TokenStore } from '../../src/store/interface.js';

const WRITE_TOKEN = 'write-token-0123456789abcdefghij';

describe('OAuth authorization URLs', () => {
  it('requires exact granted scopes, including rejecting broader or unrelated grants', () => {
    const readonly = 'https://www.googleapis.com/auth/gmail.readonly';
    expect(grantedScopesExactlyMatch(['gmail.readonly'], readonly)).toBe(true);
    expect(grantedScopesExactlyMatch(
      ['gmail.readonly'],
      `${readonly} https://mail.google.com/`
    )).toBe(false);
    expect(grantedScopesExactlyMatch(
      ['gmail.readonly'],
      `${readonly} https://www.googleapis.com/auth/drive.readonly`
    )).toBe(false);
    expect(grantedScopesExactlyMatch(['gmail.readonly'], undefined)).toBe(false);
  });

  it('creates Google state bound to the exact write credential and requested scopes', async () => {
    const caller = parseCallers(
      `ola:${WRITE_TOKEN}:ola@example.com:write`
    )[0]!;
    const config: Config = {
      port: 3000,
      baseUrl: 'https://gmail.example',
      googleClientId: 'client.apps.googleusercontent.com',
      googleClientSecret: 'secret',
      oauthRedirectUri: 'https://gmail.example/oauth/callback',
      tokenEncryptionKey: 'x'.repeat(32),
      callers: [caller],
      dbUrl: ':memory:',
      allowedOrigins: [],
    };
    let saved: OAuthState | null = null;
    const tokenStore = {
      saveOAuthState: async (state: OAuthState) => { saved = state; },
    } as unknown as TokenStore;
    const oauth = createGoogleOAuth({ config, tokenStore });

    const url = await oauth.createAuthorizationUrl(
      caller,
      ['gmail.compose', 'gmail.readonly', 'gmail.compose']
    );
    const parsed = new URL(url);
    expect(parsed.hostname).toBe('accounts.google.com');
    expect(parsed.searchParams.get('scope')).toContain('gmail.compose');
    expect(saved).toMatchObject({
      mcpUserId: 'ola',
      credentialId: callerCredentialId(caller),
      scopes: ['gmail.compose', 'gmail.readonly'],
    });
    expect(parsed.searchParams.get('state')).toBe((saved as OAuthState | null)?.state);
    const verifier = (saved as OAuthState | null)?.codeVerifier ?? '';
    expect(parsed.searchParams.get('code_challenge_method')).toBe('S256');
    expect(parsed.searchParams.get('code_challenge')).toBe(
      createHash('sha256').update(verifier).digest('base64url')
    );
  });

  it.each([
    { scopes: [] as string[], label: 'empty' },
    { scopes: ['', '  '], label: 'blank' },
    { scopes: ['gmail.labels'], label: 'labels-only' },
  ])('rejects $label scope requests before saving OAuth state', async ({ scopes }) => {
    const caller = parseCallers(`ola:${WRITE_TOKEN}:ola@example.com:write`)[0]!;
    let saveCount = 0;
    const oauth = createGoogleOAuth({
      config: {
        port: 3000,
        baseUrl: 'https://gmail.example',
        googleClientId: 'client',
        googleClientSecret: 'secret',
        oauthRedirectUri: 'https://gmail.example/oauth/callback',
        tokenEncryptionKey: 'x'.repeat(32),
        callers: [caller],
        dbUrl: ':memory:',
        allowedOrigins: [],
      },
      tokenStore: {
        saveOAuthState: async () => { saveCount += 1; },
      } as unknown as TokenStore,
    });
    await expect(oauth.createAuthorizationUrl(caller, scopes)).rejects.toThrow(/functional Gmail scope/);
    expect(saveCount).toBe(0);
  });

  it('refuses a read credential', async () => {
    const caller = parseCallers(
      `ola:read-token-0123456789abcdefghij:ola@example.com:read`
    )[0]!;
    const oauth = createGoogleOAuth({
      config: {
        port: 3000,
        baseUrl: 'https://gmail.example',
        googleClientId: 'client',
        googleClientSecret: 'secret',
        oauthRedirectUri: 'https://gmail.example/oauth/callback',
        tokenEncryptionKey: 'x'.repeat(32),
        callers: [caller],
        dbUrl: ':memory:',
        allowedOrigins: [],
      },
      tokenStore: {} as TokenStore,
    });
    await expect(oauth.createAuthorizationUrl(caller, ['gmail.readonly'])).rejects.toThrow(/write-capability/);
  });
});
