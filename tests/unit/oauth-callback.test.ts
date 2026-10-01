import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { FastifyReply, FastifyRequest } from 'fastify';

const googleMocks = vi.hoisted(() => ({
  getToken: vi.fn(),
  setCredentials: vi.fn(),
  getProfile: vi.fn(),
}));

vi.mock('googleapis', () => ({
  google: {
    auth: {
      OAuth2: class {
        getToken = googleMocks.getToken;
        setCredentials = googleMocks.setCredentials;
      },
    },
    gmail: () => ({
      users: {
        getProfile: googleMocks.getProfile,
      },
    }),
  },
}));

import { createGoogleOAuth } from '../../src/auth/googleOAuth.js';
import {
  callerCredentialId,
  parseCallers,
  type Config,
} from '../../src/config.js';
import type {
  GmailCredentials,
  OAuthState,
  TokenStore,
} from '../../src/store/interface.js';

const WRITE_TOKEN = 'write-token-0123456789abcdefghij';
const READ_TOKEN = 'read-token-0123456789abcdefghij';
const COMPOSE_SCOPE = 'https://www.googleapis.com/auth/gmail.compose';

function configWith(callers: Config['callers']): Config {
  return {
    port: 3000,
    baseUrl: 'https://gmail.example',
    googleClientId: 'client',
    googleClientSecret: 'secret',
    oauthRedirectUri: 'https://gmail.example/oauth/callback',
    tokenEncryptionKey: 'x'.repeat(32),
    callers,
    dbUrl: ':memory:',
    allowedOrigins: [],
  };
}

function replyRecorder() {
  const state: { status: number; type?: string; body?: unknown } = { status: 200 };
  const reply = {
    status(code: number) {
      state.status = code;
      return reply;
    },
    type(value: string) {
      state.type = value;
      return reply;
    },
    send(body: unknown) {
      state.body = body;
      return reply;
    },
  };
  return { state, reply: reply as unknown as FastifyReply };
}

function requestFor(code = 'authorization-code', state = 'one-time-state'): FastifyRequest {
  return { query: { code, state } } as unknown as FastifyRequest;
}

function oauthState(credentialId: string): OAuthState {
  return {
    state: 'one-time-state',
    mcpUserId: 'ola',
    credentialId,
    expiresAt: new Date(Date.now() + 60_000),
    scopes: ['gmail.compose'],
    codeVerifier: 'pkce-verifier',
  };
}

describe('OAuth callback', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    googleMocks.getToken.mockResolvedValue({
      tokens: {
        access_token: 'google-access-token',
        refresh_token: 'google-refresh-token',
        expiry_date: Date.now() + 3_600_000,
        scope: COMPOSE_SCOPE,
      },
    });
    googleMocks.getProfile.mockResolvedValue({
      data: { emailAddress: 'owner@example.com' },
    });
  });

  it('binds the PKCE exchange and saved account to the consumed write credential', async () => {
    const caller = parseCallers(`ola:${WRITE_TOKEN}:owner@example.com:write`)[0]!;
    let saved: Omit<GmailCredentials, 'createdAt' | 'updatedAt'> | undefined;
    const store = {
      consumeOAuthState: async () => oauthState(callerCredentialId(caller)),
      listAccounts: async () => [],
      saveCredentials: async (credentials: Omit<GmailCredentials, 'createdAt' | 'updatedAt'>) => {
        saved = credentials;
      },
    } as unknown as TokenStore;
    const oauth = createGoogleOAuth({ config: configWith([caller]), tokenStore: store });
    const { state, reply } = replyRecorder();

    await oauth.callbackHandler(requestFor(), reply);

    expect(state.status).toBe(200);
    expect(googleMocks.getToken).toHaveBeenCalledWith({
      code: 'authorization-code',
      codeVerifier: 'pkce-verifier',
    });
    expect(saved).toMatchObject({
      mcpUserId: 'ola',
      email: 'owner@example.com',
      accessToken: 'google-access-token',
      scope: COMPOSE_SCOPE,
    });
    expect(saved?.refreshToken).not.toContain('google-refresh-token');
  });

  it('rejects a callback when the exact credential was removed or downgraded', async () => {
    const formerWrite = parseCallers(`ola:${WRITE_TOKEN}:owner@example.com:write`)[0]!;
    const currentRead = parseCallers(`ola:${READ_TOKEN}:owner@example.com:read`)[0]!;
    const store = {
      consumeOAuthState: async () => oauthState(callerCredentialId(formerWrite)),
    } as unknown as TokenStore;
    const oauth = createGoogleOAuth({
      config: configWith([currentRead]),
      tokenStore: store,
    });
    const { state, reply } = replyRecorder();

    await oauth.callbackHandler(requestFor(), reply);

    expect(state.status).toBe(403);
    expect(state.body).toMatchObject({ error: 'unknown_credential' });
    expect(googleMocks.getToken).not.toHaveBeenCalled();
  });

  it('rejects broader Google scopes before profile lookup or token storage', async () => {
    const caller = parseCallers(`ola:${WRITE_TOKEN}:owner@example.com:write`)[0]!;
    googleMocks.getToken.mockResolvedValue({
      tokens: {
        access_token: 'google-access-token',
        refresh_token: 'google-refresh-token',
        scope: `${COMPOSE_SCOPE} https://mail.google.com/`,
      },
    });
    const saveCredentials = vi.fn();
    const store = {
      consumeOAuthState: async () => oauthState(callerCredentialId(caller)),
      saveCredentials,
    } as unknown as TokenStore;
    const oauth = createGoogleOAuth({ config: configWith([caller]), tokenStore: store });
    const { state, reply } = replyRecorder();

    await oauth.callbackHandler(requestFor(), reply);

    expect(state.status).toBe(403);
    expect(state.body).toMatchObject({ error: 'scope_mismatch' });
    expect(googleMocks.getProfile).not.toHaveBeenCalled();
    expect(saveCredentials).not.toHaveBeenCalled();
  });

  it('rejects the wrong Google account without storing credentials', async () => {
    const caller = parseCallers(`ola:${WRITE_TOKEN}:owner@example.com:write`)[0]!;
    googleMocks.getProfile.mockResolvedValue({
      data: { emailAddress: 'other@example.com' },
    });
    const saveCredentials = vi.fn();
    const store = {
      consumeOAuthState: async () => oauthState(callerCredentialId(caller)),
      saveCredentials,
    } as unknown as TokenStore;
    const oauth = createGoogleOAuth({ config: configWith([caller]), tokenStore: store });
    const { state, reply } = replyRecorder();

    await oauth.callbackHandler(requestFor(), reply);

    expect(state.status).toBe(403);
    expect(state.type).toContain('text/html');
    expect(String(state.body)).toContain('Wrong Google Account');
    expect(saveCredentials).not.toHaveBeenCalled();
  });
});
