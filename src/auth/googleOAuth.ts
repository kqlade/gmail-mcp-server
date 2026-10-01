/**
 * Google OAuth implementation for Gmail authorization.
 *
 * This module handles:
 * - Google OAuth2 client setup
 * - Authenticated authorization-URL creation
 * - /oauth/callback endpoint (exchange code, store credentials)
 * - Scope validation and mapping
 */

import { google } from 'googleapis';
import { CodeChallengeMethod } from 'google-auth-library';
import { randomBytes, createHash } from 'node:crypto';
import type { FastifyRequest, FastifyReply } from 'fastify';
import {
  callerByCredentialId,
  callerCan,
  callerCredentialId,
  type Caller,
  type Config,
} from '../config.js';
import type { TokenStore } from '../store/interface.js';
import { encrypt } from '../utils/crypto.js';

// Scope mapping from friendly names to Google OAuth scopes
export const GMAIL_SCOPES = {
  'gmail.readonly': 'https://www.googleapis.com/auth/gmail.readonly',
  'gmail.labels': 'https://www.googleapis.com/auth/gmail.labels',
  'gmail.modify': 'https://www.googleapis.com/auth/gmail.modify',
  'gmail.compose': 'https://www.googleapis.com/auth/gmail.compose',
} as const;

export type GmailScope = keyof typeof GMAIL_SCOPES;

const VALID_SCOPES = new Set<string>(Object.keys(GMAIL_SCOPES));

export function grantedScopesExactlyMatch(
  requestedScopes: string[],
  grantedScopeText: string | undefined
): boolean {
  const expected = new Set(
    requestedScopes.map((scope) => GMAIL_SCOPES[scope as GmailScope])
  );
  const granted = new Set((grantedScopeText ?? '').split(/\s+/).filter(Boolean));
  return (
    granted.size === expected.size &&
    [...expected].every((scope) => granted.has(scope))
  );
}

export interface GoogleOAuthDependencies {
  config: Config;
  tokenStore: TokenStore;
}

function escapeHtml(value: string): string {
  return value.replace(/[&<>"']/g, (ch) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' } as Record<string, string>
  )[ch] ?? ch);
}

/**
 * Generate PKCE code verifier and challenge.
 */
function generatePKCE(): { codeVerifier: string; codeChallenge: string } {
  // Generate 32 bytes of random data for code verifier
  const codeVerifier = randomBytes(32).toString('base64url');

  // Generate code challenge using SHA256
  const codeChallenge = createHash('sha256')
    .update(codeVerifier)
    .digest('base64url');

  return { codeVerifier, codeChallenge };
}

/**
 * Create Google OAuth handlers.
 */
export function createGoogleOAuth(deps: GoogleOAuthDependencies) {
  const { config, tokenStore } = deps;

  // OAuth2Client carries mutable credentials. A fresh instance per request
  // prevents simultaneous callbacks for different callers from overwriting
  // one another between setCredentials() and users.getProfile().
  const newOAuth2Client = () => new google.auth.OAuth2(
    config.googleClientId,
    config.googleClientSecret,
    config.oauthRedirectUri
  );

  /**
   * Validate and map scopes from friendly names to Google OAuth scopes.
   */
  function mapScopes(scopes: string[]): string[] {
    const googleScopes: string[] = [];

    for (const scope of scopes) {
      if (!VALID_SCOPES.has(scope)) {
        throw new Error(`Invalid scope: ${scope}. Valid scopes are: ${Array.from(VALID_SCOPES).join(', ')}`);
      }
      googleScopes.push(GMAIL_SCOPES[scope as GmailScope]);
    }

    return googleScopes;
  }

  /**
   * Create a one-time Google authorization URL from an already-authenticated
   * MCP request. There is intentionally no unauthenticated /oauth/start hop:
   * the scopes are fixed in Google's URL and repeated in the one-time state.
   */
  async function createAuthorizationUrl(caller: Caller, scopes: string[]): Promise<string> {
    if (!callerCan(caller, 'write')) {
      throw new Error('A write-capability credential is required to authorize Gmail');
    }
    const requestedScopes = [...new Set(scopes.map((scope) => scope.trim()).filter(Boolean))].sort();
    if (
      requestedScopes.length === 0 ||
      requestedScopes.every((scope) => scope === 'gmail.labels')
    ) {
      throw new Error(
        'At least one functional Gmail scope is required: gmail.readonly, gmail.modify, or gmail.compose'
      );
    }
    const googleScopes = mapScopes(requestedScopes);

    // The grant will be stored under the caller the link was minted for.
    const mcpUserId = caller.id;

    // Generate state token (256-bit random)
    const state = randomBytes(32).toString('base64url');

    // Generate PKCE
    const { codeVerifier, codeChallenge } = generatePKCE();

    // Store state for validation in callback
    await tokenStore.saveOAuthState({
      state,
      mcpUserId,
      credentialId: callerCredentialId(caller),
      expiresAt: new Date(Date.now() + 10 * 60 * 1000), // 10 minutes
      scopes: requestedScopes,
      codeVerifier,
    });

    // Generate Google authorization URL
    const authUrl = newOAuth2Client().generateAuthUrl({
      access_type: 'offline',
      scope: googleScopes,
      state,
      prompt: 'consent', // Force consent to ensure refresh token
      code_challenge: codeChallenge,
      code_challenge_method: CodeChallengeMethod.S256,
    });
    return authUrl;
  }

  /**
   * Handler for GET /oauth/callback - handles Google OAuth callback.
   */
  async function callbackHandler(
    request: FastifyRequest,
    reply: FastifyReply
  ): Promise<void> {
    const query = request.query as {
      code?: string;
      state?: string;
      error?: string;
      error_description?: string;
    };

    // Check for OAuth error
    if (query.error) {
      reply.status(400).send({
        error: query.error,
        error_description: query.error_description ?? 'OAuth authorization failed',
      });
      return;
    }

    // Validate required parameters
    if (!query.code || !query.state) {
      reply.status(400).send({
        error: 'invalid_request',
        error_description: 'Missing code or state parameter',
      });
      return;
    }

    // Validate and consume state (one-time use)
    const storedState = await tokenStore.consumeOAuthState(query.state);
    if (!storedState) {
      reply.status(400).send({
        error: 'invalid_state',
        error_description: 'Invalid or expired state parameter',
      });
      return;
    }

    // Removing or downgrading the exact write credential that minted the
    // link invalidates the in-flight flow, even if a read sibling with the
    // same caller id still exists.
    const caller = callerByCredentialId(
      config.callers,
      storedState.mcpUserId,
      storedState.credentialId
    );
    if (!caller || !callerCan(caller, 'write')) {
      reply.status(403).send({
        error: 'unknown_credential',
        error_description: 'The credential that created this authorization link no longer exists or cannot write. Nothing was connected.',
      });
      return;
    }

    try {
      const oauth2Client = newOAuth2Client();
      // Exchange code for tokens
      const { tokens } = await oauth2Client.getToken({
        code: query.code,
        codeVerifier: storedState.codeVerifier,
      });

      if (!tokens.access_token || !tokens.refresh_token) {
        reply.status(500).send({
          error: 'token_error',
          error_description: 'Failed to obtain tokens from Google',
        });
        return;
      }

      // A captured Google URL must not be broadened by editing its `scope`
      // query parameter. Require every requested scope and reject any extra
      // Gmail scope before storing the grant.
      if (!grantedScopesExactlyMatch(storedState.scopes, tokens.scope)) {
        reply.status(403).send({
          error: 'scope_mismatch',
          error_description: 'Google returned scopes different from the signed authorization request. Nothing was connected.',
        });
        return;
      }
      const expectedScopes = new Set(mapScopes(storedState.scopes));

      // Set credentials to fetch user profile
      oauth2Client.setCredentials(tokens);

      // Get user profile from Gmail API
      const gmail = google.gmail({ version: 'v1', auth: oauth2Client });
      const profile = await gmail.users.getProfile({ userId: 'me' });

      if (!profile.data.emailAddress) {
        reply.status(500).send({
          error: 'profile_error',
          error_description: 'Failed to get email address from Gmail',
        });
        return;
      }

      // The state was bound to a caller when the link was minted. A caller
      // with a pinned account may only connect that account: storing any
      // other would succeed here but every tool call would still resolve the
      // pin and fail — a confusing half-connected state. Reject at the door.
      const pinned = caller.account;
      if (pinned && profile.data.emailAddress.toLowerCase() !== pinned) {
        reply.status(403).type('text/html; charset=utf-8').send(`
        <!DOCTYPE html>
        <html>
        <head>
          <meta charset="utf-8">
          <title>Wrong Google Account</title>
        </head>
        <body>
          <h1>Wrong Google Account</h1>
          <p>You authorized <strong>${escapeHtml(profile.data.emailAddress)}</strong>, but this connection is for <strong>${escapeHtml(pinned)}</strong>.</p>
          <p>Nothing was connected. Restart the flow and pick ${escapeHtml(pinned)} in the Google account chooser.</p>
        </body>
        </html>
      `);
        return;
      }

      // Encrypt refresh token before storage
      const encryptedRefreshToken = encrypt(tokens.refresh_token, config.tokenEncryptionKey);

      // Check existing accounts to determine if this is first/reconnect
      const existingAccounts = await tokenStore.listAccounts(storedState.mcpUserId);
      const isFirstAccount = existingAccounts.length === 0;
      const isReconnect = existingAccounts.some(a => a.email === profile.data.emailAddress);

      // Store credentials (first account becomes default; reconnects keep current default status)
      await tokenStore.saveCredentials({
        mcpUserId: storedState.mcpUserId,
        googleUserId: profile.data.emailAddress, // Using email as ID since historyId is not unique
        email: profile.data.emailAddress,
        accessToken: tokens.access_token,
        refreshToken: encryptedRefreshToken,
        expiryDate: tokens.expiry_date ?? Date.now() + 3600000,
        scope: [...expectedScopes].join(' '),
        isDefault: isFirstAccount, // First account is default; additional accounts are not
      });

      // Determine action text for user feedback
      const action = isReconnect ? 'reconnected' : 'connected';
      const accountCount = isFirstAccount ? 1 : existingAccounts.length + (isReconnect ? 0 : 1);

      // Send success page
      reply.type('text/html; charset=utf-8').send(`
        <!DOCTYPE html>
        <html>
        <head>
          <meta charset="utf-8">
          <title>Gmail Connected</title>
        </head>
        <body>
          <h1>Gmail ${action.charAt(0).toUpperCase() + action.slice(1)}</h1>
          <p>Successfully ${action} as <strong>${escapeHtml(profile.data.emailAddress)}</strong></p>
          <p>You now have ${accountCount} Gmail account${accountCount > 1 ? 's' : ''} connected.</p>
          <p>You can now close this window and return to your application.</p>
        </body>
        </html>
      `);
    } catch (error) {
      console.error('OAuth callback error:', error);
      reply.status(500).send({
        error: 'exchange_error',
        error_description: 'Failed to exchange authorization code for tokens',
      });
    }
  }

  return {
    createAuthorizationUrl,
    callbackHandler,
    mapScopes,
  };
}

export type GoogleOAuth = ReturnType<typeof createGoogleOAuth>;
