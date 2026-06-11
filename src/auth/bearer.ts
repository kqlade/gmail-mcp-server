/**
 * Static bearer-token auth for the MCP endpoint.
 *
 * This server is single-operator: one shared secret (MCP_AUTH_TOKEN) is held
 * by the MCP client (e.g. Hermes) and sent as `Authorization: Bearer <token>`.
 *
 * The Google OAuth start URL is opened in a browser, where we can't send the
 * bearer header. Instead, the authenticated gmail.authorize tool embeds a
 * short-lived HMAC "start token" (derived from the secret, never the secret
 * itself) in the URL, which /oauth/start validates.
 */

import { createHmac, timingSafeEqual } from 'node:crypto';

const START_TOKEN_TTL_MS = 10 * 60 * 1000;

function safeEqual(a: string, b: string): boolean {
  const bufA = Buffer.from(a);
  const bufB = Buffer.from(b);
  if (bufA.length !== bufB.length) return false;
  return timingSafeEqual(bufA, bufB);
}

/** Validate an Authorization header against the configured secret. */
export function checkBearerToken(authorizationHeader: string | undefined, secret: string): boolean {
  if (!authorizationHeader) return false;
  const match = /^Bearer\s+(.+)$/i.exec(authorizationHeader.trim());
  if (!match) return false;
  return safeEqual(match[1]!, secret);
}

function signStartToken(expiresAtMs: number, secret: string): string {
  return createHmac('sha256', secret).update(`oauth-start:${expiresAtMs}`).digest('base64url');
}

/** Create `exp` + `sig` query params authorizing one /oauth/start visit window. */
export function createStartToken(secret: string, now = Date.now()): { exp: string; sig: string } {
  const expiresAtMs = now + START_TOKEN_TTL_MS;
  return { exp: String(expiresAtMs), sig: signStartToken(expiresAtMs, secret) };
}

/** Validate `exp` + `sig` query params produced by createStartToken. */
export function checkStartToken(
  exp: string | undefined,
  sig: string | undefined,
  secret: string,
  now = Date.now()
): boolean {
  if (!exp || !sig) return false;
  const expiresAtMs = Number(exp);
  if (!Number.isFinite(expiresAtMs) || expiresAtMs < now) return false;
  return safeEqual(sig, signStartToken(expiresAtMs, secret));
}
