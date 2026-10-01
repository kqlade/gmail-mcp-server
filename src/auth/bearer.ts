/**
 * Bearer-token auth for the MCP endpoint, per caller.
 *
 * Each caller (one person's Ripple compartment) holds its own token and sends
 * it as `Authorization: Bearer <token>`. The token is the identity: it selects
 * the mcpUserId whose Gmail credentials a request may touch, and nothing in
 * the request body can override it.
 *
 * The Google OAuth start URL is opened in a browser, where we can't send the
 * bearer header. Instead, the authenticated gmail.authorize tool embeds a
 * short-lived HMAC "start token" in the URL: the caller id in clear plus a
 * signature derived from THAT caller's secret, so a link minted for one
 * person cannot start an OAuth flow that lands in another person's row.
 */

import { createHmac, timingSafeEqual } from 'node:crypto';
import { callerCan, callerForToken, type Caller } from '../config.js';

const START_TOKEN_TTL_MS = 10 * 60 * 1000;

function safeEqual(a: string, b: string): boolean {
  const bufA = Buffer.from(a);
  const bufB = Buffer.from(b);
  if (bufA.length !== bufB.length) return false;
  return timingSafeEqual(bufA, bufB);
}

/** Extract the bearer token from an Authorization header. */
export function bearerToken(authorizationHeader: string | undefined): string {
  if (!authorizationHeader) return '';
  const match = /^Bearer\s+(.+)$/i.exec(authorizationHeader.trim());
  return match ? match[1]!.trim() : '';
}

/** The caller an Authorization header authenticates, or null. */
export function resolveCaller(authorizationHeader: string | undefined, callers: Caller[]): Caller | null {
  return callerForToken(callers, bearerToken(authorizationHeader));
}

function signStartToken(callerId: string, expiresAtMs: number, secret: string): string {
  return createHmac('sha256', secret).update(`oauth-start:${callerId}:${expiresAtMs}`).digest('base64url');
}

/** Create `caller` + `exp` + `sig` query params authorizing one /oauth/start visit window for *caller*. */
export function createStartToken(caller: Caller, now = Date.now()): { caller: string; exp: string; sig: string } {
  const expiresAtMs = now + START_TOKEN_TTL_MS;
  return { caller: caller.id, exp: String(expiresAtMs), sig: signStartToken(caller.id, expiresAtMs, caller.token) };
}

/** Validate start-token query params; returns the caller the link was minted for, or null. */
export function checkStartToken(
  query: { caller?: string; exp?: string; sig?: string },
  callers: Caller[],
  now = Date.now()
): Caller | null {
  const { caller: callerId, exp, sig } = query;
  if (!callerId || !exp || !sig) return null;
  const expiresAtMs = Number(exp);
  if (!Number.isFinite(expiresAtMs) || expiresAtMs < now) return null;
  // One mailbox identity may have separate read and write credentials. Test
  // every write credential for the id: a read token must not turn its bearer
  // secret into a direct /oauth/start mutation path.
  let matched: Caller | null = null;
  for (const caller of callers) {
    if (caller.id !== callerId) continue;
    if (
      callerCan(caller, 'write') &&
      safeEqual(sig, signStartToken(caller.id, expiresAtMs, caller.token))
    ) {
      matched = caller;
    }
  }
  return matched;
}
