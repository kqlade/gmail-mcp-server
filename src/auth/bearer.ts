/**
 * Bearer-token auth for the MCP endpoint, per caller.
 *
 * Each caller (one person's Ripple compartment) holds its own token and sends
 * it as `Authorization: Bearer <token>`. The token is the identity: it selects
 * the mcpUserId whose Gmail credentials a request may touch, and nothing in
 * the request body can override it.
 *
 * OAuth URLs are created directly by the authenticated gmail.authorize tool;
 * this module has no browser-link credential format to replay or tamper with.
 */

import { callerForToken, type Caller } from '../config.js';

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
