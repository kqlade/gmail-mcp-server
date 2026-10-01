/**
 * Gmail credentials stored for each MCP user.
 * Supports multiple Gmail accounts per user.
 */
export interface GmailCredentials {
  /** MCP user ID selected by the authenticated bearer credential */
  mcpUserId: string;
  /** Google user ID */
  googleUserId: string;
  /** Gmail email address */
  email: string;
  /** Current access token */
  accessToken: string;
  /** Encrypted refresh token */
  refreshToken: string;
  /** Access token expiry timestamp (ms since epoch) */
  expiryDate: number;
  /** Granted OAuth scopes (space-separated) */
  scope: string;
  /** Whether this is the default account for the user */
  isDefault: boolean;
  /** When credentials were first created */
  createdAt: Date;
  /** When credentials were last updated */
  updatedAt: Date;
}

/**
 * Summary info for a connected Gmail account.
 */
export interface AccountInfo {
  email: string;
  isDefault: boolean;
  scopes: string[];
  connectedAt: Date;
}

/**
 * OAuth state for CSRF protection during Google OAuth flow.
 */
export interface OAuthState {
  /** Random state token */
  state: string;
  /** MCP user ID this state is bound to */
  mcpUserId: string;
  /** Hash identifying the exact write credential that created the flow */
  credentialId: string;
  /** When the state expires */
  expiresAt: Date;
  /** Requested scopes for this authorization */
  scopes: string[];
  /** PKCE code verifier */
  codeVerifier: string;
}

/**
 * Interface for token storage implementations.
 * Supports SQLite (default) and can be extended to Postgres, etc.
 */
export interface TokenStore {
  /**
   * Initialize the store (create tables, etc.)
   */
  initialize(): Promise<void>;

  /**
   * Get Gmail credentials for an MCP user.
   * If email is specified, returns that specific account.
   * If email is omitted, returns the default account (or most recently updated as fallback).
   */
  getCredentials(mcpUserId: string, email?: string): Promise<GmailCredentials | null>;

  /**
   * Save or update Gmail credentials for an MCP user.
   * If this is the first account for the user, it becomes the default.
   */
  saveCredentials(credentials: Omit<GmailCredentials, 'createdAt' | 'updatedAt'>): Promise<void>;

  /**
   * Delete Gmail credentials.
   * If email is specified, deletes only that account.
   * If email is omitted, deletes all accounts for the user.
   * If deleting the default account, promotes the next account to default.
   */
  deleteCredentials(mcpUserId: string, email?: string): Promise<void>;

  /**
   * Update only the access token and expiry date for a specific account.
   * Used after token refresh.
   */
  updateAccessToken(mcpUserId: string, email: string, accessToken: string, expiryDate: number, refreshToken?: string): Promise<void>;

  /**
   * List all connected Gmail accounts for an MCP user.
   */
  listAccounts(mcpUserId: string): Promise<AccountInfo[]>;

  /**
   * Set which account is the default for an MCP user.
   */
  setDefaultAccount(mcpUserId: string, email: string): Promise<void>;

  /**
   * Save OAuth state for CSRF protection.
   */
  saveOAuthState(state: OAuthState): Promise<void>;

  /**
   * Get and delete OAuth state (one-time use).
   * Returns null if state doesn't exist or is expired.
   */
  consumeOAuthState(state: string): Promise<OAuthState | null>;

  /**
   * Clean up expired OAuth states.
   */
  cleanupExpiredStates(): Promise<void>;

  /**
   * Close the store connection.
   */
  close(): Promise<void>;
}
