import Database from 'better-sqlite3';
import type { TokenStore, GmailCredentials, OAuthState, AccountInfo } from './interface.js';

export class SqliteTokenStore implements TokenStore {
  private db: Database.Database;

  constructor(dbPath: string) {
    this.db = new Database(dbPath);
    this.db.pragma('journal_mode = WAL');
  }

  async initialize(): Promise<void> {
    // Check if migration is needed (old schema has mcp_user_id as sole primary key)
    const tableExists = this.db.prepare(`
      SELECT name FROM sqlite_master WHERE type='table' AND name='gmail_credentials'
    `).get();

    if (tableExists) {
      // Check if is_default column exists
      const tableInfo = this.db.prepare('PRAGMA table_info(gmail_credentials)').all() as Array<{ name: string }>;
      const hasIsDefault = tableInfo.some(col => col.name === 'is_default');

      if (!hasIsDefault) {
        // Migration needed: old schema -> new schema with composite primary key
        this.db.exec(`
          -- Create new table with composite primary key
          CREATE TABLE gmail_credentials_new (
            mcp_user_id TEXT NOT NULL,
            email TEXT NOT NULL,
            google_user_id TEXT NOT NULL,
            access_token TEXT NOT NULL,
            refresh_token TEXT NOT NULL,
            expiry_date INTEGER NOT NULL,
            scope TEXT NOT NULL,
            is_default INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (mcp_user_id, email)
          );

          -- Migrate existing data (existing accounts become default)
          INSERT INTO gmail_credentials_new
            (mcp_user_id, email, google_user_id, access_token, refresh_token,
             expiry_date, scope, is_default, created_at, updated_at)
          SELECT mcp_user_id, email, google_user_id, access_token, refresh_token,
                 expiry_date, scope, 1, created_at, updated_at
          FROM gmail_credentials;

          -- Drop old table and rename
          DROP TABLE gmail_credentials;
          ALTER TABLE gmail_credentials_new RENAME TO gmail_credentials;

          -- Create index for faster default account lookup
          CREATE INDEX idx_gmail_credentials_default
          ON gmail_credentials(mcp_user_id, is_default);
        `);
      }
    } else {
      // Fresh install - create with new schema
      this.db.exec(`
        CREATE TABLE IF NOT EXISTS gmail_credentials (
          mcp_user_id TEXT NOT NULL,
          email TEXT NOT NULL,
          google_user_id TEXT NOT NULL,
          access_token TEXT NOT NULL,
          refresh_token TEXT NOT NULL,
          expiry_date INTEGER NOT NULL,
          scope TEXT NOT NULL,
          is_default INTEGER NOT NULL DEFAULT 0,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL,
          PRIMARY KEY (mcp_user_id, email)
        );

        CREATE INDEX IF NOT EXISTS idx_gmail_credentials_default
        ON gmail_credentials(mcp_user_id, is_default);
      `);
    }

    // Create OAuth state table
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS oauth_states (
        state TEXT PRIMARY KEY,
        mcp_user_id TEXT NOT NULL,
        credential_id TEXT NOT NULL,
        expires_at TEXT NOT NULL,
        scopes TEXT NOT NULL,
        code_verifier TEXT NOT NULL
      )
    `);
    const oauthStateColumns = this.db.prepare('PRAGMA table_info(oauth_states)').all() as Array<{ name: string }>;
    if (!oauthStateColumns.some((column) => column.name === 'credential_id')) {
      // Existing in-flight links predate exact-credential binding and must
      // fail closed; the empty value cannot match a credential fingerprint.
      this.db.exec(`ALTER TABLE oauth_states ADD COLUMN credential_id TEXT NOT NULL DEFAULT ''`);
    }

    // Create index for cleanup
    this.db.exec(`
      CREATE INDEX IF NOT EXISTS idx_oauth_states_expires_at
      ON oauth_states(expires_at)
    `);
  }

  async getCredentials(mcpUserId: string, email?: string): Promise<GmailCredentials | null> {
    type CredentialRow = {
      mcp_user_id: string;
      google_user_id: string;
      email: string;
      access_token: string;
      refresh_token: string;
      expiry_date: number;
      scope: string;
      is_default: number;
      created_at: string;
      updated_at: string;
    };

    let row: CredentialRow | undefined;

    if (email) {
      // Get specific account
      const stmt = this.db.prepare(`
        SELECT * FROM gmail_credentials WHERE mcp_user_id = ? AND email = ?
      `);
      row = stmt.get(mcpUserId, email) as CredentialRow | undefined;
    } else {
      // Get default account
      const stmt = this.db.prepare(`
        SELECT * FROM gmail_credentials
        WHERE mcp_user_id = ? AND is_default = 1
      `);
      row = stmt.get(mcpUserId) as CredentialRow | undefined;

      // Fallback: if no explicit default, get the most recently updated
      if (!row) {
        const fallbackStmt = this.db.prepare(`
          SELECT * FROM gmail_credentials
          WHERE mcp_user_id = ?
          ORDER BY updated_at DESC LIMIT 1
        `);
        row = fallbackStmt.get(mcpUserId) as CredentialRow | undefined;
      }
    }

    if (!row) {
      return null;
    }

    return {
      mcpUserId: row.mcp_user_id,
      googleUserId: row.google_user_id,
      email: row.email,
      accessToken: row.access_token,
      refreshToken: row.refresh_token,
      expiryDate: row.expiry_date,
      scope: row.scope,
      isDefault: row.is_default === 1,
      createdAt: new Date(row.created_at),
      updatedAt: new Date(row.updated_at),
    };
  }

  async saveCredentials(credentials: Omit<GmailCredentials, 'createdAt' | 'updatedAt'>): Promise<void> {
    const now = new Date().toISOString();

    this.db.transaction(() => {
      // Check if this specific account already exists
      const existingAccount = this.db.prepare(`
        SELECT is_default FROM gmail_credentials WHERE mcp_user_id = ? AND email = ?
      `).get(credentials.mcpUserId, credentials.email) as { is_default: number } | undefined;

      // Check total account count for this user
      const existingCount = this.db.prepare(`
        SELECT COUNT(*) as count FROM gmail_credentials WHERE mcp_user_id = ?
      `).get(credentials.mcpUserId) as { count: number };

      let isDefault: number;
      if (existingAccount) {
        // Reconnecting: preserve existing default status unless explicitly setting as default
        isDefault = credentials.isDefault ? 1 : existingAccount.is_default;
      } else {
        // New account: first account is always default; otherwise use provided value
        isDefault = existingCount.count === 0 ? 1 : (credentials.isDefault ? 1 : 0);
      }

      // If setting as default, unset other defaults first
      if (isDefault === 1 && existingCount.count > 0) {
        this.db.prepare(`
          UPDATE gmail_credentials SET is_default = 0 WHERE mcp_user_id = ?
        `).run(credentials.mcpUserId);
      }

      const stmt = this.db.prepare(`
        INSERT INTO gmail_credentials
          (mcp_user_id, email, google_user_id, access_token, refresh_token,
           expiry_date, scope, is_default, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(mcp_user_id, email) DO UPDATE SET
          google_user_id = excluded.google_user_id,
          access_token = excluded.access_token,
          refresh_token = excluded.refresh_token,
          expiry_date = excluded.expiry_date,
          scope = excluded.scope,
          is_default = excluded.is_default,
          updated_at = excluded.updated_at
      `);

      stmt.run(
        credentials.mcpUserId,
        credentials.email,
        credentials.googleUserId,
        credentials.accessToken,
        credentials.refreshToken,
        credentials.expiryDate,
        credentials.scope,
        isDefault,
        now,
        now
      );
    })();
  }

  async deleteCredentials(mcpUserId: string, email?: string): Promise<void> {
    if (email) {
      // Delete specific account
      this.db.transaction(() => {
        const wasDefault = this.db.prepare(`
          SELECT is_default FROM gmail_credentials WHERE mcp_user_id = ? AND email = ?
        `).get(mcpUserId, email) as { is_default: number } | undefined;

        this.db.prepare(`
          DELETE FROM gmail_credentials WHERE mcp_user_id = ? AND email = ?
        `).run(mcpUserId, email);

        // If deleted account was default, promote next account (oldest by created_at)
        if (wasDefault?.is_default === 1) {
          this.db.prepare(`
            UPDATE gmail_credentials
            SET is_default = 1, updated_at = ?
            WHERE mcp_user_id = ? AND email = (
              SELECT email FROM gmail_credentials
              WHERE mcp_user_id = ?
              ORDER BY created_at ASC LIMIT 1
            )
          `).run(new Date().toISOString(), mcpUserId, mcpUserId);
        }
      })();
    } else {
      // Delete all accounts for user
      const stmt = this.db.prepare(`
        DELETE FROM gmail_credentials WHERE mcp_user_id = ?
      `);
      stmt.run(mcpUserId);
    }
  }

  async updateAccessToken(
    mcpUserId: string,
    email: string,
    accessToken: string,
    expiryDate: number,
    refreshToken?: string
  ): Promise<void> {
    const now = new Date().toISOString();

    if (refreshToken) {
      const stmt = this.db.prepare(`
        UPDATE gmail_credentials
        SET access_token = ?, expiry_date = ?, refresh_token = ?, updated_at = ?
        WHERE mcp_user_id = ? AND email = ?
      `);
      stmt.run(accessToken, expiryDate, refreshToken, now, mcpUserId, email);
    } else {
      const stmt = this.db.prepare(`
        UPDATE gmail_credentials
        SET access_token = ?, expiry_date = ?, updated_at = ?
        WHERE mcp_user_id = ? AND email = ?
      `);
      stmt.run(accessToken, expiryDate, now, mcpUserId, email);
    }
  }

  async listAccounts(mcpUserId: string): Promise<AccountInfo[]> {
    const stmt = this.db.prepare(`
      SELECT email, is_default, scope, created_at
      FROM gmail_credentials
      WHERE mcp_user_id = ?
      ORDER BY is_default DESC, created_at ASC
    `);

    const rows = stmt.all(mcpUserId) as Array<{
      email: string;
      is_default: number;
      scope: string;
      created_at: string;
    }>;

    return rows.map(row => ({
      email: row.email,
      isDefault: row.is_default === 1,
      scopes: row.scope.split(' '),
      connectedAt: new Date(row.created_at),
    }));
  }

  async setDefaultAccount(mcpUserId: string, email: string): Promise<void> {
    this.db.transaction(() => {
      // Verify the account exists
      const exists = this.db.prepare(`
        SELECT 1 FROM gmail_credentials WHERE mcp_user_id = ? AND email = ?
      `).get(mcpUserId, email);

      if (!exists) {
        throw new Error(`Account ${email} not found for user`);
      }

      // Unset all defaults for this user
      this.db.prepare(`
        UPDATE gmail_credentials SET is_default = 0 WHERE mcp_user_id = ?
      `).run(mcpUserId);

      // Set new default
      this.db.prepare(`
        UPDATE gmail_credentials SET is_default = 1, updated_at = ?
        WHERE mcp_user_id = ? AND email = ?
      `).run(new Date().toISOString(), mcpUserId, email);
    })();
  }

  async saveOAuthState(state: OAuthState): Promise<void> {
    const stmt = this.db.prepare(`
      INSERT INTO oauth_states (state, mcp_user_id, credential_id, expires_at, scopes, code_verifier)
      VALUES (?, ?, ?, ?, ?, ?)
    `);

    stmt.run(
      state.state,
      state.mcpUserId,
      state.credentialId,
      state.expiresAt.toISOString(),
      JSON.stringify(state.scopes),
      state.codeVerifier
    );
  }

  async consumeOAuthState(state: string): Promise<OAuthState | null> {
    const now = new Date().toISOString();

    // Get and delete in a transaction
    const result = this.db.transaction(() => {
      const selectStmt = this.db.prepare(`
        SELECT * FROM oauth_states WHERE state = ? AND expires_at > ?
      `);

      const row = selectStmt.get(state, now) as {
        state: string;
        mcp_user_id: string;
        credential_id: string;
        expires_at: string;
        scopes: string;
        code_verifier: string;
      } | undefined;

      if (!row) {
        return null;
      }

      // Delete the state (one-time use)
      const deleteStmt = this.db.prepare(`
        DELETE FROM oauth_states WHERE state = ?
      `);
      deleteStmt.run(state);

      return {
        state: row.state,
        mcpUserId: row.mcp_user_id,
        credentialId: row.credential_id,
        expiresAt: new Date(row.expires_at),
        scopes: JSON.parse(row.scopes) as string[],
        codeVerifier: row.code_verifier,
      };
    })();

    return result;
  }

  async cleanupExpiredStates(): Promise<void> {
    const now = new Date().toISOString();
    const stmt = this.db.prepare(`
      DELETE FROM oauth_states WHERE expires_at <= ?
    `);
    stmt.run(now);
  }

  async close(): Promise<void> {
    this.db.close();
  }
}
