import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { SqliteTokenStore } from '../../src/store/sqlite.js';
import { unlinkSync, existsSync } from 'node:fs';

describe('SqliteTokenStore', () => {
  const testDbPath = './test-tokens.db';
  let store: SqliteTokenStore;

  beforeEach(async () => {
    // Clean up any existing test database
    if (existsSync(testDbPath)) {
      unlinkSync(testDbPath);
    }
    store = new SqliteTokenStore(testDbPath);
    await store.initialize();
  });

  afterEach(async () => {
    await store.close();
    if (existsSync(testDbPath)) {
      unlinkSync(testDbPath);
    }
    // Also clean up WAL files
    if (existsSync(testDbPath + '-wal')) {
      unlinkSync(testDbPath + '-wal');
    }
    if (existsSync(testDbPath + '-shm')) {
      unlinkSync(testDbPath + '-shm');
    }
  });

  describe('credentials', () => {
    const testCredentials = {
      mcpUserId: 'user-123',
      googleUserId: 'google-456',
      email: 'test@example.com',
      accessToken: 'access-token-xyz',
      refreshToken: 'encrypted-refresh-token',
      expiryDate: Date.now() + 3600000,
      scope: 'https://www.googleapis.com/auth/gmail.readonly',
      isDefault: false, // Will be overridden to true for first account
    };

    it('should save and retrieve credentials', async () => {
      await store.saveCredentials(testCredentials);
      const retrieved = await store.getCredentials(testCredentials.mcpUserId);

      expect(retrieved).not.toBeNull();
      expect(retrieved?.mcpUserId).toBe(testCredentials.mcpUserId);
      expect(retrieved?.email).toBe(testCredentials.email);
      expect(retrieved?.accessToken).toBe(testCredentials.accessToken);
      expect(retrieved?.refreshToken).toBe(testCredentials.refreshToken);
    });

    it('should return null for non-existent user', async () => {
      const result = await store.getCredentials('non-existent');
      expect(result).toBeNull();
    });

    it('should update existing credentials for same email', async () => {
      await store.saveCredentials(testCredentials);

      const updatedCredentials = {
        ...testCredentials,
        accessToken: 'new-access-token',
      };
      await store.saveCredentials(updatedCredentials);

      const retrieved = await store.getCredentials(testCredentials.mcpUserId);
      expect(retrieved?.accessToken).toBe('new-access-token');
      expect(retrieved?.email).toBe(testCredentials.email);
    });

    it('should delete credentials', async () => {
      await store.saveCredentials(testCredentials);
      await store.deleteCredentials(testCredentials.mcpUserId);

      const result = await store.getCredentials(testCredentials.mcpUserId);
      expect(result).toBeNull();
    });

    it('should update access token only', async () => {
      await store.saveCredentials(testCredentials);

      const newAccessToken = 'refreshed-token';
      const newExpiry = Date.now() + 7200000;
      await store.updateAccessToken(testCredentials.mcpUserId, testCredentials.email, newAccessToken, newExpiry);

      const retrieved = await store.getCredentials(testCredentials.mcpUserId);
      expect(retrieved?.accessToken).toBe(newAccessToken);
      expect(retrieved?.expiryDate).toBe(newExpiry);
      expect(retrieved?.refreshToken).toBe(testCredentials.refreshToken);
    });

    it('should update access token and refresh token', async () => {
      await store.saveCredentials(testCredentials);

      const newAccessToken = 'new-access';
      const newRefreshToken = 'rotated-refresh';
      const newExpiry = Date.now() + 7200000;
      await store.updateAccessToken(testCredentials.mcpUserId, testCredentials.email, newAccessToken, newExpiry, newRefreshToken);

      const retrieved = await store.getCredentials(testCredentials.mcpUserId);
      expect(retrieved?.accessToken).toBe(newAccessToken);
      expect(retrieved?.refreshToken).toBe(newRefreshToken);
    });
  });

  describe('multi-account support', () => {
    const user1 = 'user-123';
    const account1 = {
      mcpUserId: user1,
      googleUserId: 'google-1',
      email: 'first@example.com',
      accessToken: 'token-1',
      refreshToken: 'refresh-1',
      expiryDate: Date.now() + 3600000,
      scope: 'https://www.googleapis.com/auth/gmail.readonly',
      isDefault: false,
    };
    const account2 = {
      mcpUserId: user1,
      googleUserId: 'google-2',
      email: 'second@example.com',
      accessToken: 'token-2',
      refreshToken: 'refresh-2',
      expiryDate: Date.now() + 3600000,
      scope: 'https://www.googleapis.com/auth/gmail.readonly https://www.googleapis.com/auth/gmail.labels',
      isDefault: false,
    };
    const account3 = {
      mcpUserId: user1,
      googleUserId: 'google-3',
      email: 'third@example.com',
      accessToken: 'token-3',
      refreshToken: 'refresh-3',
      expiryDate: Date.now() + 3600000,
      scope: 'https://www.googleapis.com/auth/gmail.readonly',
      isDefault: false,
    };

    it('should make first account default automatically', async () => {
      await store.saveCredentials(account1);

      const retrieved = await store.getCredentials(user1);
      expect(retrieved?.email).toBe(account1.email);
      expect(retrieved?.isDefault).toBe(true);
    });

    it('should not make subsequent accounts default', async () => {
      await store.saveCredentials(account1);
      await store.saveCredentials(account2);

      const accounts = await store.listAccounts(user1);
      expect(accounts).toHaveLength(2);

      const defaultAccount = accounts.find(a => a.isDefault);
      expect(defaultAccount?.email).toBe(account1.email);

      const secondAccount = accounts.find(a => a.email === account2.email);
      expect(secondAccount?.isDefault).toBe(false);
    });

    it('should retrieve specific account by email', async () => {
      await store.saveCredentials(account1);
      await store.saveCredentials(account2);

      const retrieved = await store.getCredentials(user1, account2.email);
      expect(retrieved?.email).toBe(account2.email);
      expect(retrieved?.accessToken).toBe(account2.accessToken);
    });

    it('should retrieve default account when no email specified', async () => {
      await store.saveCredentials(account1);
      await store.saveCredentials(account2);

      const retrieved = await store.getCredentials(user1);
      expect(retrieved?.email).toBe(account1.email);
      expect(retrieved?.isDefault).toBe(true);
    });

    it('should list all accounts for a user', async () => {
      await store.saveCredentials(account1);
      await store.saveCredentials(account2);
      await store.saveCredentials(account3);

      const accounts = await store.listAccounts(user1);
      expect(accounts).toHaveLength(3);
      expect(accounts.map(a => a.email).sort()).toEqual([
        'first@example.com',
        'second@example.com',
        'third@example.com',
      ]);
    });

    it('should return empty array for user with no accounts', async () => {
      const accounts = await store.listAccounts('no-accounts-user');
      expect(accounts).toEqual([]);
    });

    it('should change default account', async () => {
      await store.saveCredentials(account1);
      await store.saveCredentials(account2);

      // First account is default
      let accounts = await store.listAccounts(user1);
      expect(accounts.find(a => a.isDefault)?.email).toBe(account1.email);

      // Change default to second account
      await store.setDefaultAccount(user1, account2.email);

      accounts = await store.listAccounts(user1);
      const defaultAccount = accounts.find(a => a.isDefault);
      expect(defaultAccount?.email).toBe(account2.email);

      // First account should no longer be default
      const firstAccount = accounts.find(a => a.email === account1.email);
      expect(firstAccount?.isDefault).toBe(false);
    });

    it('should throw error when setting non-existent account as default', async () => {
      await store.saveCredentials(account1);

      await expect(
        store.setDefaultAccount(user1, 'nonexistent@example.com')
      ).rejects.toThrow('Account nonexistent@example.com not found for user');
    });

    it('should delete specific account by email', async () => {
      await store.saveCredentials(account1);
      await store.saveCredentials(account2);

      await store.deleteCredentials(user1, account2.email);

      const accounts = await store.listAccounts(user1);
      expect(accounts).toHaveLength(1);
      expect(accounts[0]!.email).toBe(account1.email);
    });

    it('should promote next account when deleting default', async () => {
      await store.saveCredentials(account1);
      await store.saveCredentials(account2);
      await store.saveCredentials(account3);

      // Delete the default account (first one)
      await store.deleteCredentials(user1, account1.email);

      const accounts = await store.listAccounts(user1);
      expect(accounts).toHaveLength(2);

      // One of the remaining should be promoted to default
      const defaultAccount = accounts.find(a => a.isDefault);
      expect(defaultAccount).toBeDefined();
      expect(defaultAccount?.email).not.toBe(account1.email);
    });

    it('should delete all accounts when email not specified', async () => {
      await store.saveCredentials(account1);
      await store.saveCredentials(account2);

      await store.deleteCredentials(user1);

      const accounts = await store.listAccounts(user1);
      expect(accounts).toHaveLength(0);
    });

    it('should update access token for specific account', async () => {
      await store.saveCredentials(account1);
      await store.saveCredentials(account2);

      const newToken = 'updated-token-2';
      const newExpiry = Date.now() + 7200000;
      await store.updateAccessToken(user1, account2.email, newToken, newExpiry);

      const retrieved = await store.getCredentials(user1, account2.email);
      expect(retrieved?.accessToken).toBe(newToken);

      // First account should be unchanged
      const first = await store.getCredentials(user1, account1.email);
      expect(first?.accessToken).toBe(account1.accessToken);
    });

    it('should handle reconnecting existing account', async () => {
      await store.saveCredentials(account1);
      await store.saveCredentials(account2);

      // Reconnect first account with new tokens
      const reconnected = {
        ...account1,
        accessToken: 'reconnected-token',
        refreshToken: 'reconnected-refresh',
      };
      await store.saveCredentials(reconnected);

      const accounts = await store.listAccounts(user1);
      expect(accounts).toHaveLength(2); // Still 2 accounts

      const first = await store.getCredentials(user1, account1.email);
      expect(first?.accessToken).toBe('reconnected-token');
      expect(first?.isDefault).toBe(true); // Should remain default
    });

    it('should keep accounts isolated between users', async () => {
      const user2 = 'user-456';
      const user2Account = {
        ...account1,
        mcpUserId: user2,
        email: 'user2@example.com',
      };

      await store.saveCredentials(account1);
      await store.saveCredentials(user2Account);

      const user1Accounts = await store.listAccounts(user1);
      const user2Accounts = await store.listAccounts(user2);

      expect(user1Accounts).toHaveLength(1);
      expect(user2Accounts).toHaveLength(1);
      expect(user1Accounts[0]!.email).toBe(account1.email);
      expect(user2Accounts[0]!.email).toBe(user2Account.email);
    });
  });

  describe('OAuth state', () => {
    const testState = {
      state: 'random-state-token-abc',
      mcpUserId: 'user-123',
      credentialId: 'credential-fingerprint-123',
      expiresAt: new Date(Date.now() + 600000), // 10 minutes
      scopes: ['gmail.readonly'],
      codeVerifier: 'pkce-code-verifier-xyz',
    };

    it('should save and consume state (one-time use)', async () => {
      await store.saveOAuthState(testState);

      const consumed = await store.consumeOAuthState(testState.state);
      expect(consumed).not.toBeNull();
      expect(consumed?.mcpUserId).toBe(testState.mcpUserId);
      expect(consumed?.credentialId).toBe(testState.credentialId);
      expect(consumed?.scopes).toEqual(testState.scopes);
      expect(consumed?.codeVerifier).toBe(testState.codeVerifier);

      // Should be deleted after consumption
      const secondAttempt = await store.consumeOAuthState(testState.state);
      expect(secondAttempt).toBeNull();
    });

    it('should return null for expired state', async () => {
      const expiredState = {
        ...testState,
        expiresAt: new Date(Date.now() - 1000), // Already expired
      };
      await store.saveOAuthState(expiredState);

      const result = await store.consumeOAuthState(expiredState.state);
      expect(result).toBeNull();
    });

    it('should return null for non-existent state', async () => {
      const result = await store.consumeOAuthState('non-existent-state');
      expect(result).toBeNull();
    });

    it('should clean up expired states', async () => {
      const expiredState = {
        ...testState,
        state: 'expired-state',
        expiresAt: new Date(Date.now() - 1000),
      };
      await store.saveOAuthState(expiredState);
      await store.cleanupExpiredStates();

      // State should be deleted
      const result = await store.consumeOAuthState(expiredState.state);
      expect(result).toBeNull();
    });
  });
});
