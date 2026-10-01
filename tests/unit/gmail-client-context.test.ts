import { beforeEach, describe, expect, it, vi } from 'vitest';

const googleMocks = vi.hoisted(() => ({
  draftsGet: vi.fn(),
  draftsUpdate: vi.fn(),
  messagesSend: vi.fn(),
  setCredentials: vi.fn(),
}));

vi.mock('googleapis', () => ({
  google: {
    auth: {
      OAuth2: class {
        setCredentials = googleMocks.setCredentials;
      },
    },
    gmail: () => ({
      users: {
        drafts: {
          get: googleMocks.draftsGet,
          update: googleMocks.draftsUpdate,
        },
        messages: {
          send: googleMocks.messagesSend,
        },
      },
    }),
  },
}));

import { createReplyContext, deriveReplyContextKey } from '../../src/auth/replyContext.js';
import { createGmailClientFactory } from '../../src/gmail/client.js';
import type { GmailCredentials, TokenStore } from '../../src/store/interface.js';

const ENCRYPTION_KEY = 'test-encryption-key-that-is-at-least-32-characters';
const COMPOSE_SCOPE = 'https://www.googleapis.com/auth/gmail.compose';

function credentials(mcpUserId: string): GmailCredentials {
  return {
    mcpUserId,
    googleUserId: 'owner@example.com',
    email: 'owner@example.com',
    accessToken: 'access-token',
    refreshToken: 'unused-encrypted-refresh-token',
    expiryDate: Date.now() + 3_600_000,
    scope: COMPOSE_SCOPE,
    isDefault: true,
    createdAt: new Date(),
    updatedAt: new Date(),
  };
}

function clientFor(mcpUserId: string) {
  const store = {
    getCredentials: async () => credentials(mcpUserId),
  } as unknown as TokenStore;
  return createGmailClientFactory({
    tokenStore: store,
    encryptionKey: ENCRYPTION_KEY,
    googleClientId: 'client',
    googleClientSecret: 'secret',
    pinnedAccountFor: () => 'owner@example.com',
  });
}

describe('Gmail opaque contexts', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    googleMocks.draftsUpdate.mockResolvedValue({
      data: { id: 'draft-1', message: { id: 'updated-message-1' } },
    });
    googleMocks.messagesSend.mockResolvedValue({
      data: { id: 'sent-message-1', threadId: 'thread-1' },
    });
  });

  it('round-trips a read draft into an update without dropping metadata', async () => {
    googleMocks.draftsGet.mockResolvedValue({
      data: {
        id: 'draft-1',
        message: {
          id: 'message-1',
          threadId: 'thread-1',
          snippet: 'old draft',
          payload: {
            mimeType: 'multipart/alternative',
            headers: [
              { name: 'To', value: 'to@example.com' },
              { name: 'Cc', value: 'cc@example.com' },
              { name: 'Bcc', value: 'bcc@example.com' },
              { name: 'Subject', value: 'Original subject' },
              { name: 'In-Reply-To', value: '<parent@example.com>' },
              { name: 'References', value: '<root@example.com> <parent@example.com>' },
            ],
            parts: [
              {
                mimeType: 'text/plain',
                body: { data: Buffer.from('old draft').toString('base64url') },
              },
              {
                mimeType: 'text/html',
                body: { data: Buffer.from('<p>old draft</p>').toString('base64url') },
              },
            ],
          },
        },
      },
    });
    const client = clientFor('draft-user');

    const draft = await client.getDraft('draft-user', 'draft-1');
    expect(draft).toMatchObject({
      to: 'to@example.com',
      cc: 'cc@example.com',
      bcc: 'bcc@example.com',
      subject: 'Original subject',
      body: { text: 'old draft', html: '<p>old draft</p>' },
    });
    expect(draft.draftContext).toMatch(/^dc1\./);
    expect(draft.draftContext).not.toContain('bcc@example.com');

    await client.updateDraft(
      'draft-user',
      'draft-1',
      undefined,
      undefined,
      '<p>new draft</p>',
      { draftContext: draft.draftContext! }
    );

    const request = googleMocks.draftsUpdate.mock.calls[0]?.[0] as {
      requestBody: { message: { raw: string; threadId?: string } };
    };
    const raw = Buffer.from(request.requestBody.message.raw, 'base64url').toString('utf8');
    expect(request.requestBody.message.threadId).toBe('thread-1');
    expect(raw).toContain('To: to@example.com\r\n');
    expect(raw).toContain('Cc: cc@example.com\r\n');
    expect(raw).toContain('Bcc: bcc@example.com\r\n');
    expect(raw).toContain('In-Reply-To: <parent@example.com>\r\n');
    expect(raw).toContain('References: <root@example.com> <parent@example.com>\r\n');
    expect(raw).toContain('Subject: Original subject\r\n');
    expect(raw).toContain('Content-Type: text/html; charset=utf-8\r\n');
  });

  it('binds draft context to both the compartment and draft ID', async () => {
    googleMocks.draftsGet.mockResolvedValue({
      data: {
        id: 'draft-1',
        message: {
          id: 'message-1',
          payload: {
            mimeType: 'text/plain',
            headers: [{ name: 'To', value: 'to@example.com' }],
            body: { data: Buffer.from('old').toString('base64url') },
          },
        },
      },
    });
    const client = clientFor('draft-owner');
    const draft = await client.getDraft('draft-owner', 'draft-1');

    await expect(client.updateDraft(
      'draft-owner',
      'draft-2',
      undefined,
      undefined,
      'new',
      { draftContext: draft.draftContext! }
    )).rejects.toThrow(/invalid|stale/);
    await expect(client.updateDraft(
      'other-user',
      'draft-1',
      undefined,
      undefined,
      'new',
      { draftContext: draft.draftContext! }
    )).rejects.toThrow(/another compartment|invalid|stale/);
    expect(googleMocks.draftsUpdate).not.toHaveBeenCalled();
  });

  it('uses reply context to preserve thread headers without a mailbox read', async () => {
    const replyContext = createReplyContext(
      {
        mcpUserId: 'reply-user',
        messageId: 'message-1',
        threadId: 'thread-1',
        inReplyTo: '<message-1@example.com>',
        references: '<root@example.com>',
      },
      deriveReplyContextKey(ENCRYPTION_KEY)
    );
    const client = clientFor('reply-user');

    await client.sendMessage(
      'reply-user',
      'to@example.com',
      'Re: hello',
      'Reply body',
      { replyContext }
    );

    const request = googleMocks.messagesSend.mock.calls[0]?.[0] as {
      requestBody: { raw: string; threadId?: string };
    };
    const raw = Buffer.from(request.requestBody.raw, 'base64url').toString('utf8');
    expect(request.requestBody.threadId).toBe('thread-1');
    expect(raw).toContain('In-Reply-To: <message-1@example.com>\r\n');
    expect(raw).toContain('References: <root@example.com> <message-1@example.com>\r\n');
  });
});
