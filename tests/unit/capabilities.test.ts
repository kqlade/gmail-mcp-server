import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import type { FastifyInstance } from 'fastify';
import { parseCallers, type Config } from '../../src/config.js';
import { createHttpServer } from '../../src/http/server.js';
import { createMcpServer } from '../../src/mcp/server.js';
import type { TokenStore } from '../../src/store/interface.js';

const READ_TOKEN = 'read-token-0123456789abcdefghij';
const WRITE_TOKEN = 'write-token-0123456789abcdefghij';

function inertStore(): TokenStore {
  return {
    initialize: async () => undefined,
    getCredentials: async () => null,
    saveCredentials: async () => undefined,
    deleteCredentials: async () => undefined,
    updateAccessToken: async () => undefined,
    listAccounts: async () => [],
    setDefaultAccount: async () => undefined,
    saveOAuthState: async () => undefined,
    consumeOAuthState: async () => null,
    cleanupExpiredStates: async () => undefined,
    close: async () => undefined,
  };
}

type ToolDescriptor = {
  name: string;
  inputSchema?: {
    properties?: Record<string, { enum?: string[] }>;
  };
};

function tools(body: string): ToolDescriptor[] {
  const dataLine = body.split('\n').find((line) => line.startsWith('data: '));
  const payload = JSON.parse(dataLine ? dataLine.slice('data: '.length) : body) as {
    result: { tools: ToolDescriptor[] };
  };
  return payload.result.tools;
}

function toolNames(body: string): Set<string> {
  return new Set(tools(body).map((tool) => tool.name));
}

describe('server-side capability tokens', () => {
  let server: FastifyInstance;

  beforeEach(async () => {
    const config: Config = {
      port: 3000,
      baseUrl: 'https://gmail.example',
      googleClientId: 'client',
      googleClientSecret: 'secret',
      oauthRedirectUri: 'https://gmail.example/oauth/callback',
      tokenEncryptionKey: 'x'.repeat(32),
      callers: parseCallers(
        `primary:${READ_TOKEN}:owner@example.com:read,` +
        `primary:${WRITE_TOKEN}:owner@example.com:write`
      ),
      dbUrl: ':memory:',
      allowedOrigins: [],
    };
    const tokenStore = inertStore();
    const mcpServer = await createMcpServer({ config, tokenStore });
    server = await createHttpServer({ config, tokenStore, mcpServer });
  });

  afterEach(async () => {
    await server.close();
  });

  async function listTools(token: string) {
    return server.inject({
      method: 'POST',
      url: '/mcp',
      headers: {
        authorization: `Bearer ${token}`,
        accept: 'application/json, text/event-stream',
        'content-type': 'application/json',
      },
      payload: {
        jsonrpc: '2.0',
        id: 1,
        method: 'tools/list',
        params: {},
      },
    });
  }

  it('read token advertises no mutation tools', async () => {
    const response = await listTools(READ_TOKEN);
    expect(response.statusCode).toBe(200);
    const names = toolNames(response.body);
    expect(names).toContain('gmail.triageSnapshot');
    expect(names).toContain('gmail.getMessage');
    expect(names).not.toContain('gmail.sendMessage');
    expect(names).not.toContain('gmail.organizeMessages');
    expect(names).not.toContain('gmail.authorize');
    const manageDraft = tools(response.body).find((tool) => tool.name === 'gmail.manageDraft');
    expect(manageDraft?.inputSchema?.properties?.['action']?.enum).toEqual([
      'get',
      'list',
    ]);
  });

  it('write token advertises no mailbox-read tools', async () => {
    const response = await listTools(WRITE_TOKEN);
    expect(response.statusCode).toBe(200);
    const names = toolNames(response.body);
    expect(names).toContain('gmail.sendMessage');
    expect(names).toContain('gmail.organizeMessages');
    expect(names).toContain('gmail.authorize');
    expect(names).not.toContain('gmail.getMessage');
    expect(names).not.toContain('gmail.triageSnapshot');
    const manageDraft = tools(response.body).find((tool) => tool.name === 'gmail.manageDraft');
    expect(manageDraft?.inputSchema?.properties?.['action']?.enum).toEqual([
      'create',
      'update',
      'delete',
      'send',
    ]);
  });

  it('unknown token is rejected before MCP dispatch', async () => {
    const response = await listTools('unknown-token-0123456789abcdef');
    expect(response.statusCode).toBe(401);
  });

  it.each([
    [READ_TOKEN, 'gmail.sendMessage'],
    [WRITE_TOKEN, 'gmail.getMessage'],
  ])('rejects calls to tools hidden from that capability', async (token, name) => {
    const response = await server.inject({
      method: 'POST',
      url: '/mcp',
      headers: {
        authorization: `Bearer ${token}`,
        accept: 'application/json, text/event-stream',
        'content-type': 'application/json',
      },
      payload: {
        jsonrpc: '2.0',
        id: 6,
        method: 'tools/call',
        params: { name, arguments: {} },
      },
    });
    expect(response.statusCode).toBe(200);
    const dataLine = response.body.split('\n').find((line) => line.startsWith('data: '));
    const payload = JSON.parse(dataLine ? dataLine.slice('data: '.length) : response.body) as {
      result?: { isError?: boolean; content?: Array<{ text?: string }> };
    };
    expect(payload.result?.isError).toBe(true);
    expect(payload.result?.content?.[0]?.text).toMatch(/not found/i);
  });

  it.each([
    ['email', { email: 'other@example.com', to: 'x@example.com', subject: 'x', body: 'x' }],
    ['replyToMessageId', { to: 'x@example.com', subject: 'x', body: 'x', replyToMessageId: 'message-1' }],
    ['archiveEntireThread', {
      actions: [{ action: 'archive', messageIds: ['message-1'], archiveEntireThread: true }],
    }],
  ])('rejects retired %s arguments instead of silently retargeting them', async (_name, args) => {
    const response = await server.inject({
      method: 'POST',
      url: '/mcp',
      headers: {
        authorization: `Bearer ${WRITE_TOKEN}`,
        accept: 'application/json, text/event-stream',
        'content-type': 'application/json',
      },
      payload: {
        jsonrpc: '2.0',
        id: 7,
        method: 'tools/call',
        params: { name: 'gmail.sendMessage', arguments: args },
      },
    });
    expect(response.statusCode).toBe(200);
    const payload = JSON.parse(response.body) as { error?: { code?: number; message?: string } };
    expect(payload.error?.code).toBe(-32602);
    expect(payload.error?.message).toContain(String(_name));
  });

  it('rejects retired routing arguments inside JSON-RPC batches', async () => {
    const response = await server.inject({
      method: 'POST',
      url: '/mcp',
      headers: {
        authorization: `Bearer ${WRITE_TOKEN}`,
        accept: 'application/json, text/event-stream',
        'content-type': 'application/json',
      },
      payload: [
        {
          jsonrpc: '2.0',
          id: 9,
          method: 'tools/call',
          params: {
            name: 'gmail.sendMessage',
            arguments: {
              email: 'other@example.com',
              to: 'x@example.com',
              subject: 'x',
              body: 'x',
            },
          },
        },
        {
          jsonrpc: '2.0',
          id: 10,
          method: 'tools/list',
          params: {},
        },
      ],
    });
    expect(response.statusCode).toBe(200);
    const dataLine = response.body.split('\n').find((line) => line.startsWith('data: '));
    const payload = JSON.parse(dataLine ? dataLine.slice('data: '.length) : response.body) as Array<{
      id?: number;
      error?: { code?: number; message?: string };
    }>;
    expect(Array.isArray(payload)).toBe(true);
    const retired = payload.find((item) => item.id === 9);
    const unaffected = payload.find((item) => item.id === 10);
    expect(retired?.error?.code).toBe(-32602);
    expect(retired?.error?.message).toContain('email');
    expect(unaffected?.error?.code).toBe(-32600);
    expect(unaffected?.error?.message).toContain('send each request separately');
  });

  it('returns a per-item Invalid Request for malformed batch members', async () => {
    const response = await server.inject({
      method: 'POST',
      url: '/mcp',
      headers: {
        authorization: `Bearer ${WRITE_TOKEN}`,
        accept: 'application/json, text/event-stream',
        'content-type': 'application/json',
      },
      payload: [{}],
    });
    expect(response.statusCode).toBe(200);
    const payload = JSON.parse(response.body) as Array<{
      id?: unknown;
      error?: { code?: number; message?: string };
    }>;
    expect(payload).toHaveLength(1);
    expect(payload[0]?.id).toBeNull();
    expect(payload[0]?.error?.code).toBe(-32600);
    expect(payload[0]?.error?.message).toContain('Invalid JSON-RPC request');
  });

  it('returns one Invalid Request object for an empty batch', async () => {
    const response = await server.inject({
      method: 'POST',
      url: '/mcp',
      headers: {
        authorization: `Bearer ${WRITE_TOKEN}`,
        accept: 'application/json, text/event-stream',
        'content-type': 'application/json',
      },
      payload: [],
    });
    expect(response.statusCode).toBe(200);
    const payload = JSON.parse(response.body) as {
      id?: unknown;
      error?: { code?: number; message?: string };
    };
    expect(Array.isArray(payload)).toBe(false);
    expect(payload.id).toBeNull();
    expect(payload.error?.code).toBe(-32600);
  });

  it('suppresses responses only for valid batch notifications', async () => {
    const response = await server.inject({
      method: 'POST',
      url: '/mcp',
      headers: {
        authorization: `Bearer ${WRITE_TOKEN}`,
        accept: 'application/json, text/event-stream',
        'content-type': 'application/json',
      },
      payload: [{ jsonrpc: '2.0', method: 'notifications/initialized' }],
    });
    expect(response.statusCode).toBe(202);
    expect(response.body).toBe('');
  });

  it('does not respond to retired-argument notifications', async () => {
    const response = await server.inject({
      method: 'POST',
      url: '/mcp',
      headers: {
        authorization: `Bearer ${WRITE_TOKEN}`,
        accept: 'application/json, text/event-stream',
        'content-type': 'application/json',
      },
      payload: {
        jsonrpc: '2.0',
        method: 'tools/call',
        params: {
          name: 'gmail.sendMessage',
          arguments: {
            email: 'other@example.com',
            to: 'x@example.com',
            subject: 'x',
            body: 'x',
          },
        },
      },
    });
    expect(response.statusCode).toBe(202);
    expect(response.body).toBe('');
  });

  it('does not suppress malformed notifications with scalar params', async () => {
    const response = await server.inject({
      method: 'POST',
      url: '/mcp',
      headers: {
        authorization: `Bearer ${WRITE_TOKEN}`,
        accept: 'application/json, text/event-stream',
        'content-type': 'application/json',
      },
      payload: [{ jsonrpc: '2.0', method: 'notifications/initialized', params: 7 }],
    });
    expect(response.statusCode).toBe(200);
    const payload = JSON.parse(response.body) as Array<{
      id?: unknown;
      error?: { code?: number };
    }>;
    expect(payload[0]?.id).toBeNull();
    expect(payload[0]?.error?.code).toBe(-32600);
  });
});
