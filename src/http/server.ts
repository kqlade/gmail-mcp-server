import Fastify, { FastifyInstance, FastifyRequest, FastifyReply } from 'fastify';
import type { Caller, Config } from '../config.js';
import type { TokenStore } from '../store/interface.js';
import type { McpServerInstance } from '../mcp/server.js';
import { createGoogleOAuth } from '../auth/googleOAuth.js';
import { resolveCaller } from '../auth/bearer.js';

export interface HttpServerDependencies {
  config: Config;
  tokenStore: TokenStore;
  mcpServer: McpServerInstance;
}

export async function createHttpServer(deps: HttpServerDependencies): Promise<FastifyInstance> {
  const { config, tokenStore, mcpServer } = deps;

  const googleOAuth = createGoogleOAuth({ config, tokenStore });

  const server = Fastify({
    logger: {
      level: 'info',
    },
  });

  // Custom JSON body parser
  server.addContentTypeParser('application/json', { parseAs: 'string' }, (_req, body, done) => {
    try {
      const json = JSON.parse(body as string);
      done(null, json);
    } catch (err) {
      done(err as Error, undefined);
    }
  });

  // CORS handling
  if (config.allowedOrigins.length > 0) {
    server.addHook('onRequest', async (request, reply) => {
      const origin = request.headers.origin;
      if (origin && config.allowedOrigins.includes(origin)) {
        reply.header('Access-Control-Allow-Origin', origin);
        reply.header('Access-Control-Allow-Credentials', 'true');
        reply.header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
        reply.header('Access-Control-Allow-Headers', 'Content-Type, Authorization, Mcp-Session-Id');
      }

      if (request.method === 'OPTIONS') {
        reply.status(204).send();
      }
    });
  }

  // Health check endpoint
  server.get('/healthz', async (_request: FastifyRequest, reply: FastifyReply) => {
    try {
      await tokenStore.cleanupExpiredStates();
      return { status: 'ok', timestamp: new Date().toISOString() };
    } catch {
      reply.status(503);
      return {
        status: 'degraded',
        timestamp: new Date().toISOString(),
        issues: ['Token store unavailable'],
      };
    }
  });

  // MCP Streamable HTTP endpoint (stateless; bearer token names the caller)
  server.post(
    '/mcp',
    {
      preHandler: async (request, reply) => {
        const caller = resolveCaller(request.headers.authorization, config.callers);
        if (!caller) {
          reply.status(401).header('WWW-Authenticate', 'Bearer').send({
            jsonrpc: '2.0',
            error: { code: -32001, message: 'Unauthorized' },
            id: null,
          });
          return;
        }
        (request as FastifyRequest & { caller?: Caller }).caller = caller;
      },
    },
    async (request: FastifyRequest, reply: FastifyReply) => {
      const caller = (request as FastifyRequest & { caller?: Caller }).caller;
      if (!caller) {
        // preHandler always sets this for an authenticated request; fail closed anyway.
        reply.status(401).send({ jsonrpc: '2.0', error: { code: -32001, message: 'Unauthorized' }, id: null });
        return;
      }
      const req = request.raw;
      const res = reply.raw;
      await mcpServer.handleRequest(req, res, request.body, caller);
      reply.hijack();
    }
  );

  // Stateless transport: no SSE stream to resume, so GET is not supported
  server.get('/mcp', async (_request: FastifyRequest, reply: FastifyReply) => {
    reply.status(405).send({
      jsonrpc: '2.0',
      error: { code: -32000, message: 'Method not allowed' },
      id: null,
    });
  });

  // gmail.authorize returns Google's URL directly from an authenticated MCP
  // request. The callback consumes one-time state bound to that exact write
  // credential; there is no unauthenticated start endpoint.
  server.get('/oauth/callback', googleOAuth.callbackHandler);

  return server;
}
