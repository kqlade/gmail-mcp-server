import Fastify, { FastifyInstance, FastifyRequest, FastifyReply } from 'fastify';
import type { Config } from '../config.js';
import type { TokenStore } from '../store/interface.js';
import type { McpServerInstance } from '../mcp/server.js';
import { createGoogleOAuth } from '../auth/googleOAuth.js';
import { checkBearerToken, checkStartToken } from '../auth/bearer.js';

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

  // MCP Streamable HTTP endpoint (stateless; bearer-token protected)
  server.post(
    '/mcp',
    {
      preHandler: async (request, reply) => {
        if (!checkBearerToken(request.headers.authorization, config.mcpAuthToken)) {
          reply.status(401).header('WWW-Authenticate', 'Bearer').send({
            jsonrpc: '2.0',
            error: { code: -32001, message: 'Unauthorized' },
            id: null,
          });
        }
      },
    },
    async (request: FastifyRequest, reply: FastifyReply) => {
      const req = request.raw;
      const res = reply.raw;
      await mcpServer.handleRequest(req, res, request.body);
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

  // Google OAuth endpoints. /oauth/start is opened in a browser (no bearer
  // header possible), so it requires a short-lived HMAC start token that only
  // the authenticated gmail.authorize tool can mint.
  server.get('/oauth/start', async (request: FastifyRequest, reply: FastifyReply) => {
    const query = request.query as { exp?: string; sig?: string };
    if (!checkStartToken(query.exp, query.sig, config.mcpAuthToken)) {
      reply.status(401).send({
        error: 'unauthorized',
        error_description: 'Missing or expired authorization link. Run gmail.authorize to get a fresh link.',
      });
      return;
    }
    return googleOAuth.startHandler(request, reply);
  });
  server.get('/oauth/callback', googleOAuth.callbackHandler);

  return server;
}
