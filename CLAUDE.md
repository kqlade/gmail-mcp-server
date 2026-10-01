# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Development Commands

```bash
npm run build          # Compile TypeScript to dist/
npm run start          # Run compiled server
npm run dev            # Dev mode with hot reload (tsx watch)
npm run test           # Run unit tests (vitest)
npm run test:coverage  # Tests with coverage report
npm run lint           # ESLint
npm run typecheck      # Type checking only
```

Run a single test file: `npx vitest run tests/unit/crypto.test.ts`

### Claude Code Integration

```bash
npm run claude:setup      # Install MCP server + subagents + skills
npm run claude:status     # Check what's installed
npm run claude:uninstall  # Remove integration
npm run setup:secrets     # Generate encryption keys for .env
```

## Architecture Overview

This is an MCP (Model Context Protocol) server that exposes Gmail inbox tools via Streamable HTTP transport. It implements bearer capability authentication and Google OAuth for Gmail access. Each token names **one person's identity** and carries a server-enforced read or write capability; an identity may be pinned to one Google account, and no caller can target another person's mailbox.

### Core Data Flow

```
MCP Client → Fastify HTTP (/mcp) → MCP Server → Gmail Client → Google APIs
                ↓
        Token Store (SQLite) ← encrypted credentials
```

### Key Modules

- **`src/index.ts`** - Entry point; initializes config, token store, MCP server, HTTP server
- **`src/config.ts`** - Zod-validated environment configuration with lazy initialization
- **`src/http/server.ts`** - Fastify server exposing `/mcp`, `/oauth/callback`, and `/healthz`
- **`src/mcp/server.ts`** - Capability-filtered MCP tools with Zod input schemas
- **`src/gmail/client.ts`** - Gmail API wrapper with token refresh logic (5-min threshold, exponential backoff)
- **`src/auth/bearer.ts`** - Resolves bearer credentials to callers
- **`src/auth/replyContext.ts`** - Encrypts reply metadata passed from read to write
- **`src/auth/googleOAuth.ts`** - Google OAuth flow with PKCE support and state management
- **`src/store/sqlite.ts`** - SQLite token store with composite key `(mcp_user_id, email)` for multi-account; refresh tokens encrypted with AES-256-GCM

### Authentication Layers

1. **MCP-level**: static high-entropy bearer credentials mapped to a caller and capability
2. **Gmail-level**: Google OAuth tokens stored per MCP user; supports readonly, labels, modify, compose scopes

### Callers

- A credential is one bearer token plus the identity and capability it unlocks (`Caller` in `src/config.ts`). `MCP_AUTH_TOKENS` accepts only exact `id:token:account:read` or `id:token:account:write` entries. Use separate read and write entries with the same id/account. The legacy `MCP_AUTH_TOKEN` is all-capability caller `primary` and requires `GMAIL_ACCOUNT`.
- The token alone selects the `mcpUserId`; nothing in a request body can name another person. `buildServer(caller)` in `src/mcp/server.ts` closes every tool over the caller.
- Every caller is pinned to one `account`; the OAuth callback rejects any other. There is no per-call `email` parameter.
- `gmail.authorize` creates one-time PKCE state bound to the exact write credential and requested scopes, then returns Google's URL directly. There is no unauthenticated `/oauth/start` endpoint.
- Credential resolution happens server-side in `getValidCredentials` (client.ts), with a case-insensitive fallback against the stored address for pinned callers
- `gmail.listAccounts` is read-only visibility; `gmail.setDefaultAccount` / `gmail.removeAccount` were removed on purpose
- Database keeps the composite primary key `(mcp_user_id, email)` from the multi-account era

### Tool Categories

- Status/Auth: `gmail.status`, `gmail.authorize`
- Account Visibility: `gmail.listAccounts` (read-only)
- Read: `gmail.searchMessages` (single or batch), `gmail.getMessage`, `gmail.listThreads`, `gmail.getThread`, `gmail.getAttachmentMetadata`, `gmail.triageSnapshot`
- Labels: `gmail.getLabelInfo`, `gmail.listLabels`, `gmail.createLabel`
- Modifications: `gmail.organizeMessages`, `gmail.sendMessage`
- Drafts: `gmail.manageDraft` (`get`/`list` on read; mutations on write)

### Error Handling Pattern

Custom error classes in `src/utils/errors.ts` map to MCP JSON-RPC codes:
- `NotAuthorizedError` / `InsufficientScopeError` → -32001
- `InvalidArgumentError` / `AccountNotFoundError` → -32602
- `GmailApiError` / `RateLimitedError` → -32000

### Required Environment Variables

See `.env.example`. Key variables: `PORT`, `BASE_URL`, `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, `OAUTH_REDIRECT_URI`, `TOKEN_ENCRYPTION_KEY`, `MCP_AUTH_TOKENS`

### Token Refresh Behavior

Gmail client proactively refreshes access tokens 5 minutes before expiry. On `invalid_grant` (revocation), tokens are cleared and user must re-authorize. Transient errors retry with exponential backoff (max 3 attempts).

### Thread vs Message Semantics

Gmail's inbox is **thread-based**. Important implications:

1. **Inbox visibility**: A thread appears in inbox if ANY message has the `INBOX` label
2. **Archive behavior**: To remove a thread from inbox, ALL messages must be archived
3. **API operations**:
   - `messageIds` - operates on individual messages
   - `threadIds` - operates on all messages in the thread

**Recommendation for archive operations**: pass `threadIds` returned by the
read surface to `gmail.organizeMessages`. The write surface deliberately does
not resolve message IDs with a Gmail read. Passing `messageIds` modifies only
those messages, so the thread may remain in the inbox.

### Search Results

`searchMessages` returns `messageCount` for each enriched result, indicating how many messages are in that thread. This helps identify multi-message conversations for thread-aware operations.

### Claude Code Subagents & Skills

The project includes Claude Code integration with two types of automation:

**Subagents** (`.claude/agents/`) - Auto-triggered by context:
- `gmail-triage` - Prioritize inbox ("what needs attention")
- `gmail-cleanup` - Find archive candidates ("clean up inbox")
- `gmail-explore` - Inbox analysis ("analyze my inbox")
- `gmail-research` - Search/summarize ("find emails about X")

**Skills** (`.claude/skills/`) - Explicit `/command` invocation:
- 14 skills for quick lookups and interactive workflows
- Run `/gmail-inbox`, `/gmail-search`, `/gmail-compose`, etc.

Subagents run in isolated context and return summaries. Skills run in the conversation context.
