# Gmail MCP Server

An MCP (Model Context Protocol) server that exposes Gmail inbox tools via Streamable HTTP transport. Connect your Gmail accounts to Claude Code and other MCP-compatible clients to search, read, organize, and compose emails.

## Features

- **Pinned account** - The server operates on one hardcoded Gmail account; callers can never route to another
- **Full Gmail access** - Search messages, read threads, manage labels, archive, star, and compose drafts
- **Secure token storage** - Refresh tokens encrypted with AES-256-GCM in SQLite
- **Two-layer OAuth** - MCP-level JWT authentication plus Google OAuth for Gmail
- **Docker ready** - Run with Docker Compose for easy deployment

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/yourusername/gmail-mcp.git
cd gmail-mcp
npm install
```

### 2. Configure Google OAuth

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Enable the Gmail API
4. Create OAuth 2.0 credentials (Web application)
5. Set authorized redirect URI to `http://localhost:3000/oauth/callback`
6. Copy the Client ID and Client Secret

### 3. Set up environment

```bash
# Generate secure secrets
npm run setup:secrets

# Edit .env with your Google OAuth credentials
# Set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET
```

### 4. Build and run

```bash
npm run build
npm run start
```

The server runs on `http://localhost:3000`.

### 5. Connect to Claude Code

```bash
npm run claude:setup
```

This installs the MCP server connection and Gmail skills.

## Claude Code Integration

### Setup

After the server is running, install the Claude Code integration:

```bash
npm run claude:setup
```

This installs:
- MCP server connection to `~/.claude.json`
- 4 subagents to `~/.claude/agents/` (auto-triggered by context)
- 14 skills to `~/.claude/skills/` (explicit `/command`)

### Subagents (Auto-Triggered)

Subagents run automatically when Claude detects matching context. They operate in isolated context and return concise summaries.

| Subagent | Triggers on... |
|----------|----------------|
| `gmail-triage` | "prioritize my inbox", "what needs attention", "urgent emails" |
| `gmail-cleanup` | "clean up inbox", "what can I archive", "declutter" |
| `gmail-explore` | "analyze my inbox", "email patterns", "inbox overview" |
| `gmail-research` | "find emails about X", "summarize emails from Y" |

**Examples:**
```
> What needs my attention in email?
→ Auto-triggers gmail-triage subagent

> Find all emails about the project proposal
→ Auto-triggers gmail-research subagent
```

### Skills (Explicit Commands)

Skills run when you explicitly type `/command`. They run in your conversation context.

| Command | Description |
|---------|-------------|
| `/gmail-inbox` | Quick inbox status |
| `/gmail-unread` | List unread messages |
| `/gmail-starred` | View starred messages |
| `/gmail-labels` | View all labels |
| `/gmail-drafts` | View pending drafts |
| `/gmail-search` | Search with Gmail query syntax |
| `/gmail-thread` | View a conversation thread |
| `/gmail-compose` | Draft new emails |
| `/gmail-summarize` | Summarize email threads |
| `/gmail-triage` | Prioritize unread emails |
| `/gmail-cleanup` | Archive and organize inbox |
| `/gmail-explore` | Comprehensive inbox overview |
| `/gmail-connect` | Connect Gmail account |
| `/gmail-accounts` | View connected accounts |

### Management

```bash
npm run claude:status     # Check what's installed
npm run claude:uninstall  # Remove integration
```

## Configuration

Create a `.env` file based on `.env.example`:

| Variable | Description | Required |
|----------|-------------|----------|
| `PORT` | Server port (default: 3000) | No |
| `BASE_URL` | Public URL for OAuth callbacks | Yes |
| `GOOGLE_CLIENT_ID` | Google OAuth client ID | Yes |
| `GOOGLE_CLIENT_SECRET` | Google OAuth client secret | Yes |
| `OAUTH_REDIRECT_URI` | OAuth callback URL | Yes |
| `TOKEN_ENCRYPTION_KEY` | 32-byte base64 key for token encryption | Yes |
| `MCP_AUTH_TOKENS` | Capability credentials: `id:token[:account[:read\|write\|read+write]]` entries, comma- or newline-separated. Use separate `read` and `write` tokens with the same id/account | One of these |
| `MCP_AUTH_TOKEN` | Legacy single caller (`primary`); `GMAIL_ACCOUNT` optionally pins its account | One of these |
| `DB_URL` | SQLite database path (default: `./data/gmail-mcp.db`) | No |
| `ALLOWED_ORIGINS` | Comma-separated CORS origins | No |

## Available Tools

### Authorization & Status
| Tool | Description |
|------|-------------|
| `gmail.status` | Check whether this caller's account is connected |
| `gmail.authorize` | Initiate OAuth flow to connect this caller's account |

### Account Visibility
| Tool | Description |
|------|-------------|
| `gmail.listAccounts` | List this caller's connected accounts (read-only; the operating account is resolved from the caller) |

### Reading Email
| Tool | Description |
|------|-------------|
| `gmail.searchMessages` | Search using Gmail query syntax |
| `gmail.batchSearchMessages` | Run multiple searches in parallel |
| `gmail.getMessage` | Get a single message by ID |
| `gmail.listThreads` | List conversation threads |
| `gmail.getThread` | Get all messages in a thread |
| `gmail.getAttachmentMetadata` | Get attachment info |

### Labels
| Tool | Description |
|------|-------------|
| `gmail.listLabels` | List all labels with counts |
| `gmail.getLabelInfo` | Get details about a specific label |
| `gmail.createLabel` | Create a custom label |
| `gmail.addLabels` | Add labels to messages/threads |
| `gmail.removeLabels` | Remove labels from messages/threads |

### Modifications
| Tool | Description |
|------|-------------|
| `gmail.archiveMessages` | Archive messages/threads |
| `gmail.unarchiveMessages` | Move back to inbox |
| `gmail.markAsRead` | Mark as read |
| `gmail.markAsUnread` | Mark as unread |
| `gmail.starMessages` | Add star |
| `gmail.unstarMessages` | Remove star |

### Drafts
| Tool | Description |
|------|-------------|
| `gmail.createDraft` | Create a new draft |
| `gmail.listDrafts` | List all drafts |
| `gmail.getDraft` | Get draft content |
| `gmail.updateDraft` | Update an existing draft |
| `gmail.deleteDraft` | Delete a draft |

## OAuth Scopes

Request only the scopes you need:

| Scope | Capabilities |
|-------|--------------|
| `gmail.readonly` | Search, read messages and threads |
| `gmail.labels` | Manage labels, archive, star, mark read/unread |
| `gmail.modify` | All of the above |
| `gmail.compose` | Create and manage drafts |

## Callers

Every request carries a bearer token. `MCP_AUTH_TOKENS` maps it to a caller id
(a person's handle), a server-enforced `read` or `write` capability, and,
optionally, the one Google account that caller may connect. Credentials are
stored per caller, so two people share one deployment without seeing each
other's mail, and a read token cannot advertise or invoke mutation tools.

- Every tool resolves credentials for the caller's account; there is no
  per-call `email` parameter. Agent callers used to hallucinate addresses and
  burn entire sessions on "account not connected" retry loops.
- A pinned credential (`id:token:account:read` or `:write`) can only connect that account: the
  OAuth callback rejects any other, so a wrong pick in the account chooser
  can't create a half-connected state. An unpinned caller operates on
  whichever account they connected.
- `gmail.authorize` links are signed with the caller's own token, so a link
  minted for one person cannot land a grant in another person's row.
- `gmail.setDefaultAccount` / `gmail.removeAccount` were removed: switching is
  meaningless and disconnecting would let a confused agent brick email access.

Adding a person: mint two tokens (`openssl rand -base64 32` twice), append
`handle:read-token:their@address:read` and
`handle:write-token:their@address:write` to `MCP_AUTH_TOKENS`, redeploy, put
them in Ripple as `MCP_GMAIL_READ_API_KEY` and
`MCP_GMAIL_WRITE_API_KEY`, and have them run `gmail.authorize` through the
write surface. The legacy `MCP_AUTH_TOKEN` remains an all-capability migration
fallback for an existing single-person deployment. For a zero-downtime
cutover, capability entries using id `primary` may temporarily coexist with
the legacy token; switch clients to the scoped tokens, verify them, then remove
`MCP_AUTH_TOKEN`.

## Docker Deployment

```bash
# Build and start
docker compose up -d

# View logs
docker compose logs -f

# Stop
docker compose down
```

The SQLite database persists in `./data/`.

## Development

```bash
# Dev mode with hot reload
npm run dev

# Run tests
npm run test

# Type check
npm run typecheck

# Lint
npm run lint
```

## Architecture

```
MCP Client → Fastify HTTP (/mcp) → MCP Server → Gmail Client → Google APIs
                    ↓
            Token Store (SQLite) ← encrypted credentials
```

- **`src/index.ts`** - Entry point
- **`src/http/server.ts`** - Fastify server with `/mcp`, `/oauth/*`, `/healthz` endpoints
- **`src/mcp/server.ts`** - MCP server with 26 registered tools
- **`src/gmail/client.ts`** - Gmail API wrapper with token refresh
- **`src/auth/`** - MCP JWT auth and Google OAuth flows
- **`src/store/sqlite.ts`** - SQLite token store with AES-256-GCM encryption

## License

MIT
