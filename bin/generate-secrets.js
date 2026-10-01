#!/usr/bin/env node

import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import readline from 'node:readline';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.join(__dirname, '..');
const envPath = path.join(projectRoot, '.env');
const envExamplePath = path.join(projectRoot, '.env.example');

// Colors
const colors = {
  reset: '\x1b[0m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  cyan: '\x1b[36m',
  dim: '\x1b[2m',
};

function log(msg) {
  console.log(msg);
}

function upsertEnv(content, name, value) {
  const pattern = new RegExp(`^${name}=.*$`, 'm');
  return pattern.test(content)
    ? content.replace(pattern, `${name}=${value}`)
    : `${content.replace(/\s*$/, '')}\n${name}=${value}\n`;
}

function envValue(content, name) {
  const match = content.match(new RegExp(`^${name}=(.*)$`, 'm'));
  return match?.[1]?.trim() ?? '';
}

function prompt(question) {
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  return new Promise((resolve) => {
    rl.question(question, (answer) => {
      rl.close();
      resolve(answer.toLowerCase());
    });
  });
}

// Generate secrets
const TOKEN_ENCRYPTION_KEY = crypto.randomBytes(32).toString('base64');
const MCP_READ_TOKEN = crypto.randomBytes(32).toString('base64url');
const MCP_WRITE_TOKEN = crypto.randomBytes(32).toString('base64url');

async function main() {
  const account = (await prompt('Google account email to pin these credentials to: ')).trim();
  if (!/^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(account)) {
    throw new Error('Enter a valid Google account email address');
  }
  const MCP_AUTH_TOKENS =
    `primary:${MCP_READ_TOKEN}:${account.toLowerCase()}:read,` +
    `primary:${MCP_WRITE_TOKEN}:${account.toLowerCase()}:write`;

  log('\nGenerated scoped credentials for any missing values. Existing secrets are never rotated; keep .env private.\n');

  // Check if .env exists
  if (fs.existsSync(envPath)) {
    const answer = await prompt('Append to existing .env file? (y/N) ');
    if (answer === 'y' || answer === 'yes') {
      let content = fs.readFileSync(envPath, 'utf-8');

      const existingEncryptionKey = envValue(content, 'TOKEN_ENCRYPTION_KEY');
      const existingScopedTokens = envValue(content, 'MCP_AUTH_TOKENS');
      const additions = [];

      // Rotating this key makes every stored OAuth refresh token and opaque
      // reply context unreadable. This helper only fills a missing key.
      if (!existingEncryptionKey) {
        content = upsertEnv(content, 'TOKEN_ENCRYPTION_KEY', TOKEN_ENCRYPTION_KEY);
        additions.push('TOKEN_ENCRYPTION_KEY');
      }
      // Likewise, migration is additive: keep a working legacy MCP_AUTH_TOKEN
      // until clients have switched and the operator removes it explicitly.
      if (!existingScopedTokens) {
        content = upsertEnv(content, 'MCP_AUTH_TOKENS', MCP_AUTH_TOKENS);
        additions.push('MCP_AUTH_TOKENS');
      }

      if (additions.length > 0) {
        fs.writeFileSync(envPath, content);
        log(`${colors.green}Added missing ${additions.join(' and ')} to .env${colors.reset}`);
        if (envValue(content, 'MCP_AUTH_TOKEN')) {
          log(`${colors.yellow}Kept legacy MCP_AUTH_TOKEN for cutover; remove it only after scoped clients are verified.${colors.reset}`);
        }
      } else {
        log('Existing encryption key and scoped credentials were preserved; nothing changed.');
      }
    } else {
      log('Skipped.');
    }
  } else if (fs.existsSync(envExamplePath)) {
    const answer = await prompt('Create .env from .env.example with these secrets? (y/N) ');
    if (answer === 'y' || answer === 'yes') {
      let content = fs.readFileSync(envExamplePath, 'utf-8');

      // Replace placeholder values
      content = upsertEnv(content, 'TOKEN_ENCRYPTION_KEY', TOKEN_ENCRYPTION_KEY);
      content = upsertEnv(content, 'MCP_AUTH_TOKENS', MCP_AUTH_TOKENS);

      fs.writeFileSync(envPath, content);
      log(`${colors.green}Created .env with secrets.${colors.reset}`);
      log(`${colors.dim}Edit it to add your Google OAuth credentials.${colors.reset}`);
    } else {
      log('Skipped.');
    }
  } else {
    log(`${colors.yellow}No .env or .env.example found.${colors.reset}`);
    log('Copy the secrets above into your environment configuration.');
  }

  log('');
}

main().catch(console.error);
