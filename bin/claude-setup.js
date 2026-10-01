#!/usr/bin/env node

import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// Paths
const CLAUDE_CONFIG = path.join(os.homedir(), '.claude.json');
const CLAUDE_SKILLS_DIR = path.join(os.homedir(), '.claude', 'skills');
const CLAUDE_AGENTS_DIR = path.join(os.homedir(), '.claude', 'agents');
const LOCAL_SKILLS_DIR = path.join(__dirname, '..', '.claude', 'skills');
const LOCAL_AGENTS_DIR = path.join(__dirname, '..', '.claude', 'agents');
const ENV_PATH = path.join(__dirname, '..', '.env');
const MCP_SERVER_NAMES = ['gmail-read', 'gmail-write'];
const LEGACY_MCP_SERVER_NAME = 'gmail-mcp';
const SKILL_PREFIX = 'gmail-';
const AGENT_PREFIX = 'gmail-';

// Colors for terminal output
const colors = {
  reset: '\x1b[0m',
  green: '\x1b[32m',
  red: '\x1b[31m',
  yellow: '\x1b[33m',
  cyan: '\x1b[36m',
  dim: '\x1b[2m',
};

function log(msg) {
  console.log(msg);
}

function success(msg) {
  console.log(`${colors.green}  + ${msg}${colors.reset}`);
}

function removed(msg) {
  console.log(`${colors.red}  - ${msg}${colors.reset}`);
}

function info(msg) {
  console.log(`${colors.dim}  ${msg}${colors.reset}`);
}

function header(msg) {
  console.log(`\n${colors.cyan}${msg}${colors.reset}`);
}

// Read Claude config file
function readClaudeConfig() {
  if (!fs.existsSync(CLAUDE_CONFIG)) {
    return {};
  }
  try {
    return JSON.parse(fs.readFileSync(CLAUDE_CONFIG, 'utf-8'));
  } catch {
    return {};
  }
}

// Write Claude config file
function writeClaudeConfig(config) {
  fs.writeFileSync(CLAUDE_CONFIG, JSON.stringify(config, null, 2) + '\n');
  fs.chmodSync(CLAUDE_CONFIG, 0o600);
}

function readLocalCredentials() {
  if (!fs.existsSync(ENV_PATH)) {
    throw new Error('Missing .env. Run npm run setup:secrets first.');
  }
  const env = {};
  for (const rawLine of fs.readFileSync(ENV_PATH, 'utf-8').split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line || line.startsWith('#')) continue;
    const equals = line.indexOf('=');
    if (equals < 1) continue;
    env[line.slice(0, equals)] = line.slice(equals + 1).replace(/^["']|["']$/g, '');
  }
  const rawEntries = String(env.MCP_AUTH_TOKENS ?? '')
    .split(/[\n,]/)
    .map((entry) => entry.trim().split(':'))
    .filter((parts) => parts.some(Boolean));
  const pairs = new Map();
  for (const parts of rawEntries) {
    const [id, token, account, capability] = parts;
    if (
      parts.length !== 4 ||
      !id ||
      !token ||
      !account ||
      (capability !== 'read' && capability !== 'write')
    ) {
      throw new Error('MCP_AUTH_TOKENS contains a malformed entry.');
    }
    const key = `${id.toLowerCase()}\0${account.toLowerCase()}`;
    const pair = pairs.get(key) ?? { id: id.toLowerCase(), account: account.toLowerCase() };
    if (pair[capability]) {
      throw new Error(`MCP_AUTH_TOKENS repeats ${capability} for ${pair.id}.`);
    }
    pair[capability] = token;
    pairs.set(key, pair);
  }
  const requestedId = String(env.CLAUDE_GMAIL_CALLER_ID ?? '').trim().toLowerCase();
  const completePairs = [...pairs.values()].filter(
    (pair) => pair.read && pair.write && (!requestedId || pair.id === requestedId)
  );
  if (completePairs.length !== 1) {
    throw new Error(
      completePairs.length === 0
        ? 'MCP_AUTH_TOKENS must contain a matching pinned read/write pair.'
        : 'Multiple caller pairs exist; set CLAUDE_GMAIL_CALLER_ID in .env to choose one.'
    );
  }
  const [{ read, write }] = completePairs;
  const baseUrl = String(env.BASE_URL || 'http://localhost:3000').replace(/\/+$/, '');
  return { read, write, url: `${baseUrl}/mcp` };
}

// Check if MCP server is installed
function isMcpServerInstalled() {
  const config = readClaudeConfig();
  return MCP_SERVER_NAMES.every((name) => config.mcpServers?.[name] !== undefined);
}

// Install MCP server to ~/.claude.json
function installMcpServer() {
  const config = readClaudeConfig();
  const credentials = readLocalCredentials();

  if (!config.mcpServers) {
    config.mcpServers = {};
  }

  const wasInstalled = isMcpServerInstalled();
  delete config.mcpServers[LEGACY_MCP_SERVER_NAME];
  config.mcpServers['gmail-read'] = {
    type: 'http',
    url: credentials.url,
    headers: { Authorization: `Bearer ${credentials.read}` },
  };
  config.mcpServers['gmail-write'] = {
    type: 'http',
    url: credentials.url,
    headers: { Authorization: `Bearer ${credentials.write}` },
  };

  writeClaudeConfig(config);

  if (wasInstalled) {
    info('Updated gmail-read and gmail-write in ~/.claude.json');
  } else {
    success('Added authenticated gmail-read and gmail-write connections to ~/.claude.json');
  }

  return true;
}

// Uninstall MCP server from ~/.claude.json
function uninstallMcpServer() {
  const config = readClaudeConfig();
  const names = [...MCP_SERVER_NAMES, LEGACY_MCP_SERVER_NAME];
  const installed = names.filter((name) => config.mcpServers?.[name]);
  if (installed.length > 0) {
    for (const name of installed) delete config.mcpServers[name];
    writeClaudeConfig(config);
    removed(`Removed ${installed.join(', ')} from ~/.claude.json`);
    return true;
  }

  info('Gmail MCP connections not found in ~/.claude.json');
  return false;
}

// Get list of skill files from local directory (returns skill names without .md)
function getLocalSkills() {
  if (!fs.existsSync(LOCAL_SKILLS_DIR)) {
    return [];
  }
  return fs.readdirSync(LOCAL_SKILLS_DIR)
    .filter(f => f.endsWith('.md') && f.startsWith(SKILL_PREFIX))
    .map(f => f.replace('.md', ''));
}

// Get list of installed gmail skill directories
function getInstalledSkills() {
  if (!fs.existsSync(CLAUDE_SKILLS_DIR)) {
    return [];
  }
  return fs.readdirSync(CLAUDE_SKILLS_DIR)
    .filter(f => {
      const skillPath = path.join(CLAUDE_SKILLS_DIR, f);
      const skillFile = path.join(skillPath, 'SKILL.md');
      return f.startsWith(SKILL_PREFIX) &&
             fs.statSync(skillPath).isDirectory() &&
             fs.existsSync(skillFile);
    });
}

// Install skills by creating subdirectories with SKILL.md
function installSkills() {
  const skills = getLocalSkills();

  if (skills.length === 0) {
    info(`No skills found in ${LOCAL_SKILLS_DIR}`);
    return 0;
  }

  // Create skills directory if needed
  if (!fs.existsSync(CLAUDE_SKILLS_DIR)) {
    fs.mkdirSync(CLAUDE_SKILLS_DIR, { recursive: true });
  }

  let installed = 0;
  for (const skillName of skills) {
    const src = path.join(LOCAL_SKILLS_DIR, `${skillName}.md`);
    const skillDir = path.join(CLAUDE_SKILLS_DIR, skillName);
    const dest = path.join(skillDir, 'SKILL.md');

    // Create skill subdirectory
    if (!fs.existsSync(skillDir)) {
      fs.mkdirSync(skillDir, { recursive: true });
    }

    fs.copyFileSync(src, dest);
    success(skillName);
    installed++;
  }

  return installed;
}

// Uninstall skills by removing directories
function uninstallSkills() {
  const skills = getInstalledSkills();

  if (skills.length === 0) {
    info('No Gmail skills found in ~/.claude/skills/');
    return 0;
  }

  let removedCount = 0;
  for (const skillName of skills) {
    const skillDir = path.join(CLAUDE_SKILLS_DIR, skillName);
    fs.rmSync(skillDir, { recursive: true, force: true });
    removed(skillName);
    removedCount++;
  }

  return removedCount;
}

// Get list of agent files from local directory (returns agent names without .md)
function getLocalAgents() {
  if (!fs.existsSync(LOCAL_AGENTS_DIR)) {
    return [];
  }
  return fs.readdirSync(LOCAL_AGENTS_DIR)
    .filter(f => f.endsWith('.md') && f.startsWith(AGENT_PREFIX))
    .map(f => f.replace('.md', ''));
}

// Get list of installed gmail agents
function getInstalledAgents() {
  if (!fs.existsSync(CLAUDE_AGENTS_DIR)) {
    return [];
  }
  return fs.readdirSync(CLAUDE_AGENTS_DIR)
    .filter(f => f.endsWith('.md') && f.startsWith(AGENT_PREFIX))
    .map(f => f.replace('.md', ''));
}

// Install agents by copying files
function installAgents() {
  const agents = getLocalAgents();

  if (agents.length === 0) {
    info(`No agents found in ${LOCAL_AGENTS_DIR}`);
    return 0;
  }

  // Create agents directory if needed
  if (!fs.existsSync(CLAUDE_AGENTS_DIR)) {
    fs.mkdirSync(CLAUDE_AGENTS_DIR, { recursive: true });
  }

  let installed = 0;
  for (const agentName of agents) {
    const src = path.join(LOCAL_AGENTS_DIR, `${agentName}.md`);
    const dest = path.join(CLAUDE_AGENTS_DIR, `${agentName}.md`);
    fs.copyFileSync(src, dest);
    success(agentName);
    installed++;
  }

  return installed;
}

// Uninstall agents by removing files
function uninstallAgents() {
  const agents = getInstalledAgents();

  if (agents.length === 0) {
    info('No Gmail agents found in ~/.claude/agents/');
    return 0;
  }

  let removedCount = 0;
  for (const agentName of agents) {
    const agentFile = path.join(CLAUDE_AGENTS_DIR, `${agentName}.md`);
    fs.unlinkSync(agentFile);
    removed(agentName);
    removedCount++;
  }

  return removedCount;
}

// Show installation status
function showStatus() {
  log('\nGmail MCP - Installation Status\n');

  // MCP Server status
  header('MCP Server');
  if (isMcpServerInstalled()) {
    const config = readClaudeConfig();
    success(`Installed read/write (${config.mcpServers['gmail-read'].url})`);
  } else {
    info('Not installed');
  }

  // Agents status
  header('Subagents');
  const installedAgents = getInstalledAgents();
  if (installedAgents.length > 0) {
    success(`${installedAgents.length} subagent(s) installed (auto-triggered):`);
    for (const agentName of installedAgents) {
      info(agentName);
    }
  } else {
    info('No subagents installed');
  }

  // Skills status
  header('Skills');
  const installedSkills = getInstalledSkills();
  if (installedSkills.length > 0) {
    success(`${installedSkills.length} skill(s) installed (explicit /command):`);
    for (const skillName of installedSkills) {
      info(`/${skillName}`);
    }
  } else {
    info('No skills installed');
  }

  log('');
}

// Main setup command
function setup() {
  log('\nGmail MCP - Claude Code Setup\n');

  header('MCP Server');
  installMcpServer();

  header('Subagents');
  const agentCount = installAgents();

  header('Skills');
  const skillCount = installSkills();

  log(`\n${colors.green}Done!${colors.reset} ${agentCount} subagent(s) + ${skillCount} skill(s) installed.`);
  log(`\nSubagents auto-trigger on context (e.g., "prioritize my inbox")`);
  log(`Skills run explicitly (e.g., ${colors.cyan}/gmail-inbox${colors.reset})\n`);
}

// Main uninstall command
function uninstall() {
  log('\nGmail MCP - Uninstall\n');

  header('MCP Server');
  uninstallMcpServer();

  header('Subagents');
  const agentCount = uninstallAgents();

  header('Skills');
  const skillCount = uninstallSkills();

  log(`\n${colors.green}Done!${colors.reset} Removed MCP server, ${agentCount} subagent(s), and ${skillCount} skill(s).\n`);
}

// Parse command and run
const command = process.argv[2] || 'setup';

switch (command) {
  case 'setup':
  case 'install':
    setup();
    break;
  case 'uninstall':
  case 'remove':
    uninstall();
    break;
  case 'status':
    showStatus();
    break;
  default:
    log(`Unknown command: ${command}`);
    log('Usage: node claude-setup.js [setup|uninstall|status]');
    process.exit(1);
}
