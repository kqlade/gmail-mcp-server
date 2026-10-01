import { describe, expect, it } from 'vitest';
import {
  createReplyContext,
  deriveReplyContextKey,
  openReplyContext,
} from '../../src/auth/replyContext.js';

const KEY = 'test-encryption-key-that-is-at-least-32-characters';
const DERIVED_KEY = deriveReplyContextKey(KEY);
const DATA = {
  mcpUserId: 'ola',
  messageId: 'message-1',
  threadId: 'thread-1',
  inReplyTo: '<message-1@example.com>',
  references: '<earlier@example.com>',
};

describe('reply context', () => {
  it('round-trips opaque reply metadata for the same compartment', () => {
    const token = createReplyContext(DATA, DERIVED_KEY);
    expect(token).not.toContain(DATA.messageId);
    expect(token).not.toContain(DATA.inReplyTo);
    expect(openReplyContext(token, 'ola', DERIVED_KEY)).toEqual(DATA);
  });

  it('rejects another compartment and tampering', () => {
    const token = createReplyContext(DATA, DERIVED_KEY);
    expect(() => openReplyContext(token, 'sam', DERIVED_KEY)).toThrow(/another compartment/);
    expect(() => openReplyContext(token.slice(0, -2) + 'xx', 'ola', DERIVED_KEY)).toThrow(/invalid/);
  });

  it('rejects another encryption key', () => {
    const token = createReplyContext(DATA, DERIVED_KEY);
    const otherKey = deriveReplyContextKey('another-encryption-key-that-is-long-enough');
    expect(() => openReplyContext(token, 'ola', otherKey)).toThrow(/invalid/);
  });

  it('bounds long References headers before emitting a usable context', () => {
    const token = createReplyContext(
      { ...DATA, references: Array.from({ length: 3000 }, (_, i) => `<m${i}@example.com>`).join(' ') },
      DERIVED_KEY
    );
    const opened = openReplyContext(token, 'ola', DERIVED_KEY);
    expect(opened.references?.length).toBeLessThanOrEqual(16_384);
    expect(opened.references).toContain('<m2999@example.com>');
  });

  it('never emits an envelope larger than the opener accepts', () => {
    expect(() => createReplyContext(
      { ...DATA, references: '"\\'.repeat(8192) },
      DERIVED_KEY
    )).toThrow(/too large/);
  });
});
