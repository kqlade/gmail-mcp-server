---
name: gmail-drafts
description: List all pending email drafts
---

# /gmail-drafts - View Drafts

List all saved email drafts.

## Steps

1. Use `gmail_status` to verify authentication
2. Use `gmail.manageDraft({ action: "list" })` with the read connection to get draft IDs
3. Use `gmail.manageDraft({ action: "get", draftId })` for the drafts whose recipients or subjects you need to show

## Output Format

```
📝 {count} drafts

• To: {recipient} - {subject} - draft {draftId}
• To: {recipient} - {subject} - draft {draftId}
• To: (no recipient) - {subject} - draft {draftId}
...
```

## Notes

- `list` returns only draft and message IDs; call `get` before showing recipient or subject
- Show recipient(s), subject, and draft ID
- If recipient is empty, show "(no recipient)"
- If subject is empty, show "(no subject)"
- Keep the order returned by Gmail
- If no drafts, let user know their drafts folder is empty
- Remind user that drafts can be sent from Gmail directly
