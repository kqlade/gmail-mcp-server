---
name: gmail-labels
description: List Gmail labels with targeted message counts
---

# /gmail-labels - View Labels

List label names and show counts only for labels fetched explicitly.

## Steps

1. Use `gmail_status` to verify authentication
2. Use `gmail_listLabels` to get IDs, names, and types
3. Use `gmail_getLabelInfo` for INBOX, UNREAD, STARRED, SENT, DRAFT, SPAM, and TRASH; fetch custom-label counts only when the user asks

## Output Format

```
🏷️ Gmail Labels

System Labels:
• INBOX: {total} ({unread} unread)
• SENT: {total}
• DRAFT: {count}
• STARRED: {count}
• SPAM: {count}
• TRASH: {count}

Custom Labels:
• {label_name}
• {label_name}
• {parent}/{child}
...
```

## Notes

- Separate system labels from custom labels
- `gmail_listLabels` does not return counts; never infer them
- Show unread count only when returned by `gmail_getLabelInfo`
- Show nested labels with their hierarchy (parent/child)
- Sort custom labels alphabetically
- If no custom labels exist, note that user can create them
