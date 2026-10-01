---
name: gmail-cleanup
description: Identify emails to archive - old messages, newsletters, bulk cleanup candidates
---

# /gmail-cleanup - Inbox Cleanup

Identify emails that can be archived to reduce inbox clutter.

## Steps

1. Use `gmail_status` to verify authentication
2. Use `gmail_getLabelInfo` for INBOX to get total count
3. Use `gmail.searchMessages` with `queries` to find cleanup candidates:
   ```
   gmail.searchMessages({
     queries: [
       { query: "in:inbox older_than:30d", maxResults: 50 },
       { query: "in:inbox category:promotions", maxResults: 30 },
       { query: "in:inbox category:updates", maxResults: 30 },
       { query: "in:inbox from:noreply OR from:no-reply", maxResults: 30 },
       { query: "in:inbox unsubscribe older_than:7d", maxResults: 30 }
     ]
   })
   ```
4. Group results by sender/type for bulk actions
5. Calculate potential inbox reduction

## Output Format

```
🧹 Cleanup Candidates

Inbox: {total} messages

## Old Emails (>{count})
{count} emails older than 30 days
Top senders:
• {sender} - {count} emails
• {sender} - {count} emails

## Newsletters & Promotions ({count})
• {sender} - {count} emails
• {sender} - {count} emails
• {sender} - {count} emails

## Automated Notifications ({count})
• {sender/type} - {count} emails
• {sender/type} - {count} emails

## Suggested Actions

1. Archive all from {sender}? ({count} emails)
   → gmail.organizeMessages({ actions: [{ action: "archive", threadIds: [...] }] })

2. Archive promotions older than 7 days? ({count} emails)
   → gmail.organizeMessages({ actions: [{ action: "archive", threadIds: [...] }] })

3. Archive automated notifications? ({count} emails)
   → gmail.organizeMessages({ actions: [{ action: "archive", threadIds: [...] }] })

Potential reduction: ~{count} emails ({percent}% of inbox)
```

## Notes

- Never auto-archive - always present options and wait for user confirmation
- Group by sender to enable bulk actions
- Show the actual threadIds so user can confirm specific archives
- "unsubscribe" in body often indicates newsletters
- category:promotions and category:updates are Gmail's auto-categories
- Focus on safe-to-archive: old, automated, promotional
- Don't suggest archiving starred or flagged emails
- If inbox is already small (<50), tell user it's well-maintained
- Remind user that archived emails are still searchable
