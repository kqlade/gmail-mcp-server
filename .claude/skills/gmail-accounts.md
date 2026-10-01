---
name: gmail-accounts
description: View the Gmail account pinned to this connection
---

# /gmail-accounts - View Connected Accounts

Show the account this caller is pinned to and whether it is connected.

## Steps

1. Use `gmail_status` to check connection status
2. Use `gmail_listAccounts` to get all connected accounts

## Output Format

```
📧 Connected Gmail Accounts

• {email} (default) ✓
• {email}
• {email}

The bearer credential is pinned to this account; it cannot switch mailboxes.
```

### If no accounts connected:
```
📧 No Gmail accounts connected

Use /gmail-connect to authorize a Gmail account
```

## Notes

- Clearly indicate the pinned account and connection status
- There is no per-call email parameter or default-account mutation
- To use another mailbox, provision a separate pinned caller credential
