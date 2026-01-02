# Confirmation System Setup Guide

This guide walks you through setting up the web dashboard and email confirmation system for the Portfolio Intelligence Engine.

## Table of Contents

1. [Overview](#overview)
2. [GitHub Pages Setup](#github-pages-setup)
3. [GitHub Personal Access Token](#github-personal-access-token)
4. [Email Automation Options](#email-automation-options)
5. [Testing the Flow](#testing-the-flow)
6. [Troubleshooting](#troubleshooting)

---

## Overview

The confirmation system provides two ways to review and approve portfolio rotations:

1. **Web Dashboard** - Interactive UI with full control over each trade
2. **Email Reply** - Quick CONFIRM/DENY via email response

Both methods require secure token validation and expire after 24 hours.

### System Flow

```
Portfolio Validator (Monday 5 AM ET)
    ↓
Generate proposed CSV + dashboard JSON + token
    ↓
Commit to repo + Send email
    ↓
User reviews via dashboard OR replies to email
    ↓
GitHub Actions validates token + executes trades
    ↓
Update tracker + Send confirmation email
```

---

## GitHub Pages Setup

Enable GitHub Pages to host the web dashboard.

### Steps

1. **Go to Repository Settings**
   - Navigate to: `Settings` → `Pages`

2. **Configure Source**
   - **Source**: Deploy from a branch
   - **Branch**: `main` (or your default branch)
   - **Folder**: `/docs`

3. **Save and Wait**
   - GitHub will deploy your site to:
     ```
     https://valley-cortexcloud.github.io/ai-predications-app-final/dashboard/
     ```
   - First deployment takes 1-2 minutes

4. **Verify Dashboard Loads**
   - Visit: `https://valley-cortexcloud.github.io/ai-predications-app-final/dashboard/`
   - You should see: "Demo Mode - No token provided"
   - This confirms the dashboard is live

### Custom Domain (Optional)

If you want a custom domain like `portfolio.yourcompany.com`:

1. Add a `CNAME` file in `/docs/` with your domain
2. Configure DNS with your provider (CNAME record pointing to `valley-cortexcloud.github.io`)
3. Update `DASHBOARD_BASE_URL` in scripts

---

## GitHub Personal Access Token

The dashboard needs a PAT to trigger GitHub Actions workflows via `repository_dispatch`.

### Create PAT

1. **Go to GitHub Settings**
   - Navigate to: `Settings` → `Developer settings` → `Personal access tokens` → `Tokens (classic)`

2. **Generate New Token**
   - Click: `Generate new token (classic)`
   - **Note**: "Portfolio Dashboard API Access"
   - **Expiration**: 90 days (or longer)
   - **Scopes**: Select **only**:
     - ✅ `repo` (Full control of private repositories)
       - This includes `workflow` scope needed for `repository_dispatch`

3. **Copy Token**
   - **Important**: Copy the token immediately - you won't see it again
   - Format: `ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`

### Store Token Securely

**Option A: User-side (Recommended for personal use)**
- Store token in your password manager
- Dashboard will prompt for token when submitting actions
- Token is only used in browser, never stored server-side

**Option B: GitHub Secrets (For automated webhook)**
- Go to: `Settings` → `Secrets and variables` → `Actions`
- Add secret: `DASHBOARD_PAT` with your token
- Modify dashboard to use this for API calls

### Token Security

- ⚠️ **Never commit tokens to the repository**
- 🔒 **Keep tokens private** - they grant access to your repository
- 🔄 **Rotate regularly** - regenerate every 90 days
- 🎯 **Minimal scope** - only grant `repo` access, nothing more

---

## Email Automation Options

There are three ways to enable email reply confirmations:

### Option A: Zapier Free Tier (Easiest)

**Pros**: No coding, 5-minute setup, 100 tasks/month free  
**Cons**: Limited to 100 confirmations/month

**Setup**:

1. **Create Zapier Account**
   - Sign up at [zapier.com](https://zapier.com)
   - Free tier: 100 tasks/month

2. **Create New Zap**
   - **Trigger**: Gmail - "New Email"
   - **Filter**: From: `jvalley19@gmail.com` (your email)
   - **Filter**: Subject contains: "Portfolio Review Required"

3. **Add Action: Formatter**
   - **Input**: Email body
   - **Transform**: Extract
   - **Pattern**: `CONFIRM|DENY` (extract first occurrence)

4. **Add Action: Webhooks by Zapier**
   - **Action**: POST
   - **URL**: `https://api.github.com/repos/Valley-CortexCloud/ai-predications-app-final/dispatches`
   - **Headers**:
     ```json
     {
       "Accept": "application/vnd.github+json",
       "Authorization": "Bearer YOUR_PAT_HERE",
       "X-GitHub-Api-Version": "2022-11-28"
     }
     ```
   - **Payload**:
     ```json
     {
       "event_type": "email_confirmation",
       "client_payload": {
         "action": "{{extracted_action}}",
         "date": "{{extracted_date}}",
         "token": "{{extracted_token}}"
       }
     }
     ```

5. **Test Zap**
   - Send test email with "CONFIRM" in reply
   - Verify webhook triggers GitHub Action

### Option B: Google Apps Script (Free, Unlimited)

**Pros**: Completely free, unlimited use, more control  
**Cons**: Requires 15-20 minutes setup, some coding

**Setup**:

1. **Open Gmail Settings**
   - Go to: Gmail → Settings (gear icon) → "See all settings"
   - Navigate to: "Forwarding and POP/IMAP"
   - Enable IMAP

2. **Create Apps Script**
   - Go to: [script.google.com](https://script.google.com)
   - New Project: "Portfolio Email Monitor"

3. **Paste This Code**:

```javascript
function monitorPortfolioEmails() {
  // Search for unread emails from portfolio system
  const query = 'subject:"Portfolio Review Required" is:unread';
  const threads = GmailApp.search(query, 0, 10);
  
  threads.forEach(thread => {
    const messages = thread.getMessages();
    const lastMessage = messages[messages.length - 1];
    
    // Get reply if exists
    if (messages.length > 1) {
      const reply = messages[1].getPlainBody();
      
      // Extract action
      let action = null;
      if (reply.toUpperCase().includes('CONFIRM')) {
        action = 'CONFIRM';
      } else if (reply.toUpperCase().includes('DENY')) {
        action = 'DENY';
      }
      
      if (action) {
        // Extract date and token from original email
        const originalBody = messages[0].getPlainBody();
        const dateMatch = originalBody.match(/Rotation Date: (\d{4}-\d{2}-\d{2})/);
        const tokenMatch = originalBody.match(/Token: ([a-zA-Z0-9_-]+)/);
        
        if (dateMatch && tokenMatch) {
          // Trigger GitHub webhook
          triggerGitHubAction(action, dateMatch[1], tokenMatch[1]);
          
          // Mark as read
          thread.markRead();
          
          Logger.log(`Processed ${action} for ${dateMatch[1]}`);
        }
      }
    }
  });
}

function triggerGitHubAction(action, date, token) {
  const url = 'https://api.github.com/repos/Valley-CortexCloud/ai-predications-app-final/dispatches';
  
  const payload = {
    event_type: 'email_confirmation',
    client_payload: {
      action: action,
      date: date,
      token: token
    }
  };
  
  const options = {
    method: 'post',
    contentType: 'application/json',
    headers: {
      'Accept': 'application/vnd.github+json',
      'Authorization': 'Bearer YOUR_PAT_HERE',
      'X-GitHub-Api-Version': '2022-11-28'
    },
    payload: JSON.stringify(payload)
  };
  
  UrlFetchApp.fetch(url, options);
}
```

4. **Replace `YOUR_PAT_HERE`** with your GitHub PAT

5. **Set Up Trigger**
   - Click: Triggers (clock icon on left)
   - Add Trigger:
     - Function: `monitorPortfolioEmails`
     - Event source: Time-driven
     - Type: Minutes timer
     - Interval: Every 5 minutes
   - Save

6. **Authorize Script**
   - First run will prompt for Gmail permissions
   - Review and approve

### Option C: Manual Webhook (For Testing)

Use GitHub UI to manually trigger workflows during testing.

**Steps**:

1. **Go to Actions Tab**
   - Navigate to: Repository → Actions

2. **Select Workflow**
   - Choose: "Email Confirmation"

3. **Run Workflow**
   - Click: "Run workflow"
   - Fill in inputs:
     - Date: `2026-01-05`
     - Action: `CONFIRM` or `DENY`
     - Token: Copy from email or token file

4. **Submit**
   - Workflow will validate token and process action

---

## Testing the Flow

### End-to-End Test

1. **Trigger Portfolio Validation**
   - Go to: Actions → "Portfolio Validation (Monday)" → "Run workflow"
   - Wait for completion (~2-3 minutes)

2. **Check Email**
   - You should receive: "📊 Portfolio Review Required"
   - Email contains review link and token

3. **Open Dashboard**
   - Click review link in email
   - Dashboard loads with proposed changes
   - Token countdown shows time remaining

4. **Review and Modify**
   - Toggle SELL ↔ HOLD buttons
   - Toggle BUY ↔ SKIP buttons
   - Review summary at bottom

5. **Submit Confirmation**
   - Click: "✅ CONFIRM & EXECUTE"
   - Confirm dialog appears
   - Workflow triggers automatically

6. **Verify Execution**
   - Check: Actions → "Dashboard Confirmation"
   - Should show successful run
   - Check email for confirmation

7. **Verify Trades**
   - Log in to Alpaca paper account
   - Orders should appear in order history
   - Check `tracker.csv` updated in repo

### Test Email Reply (If Configured)

1. **Reply to Portfolio Email**
   - Type: `CONFIRM` (just one word)
   - Send reply

2. **Wait for Trigger**
   - Zapier/Apps Script should trigger within 5 minutes
   - Check: Actions → "Email Confirmation"

3. **Verify Execution**
   - Same as dashboard test above

---

## Troubleshooting

### Dashboard Issues

**Problem**: Dashboard shows "Failed to load portfolio data"

**Solutions**:
- Check GitHub Pages is enabled and deployed
- Verify date parameter in URL matches a generated JSON file
- Open browser console (F12) to see detailed error
- Ensure JSON file exists: `docs/dashboard/data/YYYY-MM-DD.json`

**Problem**: "Invalid token" error

**Solutions**:
- Verify token in URL matches token in JSON file
- Check token hasn't expired (24 hours)
- Ensure token wasn't already used

**Problem**: Dashboard loads but "Submit" doesn't work

**Solutions**:
- Check browser console for errors
- Verify PAT is valid and has `repo` scope
- Test with manual workflow trigger first

### Workflow Issues

**Problem**: Portfolio validation fails

**Solutions**:
- Check `XAI_API_KEY` secret is set correctly
- Verify `tracker.csv` file exists and is valid
- Check workflow logs for specific error

**Problem**: "Token file not found" in confirmation workflow

**Solutions**:
- Ensure portfolio validation completed successfully
- Check `data/portfolio/tokens/` directory has JSON file for date
- Verify git commit succeeded after validation

**Problem**: Trades not executing

**Solutions**:
- Check `ALPACA_API_KEY` and `ALPACA_API_SECRET` are set
- Verify using `--paper` flag for paper trading
- Check Alpaca API status
- Review trade executor logs in workflow

### Email Issues

**Problem**: Not receiving proposal emails

**Solutions**:
- Verify `EMAIL_USER` and `EMAIL_PASS` secrets are set
- Check Gmail "App Password" (not regular password)
- Look in spam folder
- Check workflow logs for email sending errors

**Problem**: Email replies not triggering workflows

**Solutions**:
- **Zapier**: Check Zap history for errors
- **Apps Script**: Check execution logs for errors
- Verify PAT in webhook is valid
- Test with manual workflow trigger first

---

## Security Best Practices

### Production Deployment

When moving from testing to production:

1. **Use Private Repository**
   - Keep portfolio data and strategies confidential
   - Tokens are stored in repo (private repo required)

2. **Rotate Credentials**
   - GitHub PAT: Every 90 days
   - Gmail App Password: Every 180 days
   - Alpaca API Keys: Annually

3. **Enable 2FA**
   - GitHub account
   - Gmail account
   - Alpaca account

4. **Monitor Access**
   - Review GitHub audit log monthly
   - Check Actions workflow runs for anomalies

5. **Switch to Live Trading Carefully**
   - Test thoroughly in paper mode first
   - Start with small position sizes
   - Remove `--paper` flag only when confident

### Token Management

- ✅ Tokens are cryptographically random (32 bytes)
- ✅ Tokens expire after 24 hours
- ✅ Tokens are single-use (marked as used after first confirmation)
- ✅ Tokens stored in private repo (not in public URLs except during use)

---

## Support

Having issues? Here's how to get help:

1. **Check Logs**
   - GitHub Actions → Workflow run → View logs
   - Browser console (F12) for dashboard issues

2. **Review Documentation**
   - `docs/PORTFOLIO_ARCHITECTURE.md` - System design
   - This file - Setup instructions

3. **Create Issue**
   - [Open GitHub Issue](https://github.com/Valley-CortexCloud/ai-predications-app-final/issues)
   - Include: Error message, workflow logs, steps to reproduce

4. **Email Support**
   - jvalley19@gmail.com
   - Include: Repository name, workflow run ID

---

## Next Steps

Once everything is working:

1. **Customize Email Template**
   - Edit `scripts/send_proposal_email.py`
   - Personalize subject/body

2. **Enhance Dashboard**
   - Modify `docs/dashboard/index.html`
   - Add charts, additional metrics

3. **Automate Fully**
   - Remove manual confirmation requirement
   - Add risk limits and circuit breakers
   - See "V2 Roadmap" in PORTFOLIO_ARCHITECTURE.md

4. **Monitor Performance**
   - Track win rate, Sharpe ratio
   - Compare vs buy-and-hold
   - Iterate on exit thresholds

---

**Version**: 1.0.0  
**Last Updated**: January 2026  
**License**: MIT
