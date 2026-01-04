# Cloudflare Workers Portfolio Dashboard Setup Guide

This guide walks you through deploying the Portfolio Intelligence Dashboard on Cloudflare Workers, replacing the need for GitHub Pages and allowing the repository to remain **PRIVATE**.

## Architecture Overview

```
                         ONE Cloudflare Worker
                    ┌─────────────────────────────────┐
                    │                                 │
   GET /            │  → Serve dashboard HTML         │
   GET /data/*.json │  → Fetch from GitHub raw files  │
   POST /api/confirm│  → Trigger GitHub workflow      │
                    │                                 │
                    │  Secrets stored securely:       │
                    │  - GITHUB_PAT                   │
                    │                                 │
                    └─────────────────────────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────────┐
                    │  GitHub (PRIVATE repo!)         │
                    │                                 │
                    │  - Stores data JSON files       │
                    │  - Runs workflows               │
                    │  - Never exposed publicly       │
                    └─────────────────────────────────┘
```

## Prerequisites

- A Cloudflare account (free tier works perfectly)
- Node.js 16+ installed on your local machine
- Access to the GitHub repository with admin permissions

## Step 1: Create a Cloudflare Account

1. Go to [https://dash.cloudflare.com/sign-up](https://dash.cloudflare.com/sign-up)
2. Sign up for a free account
3. Verify your email address
4. No credit card required for the free tier

## Step 2: Install Wrangler CLI

Wrangler is Cloudflare's official CLI tool for Workers.

```bash
# Install globally via npm
npm install -g wrangler

# Verify installation
wrangler --version
```

## Step 3: Authenticate Wrangler with Cloudflare

```bash
# This will open a browser window to authenticate
wrangler login
```

Follow the prompts to authorize Wrangler to access your Cloudflare account.

## Step 4: Create GitHub Personal Access Token (PAT)

The worker needs a GitHub PAT to:
- Fetch data from the private repository
- Trigger GitHub Actions workflows

### 4.1 Create the Token

1. Go to [https://github.com/settings/tokens/new](https://github.com/settings/tokens/new)
2. Click "Generate new token" → "Generate new token (classic)"
3. Configure the token:
   - **Note**: `Portfolio Dashboard Worker`
   - **Expiration**: Choose "No expiration" or set a long expiration (90 days+)
   - **Scopes**: Select the following:
     - ✅ `repo` (Full control of private repositories)
     - ✅ `workflow` (Update GitHub Action workflows)
4. Click "Generate token"
5. **IMPORTANT**: Copy the token immediately (you won't see it again!)
6. Store it securely (e.g., in a password manager)

### 4.2 Token Security Notes

- Never commit this token to the repository
- Never share it publicly
- The token will be stored securely in Cloudflare Workers secrets
- You can revoke and regenerate it anytime from GitHub settings

## Step 5: Deploy the Worker

Navigate to the `cloudflare-worker` directory:

```bash
cd cloudflare-worker
```

### 5.1 Deploy the Worker

```bash
wrangler deploy
```

This will:
- Build and package the worker
- Upload it to Cloudflare
- Provide you with a worker URL (e.g., `https://portfolio-dashboard.YOUR_SUBDOMAIN.workers.dev`)

**IMPORTANT**: Save this URL! You'll need it for Step 6.

### 5.2 Expected Output

```
 ⛅️ wrangler 3.x.x
------------------
Your worker has been deployed to:
  https://portfolio-dashboard.YOUR_SUBDOMAIN.workers.dev

✨ Done! Your worker is now live!
```

## Step 6: Add GitHub PAT Secret

Add your GitHub Personal Access Token as a secret to the worker:

```bash
wrangler secret put GITHUB_PAT
```

When prompted, paste your GitHub PAT (from Step 4) and press Enter.

**Verification**: The secret is now stored securely in Cloudflare. You can't retrieve it, but the worker can use it.

## Step 7: Test the Worker

### 7.1 Test Health Endpoint

```bash
curl https://portfolio-dashboard.YOUR_SUBDOMAIN.workers.dev/api/health
```

Expected response:
```json
{
  "status": "healthy",
  "timestamp": "2026-01-04T14:00:00.000Z",
  "version": "1.0.0"
}
```

### 7.2 Test Dashboard UI

Open in your browser:
```
https://portfolio-dashboard.YOUR_SUBDOMAIN.workers.dev/
```

You should see the dashboard UI (it will show an error about missing date parameter, which is expected).

### 7.3 Test Data Fetch (with real token and date)

If you have a real token and date from a portfolio validation run:

```bash
curl "https://portfolio-dashboard.YOUR_SUBDOMAIN.workers.dev/api/data/2026-01-03?token=YOUR_TOKEN"
```

This should return portfolio data JSON.

## Step 8: Update Repository Configuration

Now that your worker is deployed, you need to update the repository to use the new Cloudflare Worker URL instead of GitHub Pages.

### 8.1 Update Email Script

Edit `scripts/send_proposal_email.py` and change the `DASHBOARD_BASE_URL`:

```python
# OLD (GitHub Pages)
DASHBOARD_BASE_URL = "https://valley-cortexcloud.github.io/ai-predications-app-final/dashboard"

# NEW (Cloudflare Worker)
DASHBOARD_BASE_URL = "https://portfolio-dashboard.YOUR_SUBDOMAIN.workers.dev"
```

### 8.2 Update Dashboard Data Generator

Edit `scripts/generate_dashboard_data.py` and change the `DASHBOARD_BASE_URL`:

```python
# OLD (GitHub Pages)
DASHBOARD_BASE_URL = "https://valley-cortexcloud.github.io/ai-predications-app-final/dashboard"

# NEW (Cloudflare Worker)
DASHBOARD_BASE_URL = "https://portfolio-dashboard.YOUR_SUBDOMAIN.workers.dev"
```

### 8.3 Commit Changes

```bash
git add scripts/send_proposal_email.py scripts/generate_dashboard_data.py
git commit -m "Update dashboard URLs to use Cloudflare Worker"
git push
```

## Step 9: End-to-End Test

### 9.1 Trigger Portfolio Validation

Wait for the next Monday at 5 AM ET, or manually trigger the workflow:

```bash
gh workflow run portfolio-validation.yml
```

### 9.2 Check Email

You should receive an email with a dashboard link pointing to your Cloudflare Worker:
```
https://portfolio-dashboard.YOUR_SUBDOMAIN.workers.dev/?token=xxx&date=YYYY-MM-DD
```

### 9.3 Test Dashboard

1. Click the link in the email
2. The dashboard should load portfolio data
3. Toggle some actions (SELL/HOLD, BUY/SKIP)
4. Click "CONFIRM & EXECUTE"
5. Check that the GitHub workflow `dashboard-confirmation.yml` is triggered
6. Verify trades are executed in Alpaca paper trading account
7. Receive confirmation email

## Step 10: (Optional) Add Custom Domain

If you want to use a custom domain instead of `*.workers.dev`:

### 10.1 Requirements
- A domain registered with Cloudflare or transferred to Cloudflare
- The domain must be on Cloudflare's nameservers

### 10.2 Add Custom Domain

1. Go to [Cloudflare Workers Dashboard](https://dash.cloudflare.com/)
2. Navigate to **Workers & Pages** → **portfolio-dashboard**
3. Click **Settings** → **Domains & Routes**
4. Click **Add Custom Domain**
5. Enter your custom domain (e.g., `portfolio.yourdomain.com`)
6. Click **Add Domain**

Cloudflare will automatically:
- Create the DNS record
- Issue an SSL certificate
- Route traffic to your worker

### 10.3 Update Repository URLs

If using a custom domain, update the URLs in Step 8 to use your custom domain instead of the `*.workers.dev` URL.

## Troubleshooting

### Issue: "GitHub fetch failed: 404"

**Cause**: The GitHub PAT doesn't have access to the repository, or the file doesn't exist.

**Solution**:
1. Verify the repository is private
2. Check that the PAT has `repo` scope
3. Verify the file exists at the expected path in GitHub
4. Re-add the secret: `wrangler secret put GITHUB_PAT`

### Issue: "Token validation failed"

**Cause**: The token file doesn't exist in the repository or the token is invalid.

**Solution**:
1. Ensure portfolio validation workflow has run successfully
2. Check that `data/portfolio/tokens/*.json` files exist in the repo
3. Verify the token parameter matches the token in the file

### Issue: "Workflow trigger failed: 403"

**Cause**: The GitHub PAT doesn't have `workflow` scope.

**Solution**:
1. Go to GitHub token settings
2. Add the `workflow` scope
3. Re-add the secret: `wrangler secret put GITHUB_PAT`

### Issue: "CORS error in browser"

**Cause**: CORS headers are not being set correctly.

**Solution**:
- This should be automatically handled by the worker
- Check browser console for specific error
- Verify the worker is deployed with latest code

### Issue: Worker not updating after code changes

**Solution**:
```bash
cd cloudflare-worker
wrangler deploy
```

This will redeploy with the latest code.

## Monitoring and Logs

### View Live Logs

```bash
wrangler tail
```

This streams live logs from your worker. Useful for debugging.

### View Metrics

1. Go to [Cloudflare Workers Dashboard](https://dash.cloudflare.com/)
2. Navigate to **Workers & Pages** → **portfolio-dashboard**
3. View the **Analytics** tab for:
   - Requests per second
   - Error rates
   - CPU usage

## Rate Limits and Quotas

### Free Tier Limits (Cloudflare Workers)
- **Requests**: 100,000 per day
- **CPU Time**: 10ms per request
- **Worker Size**: 1 MB after compression

### GitHub API Rate Limits
- **Authenticated**: 5,000 requests per hour
- Our usage is minimal (1-2 requests per dashboard view)

## Security Best Practices

1. **Token Security**
   - Never commit GitHub PAT to repository
   - Use Wrangler secrets for sensitive data
   - Rotate tokens periodically

2. **Token Validation**
   - Worker validates token server-side
   - Checks token expiry
   - Prevents replay attacks

3. **Repository Privacy**
   - Keep GitHub repository PRIVATE
   - Never expose raw data publicly
   - All data access goes through authenticated worker

## Maintenance

### Update Worker Code

If you need to update the worker:

```bash
cd cloudflare-worker
# Edit worker.js
wrangler deploy
```

### Rotate GitHub PAT

1. Create a new GitHub PAT (Step 4)
2. Update the secret:
   ```bash
   wrangler secret put GITHUB_PAT
   ```
3. Revoke the old token on GitHub

### View Worker Deployments

```bash
wrangler deployments list
```

## Support and Resources

- **Cloudflare Workers Docs**: [https://developers.cloudflare.com/workers/](https://developers.cloudflare.com/workers/)
- **Wrangler CLI Docs**: [https://developers.cloudflare.com/workers/wrangler/](https://developers.cloudflare.com/workers/wrangler/)
- **GitHub PAT Docs**: [https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token)

## Migration Checklist

- [ ] Cloudflare account created
- [ ] Wrangler CLI installed and authenticated
- [ ] GitHub PAT created with `repo` and `workflow` scopes
- [ ] Worker deployed successfully
- [ ] GitHub PAT secret added to worker
- [ ] Health endpoint tested
- [ ] Dashboard UI tested
- [ ] Repository scripts updated with worker URL
- [ ] End-to-end test completed
- [ ] Email confirmation received
- [ ] Trades executed successfully

## What About GitHub Pages?

Once the Cloudflare Worker is deployed and tested, the `docs/dashboard/` directory is no longer needed and can be safely deleted. The repository can remain fully private.

The workflow (`portfolio-validation.yml`) will continue to generate dashboard data files (`docs/dashboard/data/*.json`), which the Cloudflare Worker fetches securely using the GitHub PAT.

---

**Questions?** Open an issue or refer to the main repository documentation.
