# Cloudflare Workers Migration Guide

## Overview

This repository has migrated from GitHub Pages to Cloudflare Workers for hosting the Portfolio Intelligence Dashboard. This change enables:

1. ✅ **Private Repository** - No need to make the repo public for GitHub Pages
2. ✅ **Secure Data Access** - Worker fetches data from private GitHub repo using PAT
3. ✅ **Direct Workflow Triggering** - No need for manual workflow dispatch
4. ✅ **Better Security** - Server-side token validation and replay protection
5. ✅ **No External Dependencies** - Single worker handles everything

## Architecture Change

### Before (GitHub Pages)
```
User → GitHub Pages (public) → Static JSON files (public)
      → Manual workflow trigger
```

**Problem**: Requires repository to be public or GitHub Pages to be enabled on private repos (paid feature).

### After (Cloudflare Workers)
```
User → Cloudflare Worker → GitHub API (authenticated) → Private Repo
                         → Triggers workflow directly
```

**Benefit**: Repository stays private, all access is authenticated and secure.

## What Changed

### New Files
- `cloudflare-worker/worker.js` - Complete worker implementation
- `cloudflare-worker/wrangler.toml` - Worker configuration
- `cloudflare-worker/SETUP.md` - Deployment instructions
- `cloudflare-worker/README.md` - Quick reference

### Modified Files
- `scripts/send_proposal_email.py` - Now uses `DASHBOARD_URL` environment variable
- `scripts/generate_dashboard_data.py` - Now uses `DASHBOARD_URL` environment variable
- `.github/workflows/portfolio-validation.yml` - Added `DASHBOARD_URL` secret
- `.gitignore` - Added Cloudflare worker artifacts

### Deprecated Files
- `docs/dashboard/index.html` - **Can be deleted after migration**
  - The HTML is now embedded in the Cloudflare Worker
  - No longer served from GitHub Pages

Note: The `docs/dashboard/data/*.json` files are still generated and used by the worker to fetch data.

## Migration Steps

### 1. Deploy Cloudflare Worker

Follow the instructions in [`cloudflare-worker/SETUP.md`](../cloudflare-worker/SETUP.md).

### 2. Update GitHub Secrets

Add a new repository secret:
- **Name**: `DASHBOARD_URL`
- **Value**: Your Cloudflare Worker URL (e.g., `https://portfolio-dashboard.YOUR_SUBDOMAIN.workers.dev`)

Go to: `Settings` → `Secrets and variables` → `Actions` → `New repository secret`

### 3. Test End-to-End

1. Trigger the `portfolio-validation` workflow manually
2. Check that the email contains the new Cloudflare Worker URL
3. Click the dashboard link and verify it loads
4. Test confirmation submission
5. Verify workflow triggers and trades execute

### 4. Clean Up (Optional)

Once verified, you can:
- Delete `docs/dashboard/index.html` (no longer needed)
- Keep `docs/dashboard/data/` directory (still used by worker)

## Rollback Plan

If you need to roll back to GitHub Pages:

1. Revert changes to `scripts/send_proposal_email.py` and `scripts/generate_dashboard_data.py`
2. Change `DASHBOARD_BASE_URL` back to GitHub Pages URL
3. Re-enable GitHub Pages in repository settings
4. The old `docs/dashboard/index.html` will still work (don't delete it yet)

## Benefits

### Security
- ✅ Repository can remain fully private
- ✅ GitHub PAT stored securely in Cloudflare
- ✅ Server-side token validation
- ✅ Replay attack prevention
- ✅ No public exposure of data files

### Performance
- ✅ Global CDN (Cloudflare's edge network)
- ✅ Fast response times
- ✅ Automatic SSL/TLS
- ✅ No cold starts

### Cost
- ✅ Free tier: 100,000 requests/day
- ✅ No GitHub Pages requirements
- ✅ No external services needed

### Maintenance
- ✅ Single deployment (`wrangler deploy`)
- ✅ Easy updates and rollbacks
- ✅ Built-in monitoring and logs

## Support

For deployment help, see:
- [`cloudflare-worker/SETUP.md`](../cloudflare-worker/SETUP.md) - Full setup guide
- [`cloudflare-worker/README.md`](../cloudflare-worker/README.md) - Quick reference
- [Cloudflare Workers Docs](https://developers.cloudflare.com/workers/)

## Timeline

- **Created**: 2026-01-04
- **Status**: Ready for deployment
- **Migration**: Follow SETUP.md to deploy
