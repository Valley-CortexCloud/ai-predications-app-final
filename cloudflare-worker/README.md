# Cloudflare Workers Portfolio Dashboard

This directory contains the Cloudflare Worker that serves the Portfolio Intelligence Dashboard and provides secure API endpoints for interacting with the private GitHub repository.

## 📁 Files

- **`worker.js`** - Main Cloudflare Worker implementation
  - Dashboard UI (embedded HTML)
  - API endpoints for data fetching and workflow triggering
  - Security validation and CORS handling

- **`wrangler.toml`** - Cloudflare Worker configuration
  - Worker name and compatibility settings
  - Public environment variables
  - Secret configuration instructions

- **`SETUP.md`** - Complete deployment guide
  - Step-by-step instructions for deploying to Cloudflare
  - GitHub PAT creation and configuration
  - Testing and troubleshooting

## 🚀 Quick Start

1. **Install Wrangler CLI**
   ```bash
   npm install -g wrangler
   ```

2. **Authenticate with Cloudflare**
   ```bash
   wrangler login
   ```

3. **Deploy the Worker**
   ```bash
   cd cloudflare-worker
   wrangler deploy
   ```

4. **Add GitHub PAT Secret**
   ```bash
   wrangler secret put GITHUB_PAT
   ```

5. **Test the Deployment**
   ```bash
   curl https://portfolio-dashboard.YOUR_SUBDOMAIN.workers.dev/api/health
   ```

For complete setup instructions, see [SETUP.md](./SETUP.md).

## 🔒 Security

- GitHub PAT is stored as a Cloudflare secret (never in code)
- Token validation on all API requests
- Replay attack prevention
- CORS headers for browser security
- All data fetched from private GitHub repository

## 📊 API Endpoints

- `GET /` - Serve dashboard UI
- `GET /api/health` - Health check
- `GET /api/data/:date?token=xxx` - Fetch portfolio data
- `POST /api/confirm` - Submit confirmation/denial

## 🛠️ Local Development

```bash
# Start local development server
wrangler dev

# View live logs
wrangler tail

# List deployments
wrangler deployments list
```

## 📝 Notes

- The repository can remain **PRIVATE** - the worker handles all data access
- No GitHub Pages needed
- Free tier supports 100,000 requests/day
- Automatic SSL/TLS encryption

## 📖 Documentation

- [Cloudflare Workers Documentation](https://developers.cloudflare.com/workers/)
- [Wrangler CLI Documentation](https://developers.cloudflare.com/workers/wrangler/)
- [Full Setup Guide](./SETUP.md)
