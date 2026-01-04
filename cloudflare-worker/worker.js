/**
 * Cloudflare Worker for Portfolio Intelligence Dashboard
 * 
 * This worker serves the dashboard UI and provides secure API endpoints
 * for fetching data from a private GitHub repository and triggering workflows.
 * 
 * Environment Variables Required:
 * - GITHUB_PAT: GitHub Personal Access Token with repo and actions scope
 * - GITHUB_OWNER: Repository owner (e.g., "Valley-CortexCloud")
 * - GITHUB_REPO: Repository name (e.g., "ai-predications-app-final")
 * 
 * Optional Environment Variables:
 * - TOKEN_PATH: Path to token files (default: "data/portfolio/tokens")
 * - DATA_PATH: Path to dashboard data files (default: "docs/dashboard/data")
 */

// Configuration - can be overridden via environment variables
const DEFAULT_TOKEN_PATH = "data/portfolio/tokens";
const DEFAULT_DATA_PATH = "docs/dashboard/data";

// ============================================================================
// Dashboard HTML Template
// ============================================================================

const DASHBOARD_HTML = `<!DOCTYPE html>
<html lang="en" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Portfolio Intelligence Dashboard</title>
    
    <!-- TailwindCSS CDN -->
    <script src="https://cdn.tailwindcss.com"></script>
    
    <!-- Alpine.js CDN -->
    <script defer src="https://cdn.jsdelivr.net/npm/alpinejs@3.x.x/dist/cdn.min.js"></script>
    
    <script>
        // Tailwind dark mode configuration
        tailwind.config = {
            darkMode: 'class',
            theme: {
                extend: {
                    colors: {
                        dark: {
                            bg: '#0f172a',
                            card: '#1e293b',
                            border: '#334155'
                        }
                    }
                }
            }
        }
    </script>
    
    <style>
        [x-cloak] { display: none !important; }
        
        .glow-green {
            box-shadow: 0 0 10px rgba(34, 197, 94, 0.5);
        }
        
        .glow-red {
            box-shadow: 0 0 10px rgba(239, 68, 68, 0.5);
        }
        
        .exit-score-high {
            color: #ef4444;
            font-weight: 600;
        }
        
        .exit-score-medium {
            color: #f59e0b;
            font-weight: 600;
        }
        
        .exit-score-low {
            color: #22c55e;
            font-weight: 600;
        }
        
        .btn-toggle {
            transition: all 0.2s ease;
        }
        
        .btn-toggle.selected {
            transform: scale(1.05);
        }
    </style>
</head>
<body class="bg-dark-bg text-gray-100 min-h-screen">
    
    <div x-data="dashboardApp()" x-init="init()" x-cloak class="container mx-auto px-4 py-8 max-w-6xl">
        
        <!-- Header -->
        <div class="mb-8">
            <h1 class="text-4xl font-bold mb-2">🧠 Portfolio Intelligence</h1>
            <p class="text-gray-400 text-lg" x-text="'Rotation Review for ' + data.date"></p>
            
            <!-- Token Expiry Countdown -->
            <div class="mt-4 bg-yellow-900/30 border border-yellow-700 rounded-lg p-3" x-show="!tokenExpired && data.expires">
                <p class="text-yellow-300">
                    ⏰ Token expires in: <span class="font-mono font-bold" x-text="expiryCountdown"></span>
                </p>
            </div>
            
            <!-- Token Expired Warning -->
            <div class="mt-4 bg-red-900/30 border border-red-700 rounded-lg p-3" x-show="tokenExpired">
                <p class="text-red-300">
                    ❌ This token has expired. Please request a new portfolio review.
                </p>
            </div>
            
            <!-- Loading State -->
            <div class="mt-4 bg-blue-900/30 border border-blue-700 rounded-lg p-3" x-show="loading">
                <p class="text-blue-300">
                    ⏳ Loading portfolio data...
                </p>
            </div>
            
            <!-- Error State -->
            <div class="mt-4 bg-red-900/30 border border-red-700 rounded-lg p-3" x-show="error">
                <p class="text-red-300" x-text="error"></p>
            </div>
        </div>
        
        <!-- Summary Cards -->
        <div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8" x-show="!loading && !error">
            <div class="bg-dark-card border border-dark-border rounded-lg p-4">
                <div class="text-red-400 text-2xl font-bold" x-text="summary.total_sells + ' Sells'"></div>
                <div class="text-gray-400 text-sm">Positions to exit</div>
            </div>
            <div class="bg-dark-card border border-dark-border rounded-lg p-4">
                <div class="text-green-400 text-2xl font-bold" x-text="summary.total_buys + ' Buys'"></div>
                <div class="text-gray-400 text-sm">New opportunities</div>
            </div>
            <div class="bg-dark-card border border-dark-border rounded-lg p-4">
                <div class="text-blue-400 text-2xl font-bold" x-text="data.turnover_pct + '%'"></div>
                <div class="text-gray-400 text-sm">Portfolio turnover</div>
            </div>
        </div>
        
        <!-- Current Holdings Section -->
        <div class="mb-8" x-show="holdings.length > 0 && !loading && !error">
            <h2 class="text-2xl font-bold mb-4">📊 Current Holdings</h2>
            
            <div class="space-y-3">
                <template x-for="(holding, index) in holdings" :key="index">
                    <div class="bg-dark-card border border-dark-border rounded-lg p-4">
                        <div class="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
                            <!-- Left: Symbol and Info -->
                            <div class="flex-1">
                                <div class="flex items-center gap-3 mb-2">
                                    <h3 class="text-xl font-bold" x-text="holding.symbol"></h3>
                                    <span 
                                        class="text-sm font-mono"
                                        :class="{
                                            'exit-score-high': holding.exit_score > 70,
                                            'exit-score-medium': holding.exit_score >= 60 && holding.exit_score <= 70,
                                            'exit-score-low': holding.exit_score < 60
                                        }"
                                        x-text="'Score: ' + holding.exit_score"
                                    ></span>
                                    <span class="text-sm text-gray-400" x-text="holding.days_held + ' days'"></span>
                                </div>
                                
                                <p class="text-gray-300 text-sm mb-2" x-text="holding.reason"></p>
                                
                                <div class="flex gap-4 text-xs text-gray-400">
                                    <span x-text="'Entry: $' + holding.entry_price.toFixed(2)"></span>
                                    <span x-text="'Current: $' + holding.current_price.toFixed(2)"></span>
                                    <span 
                                        :class="holding.pnl_pct >= 0 ? 'text-green-400' : 'text-red-400'"
                                        x-text="(holding.pnl_pct >= 0 ? '+' : '') + holding.pnl_pct.toFixed(1) + '%'"
                                    ></span>
                                </div>
                            </div>
                            
                            <!-- Right: Action Buttons -->
                            <div class="flex gap-2">
                                <button 
                                    @click="toggleHoldingAction(index, 'SELL')"
                                    class="btn-toggle px-4 py-2 rounded-lg font-semibold transition-all"
                                    :class="holding.action === 'SELL' ? 'bg-red-600 text-white glow-red selected' : 'bg-gray-700 text-gray-300 hover:bg-gray-600'"
                                >
                                    SELL
                                </button>
                                <button 
                                    @click="toggleHoldingAction(index, 'HOLD')"
                                    class="btn-toggle px-4 py-2 rounded-lg font-semibold transition-all"
                                    :class="holding.action === 'HOLD' ? 'bg-blue-600 text-white glow-green selected' : 'bg-gray-700 text-gray-300 hover:bg-gray-600'"
                                >
                                    HOLD
                                </button>
                            </div>
                        </div>
                    </div>
                </template>
            </div>
        </div>
        
        <!-- New Recommendations Section -->
        <div class="mb-8" x-show="recommendations.length > 0 && !loading && !error">
            <h2 class="text-2xl font-bold mb-4">🚀 New Recommendations</h2>
            
            <div class="space-y-3">
                <template x-for="(rec, index) in recommendations" :key="index">
                    <div class="bg-dark-card border border-dark-border rounded-lg p-4">
                        <div class="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
                            <!-- Left: Symbol and Info -->
                            <div class="flex-1">
                                <div class="flex items-center gap-3 mb-2">
                                    <h3 class="text-xl font-bold" x-text="rec.symbol"></h3>
                                    <span class="text-sm px-2 py-1 bg-green-900/30 text-green-300 rounded" x-text="rec.conviction"></span>
                                    <span class="text-xs text-gray-400" x-text="'Rank: ' + rec.supercharged_rank" x-show="rec.supercharged_rank > 0"></span>
                                </div>
                                
                                <p class="text-gray-300 text-sm" x-text="rec.reason"></p>
                            </div>
                            
                            <!-- Right: Action Buttons -->
                            <div class="flex gap-2">
                                <button 
                                    @click="toggleRecommendationAction(index, 'BUY')"
                                    class="btn-toggle px-4 py-2 rounded-lg font-semibold transition-all"
                                    :class="rec.action === 'BUY' ? 'bg-green-600 text-white glow-green selected' : 'bg-gray-700 text-gray-300 hover:bg-gray-600'"
                                >
                                    BUY
                                </button>
                                <button 
                                    @click="toggleRecommendationAction(index, 'SKIP')"
                                    class="btn-toggle px-4 py-2 rounded-lg font-semibold transition-all"
                                    :class="rec.action === 'SKIP' ? 'bg-gray-600 text-white selected' : 'bg-gray-700 text-gray-300 hover:bg-gray-600'"
                                >
                                    SKIP
                                </button>
                            </div>
                        </div>
                    </div>
                </template>
            </div>
        </div>
        
        <!-- Action Buttons -->
        <div class="flex flex-col md:flex-row gap-4 mb-8" x-show="!loading && !error && !tokenExpired">
            <button 
                @click="confirmChanges()"
                :disabled="submitting"
                class="flex-1 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white font-bold py-4 px-6 rounded-lg text-lg transition-all"
            >
                <span x-show="!submitting">✅ CONFIRM & EXECUTE</span>
                <span x-show="submitting">⏳ Submitting...</span>
            </button>
            
            <button 
                @click="denyChanges()"
                :disabled="submitting"
                class="flex-1 bg-red-600 hover:bg-red-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white font-bold py-4 px-6 rounded-lg text-lg transition-all"
            >
                <span x-show="!submitting">❌ DENY ALL</span>
                <span x-show="submitting">⏳ Submitting...</span>
            </button>
        </div>
        
        <!-- Summary of Changes -->
        <div class="bg-dark-card border border-dark-border rounded-lg p-4 mb-8" x-show="!loading && !error">
            <h3 class="text-lg font-bold mb-2">📋 Summary of Your Changes</h3>
            <div class="text-sm text-gray-300 space-y-1">
                <p>
                    <span class="font-semibold text-red-400" x-text="getSelectedActions().sells.length"></span> positions to sell
                </p>
                <p>
                    <span class="font-semibold text-green-400" x-text="getSelectedActions().buys.length"></span> new positions to buy
                </p>
                <p>
                    <span class="font-semibold text-blue-400" x-text="getSelectedActions().holds.length"></span> positions to hold
                </p>
                <p>
                    <span class="font-semibold text-gray-400" x-text="getSelectedActions().skips.length"></span> recommendations to skip
                </p>
            </div>
        </div>
        
        <!-- Success Message -->
        <div class="bg-green-900/30 border border-green-700 rounded-lg p-4" x-show="successMessage">
            <p class="text-green-300" x-text="successMessage"></p>
        </div>
        
        <!-- Footer -->
        <div class="mt-12 text-center text-gray-500 text-sm">
            <p>Portfolio Intelligence Engine v1.0</p>
            <p>Renaissance-inspired exit detection with Grok AI</p>
        </div>
    </div>
    
    <script>
        function dashboardApp() {
            return {
                // State
                data: {
                    date: '',
                    token: '',
                    expires: '',
                    turnover_pct: 0,
                    holdings: [],
                    recommendations: [],
                    summary: {}
                },
                holdings: [],
                recommendations: [],
                summary: {},
                loading: true,
                error: '',
                tokenExpired: false,
                expiryCountdown: '',
                submitting: false,
                successMessage: '',
                countdownInterval: null,
                
                // Initialize
                async init() {
                    // Parse URL parameters
                    const params = new URLSearchParams(window.location.search);
                    const token = params.get('token');
                    const date = params.get('date');
                    
                    if (!date) {
                        this.error = '❌ No date parameter provided. URL should include ?date=YYYY-MM-DD';
                        this.loading = false;
                        return;
                    }
                    
                    if (!token) {
                        this.error = '❌ No security token provided. Please use the link from your email.';
                        this.loading = false;
                        return;
                    }
                    
                    this.data.date = date;
                    this.data.token = token;
                    
                    // Load data from worker API
                    await this.loadData(date, token);
                    
                    // Start expiry countdown
                    if (this.data.expires) {
                        this.startExpiryCountdown();
                    }
                },
                
                // Load portfolio data
                async loadData(date, token) {
                    try {
                        // Fetch from worker API endpoint
                        const dataUrl = \`/api/data/\${date}?token=\${encodeURIComponent(token)}\`;
                        
                        console.log(\`Loading data from: \${dataUrl}\`);
                        
                        const response = await fetch(dataUrl);
                        
                        if (!response.ok) {
                            const errorData = await response.json().catch(() => ({}));
                            throw new Error(errorData.error || \`Failed to load data: \${response.status} \${response.statusText}\`);
                        }
                        
                        const jsonData = await response.json();
                        
                        // Load data
                        this.data = jsonData;
                        this.holdings = jsonData.holdings.map(h => ({
                            ...h,
                            action: h.proposed_action // Initialize with proposed action
                        }));
                        this.recommendations = jsonData.recommendations.map(r => ({
                            ...r,
                            action: r.proposed_action // Initialize with proposed action
                        }));
                        this.summary = jsonData.summary || {};
                        
                        // Check if token is expired
                        if (this.data.expires) {
                            const expiresDate = new Date(this.data.expires);
                            if (expiresDate < new Date()) {
                                this.tokenExpired = true;
                            }
                        }
                        
                        this.loading = false;
                        
                        console.log('Data loaded successfully:', this.data);
                        
                    } catch (err) {
                        console.error('Error loading data:', err);
                        this.error = \`❌ Failed to load portfolio data: \${err.message}\`;
                        this.loading = false;
                    }
                },
                
                // Toggle holding action (SELL ↔ HOLD)
                toggleHoldingAction(index, action) {
                    this.holdings[index].action = action;
                },
                
                // Toggle recommendation action (BUY ↔ SKIP)
                toggleRecommendationAction(index, action) {
                    this.recommendations[index].action = action;
                },
                
                // Get selected actions summary
                getSelectedActions() {
                    const sells = this.holdings.filter(h => h.action === 'SELL').map(h => h.symbol);
                    const holds = this.holdings.filter(h => h.action === 'HOLD').map(h => h.symbol);
                    const buys = this.recommendations.filter(r => r.action === 'BUY').map(r => r.symbol);
                    const skips = this.recommendations.filter(r => r.action === 'SKIP').map(r => r.symbol);
                    
                    return { sells, holds, buys, skips };
                },
                
                // Confirm changes
                async confirmChanges() {
                    if (this.tokenExpired) {
                        alert('Token has expired. Please request a new portfolio review.');
                        return;
                    }
                    
                    const actions = this.getSelectedActions();
                    
                    // Confirm with user
                    const confirmMsg = \`Confirm execution:\\n\\n\` +
                        \`SELLS: \${actions.sells.length} (\${actions.sells.join(', ')})\` + '\\n' +
                        \`BUYS: \${actions.buys.length} (\${actions.buys.join(', ')})\` + '\\n' +
                        \`HOLDS: \${actions.holds.length}\` + '\\n' +
                        \`SKIPS: \${actions.skips.length}\` + '\\n\\n' +
                        \`This will trigger trade execution in your Alpaca paper account.\`;
                    
                    if (!confirm(confirmMsg)) {
                        return;
                    }
                    
                    this.submitting = true;
                    
                    try {
                        // Submit to worker API
                        await this.submitAction('CONFIRM', actions);
                        
                        this.successMessage = '✅ Confirmation submitted! Trades will be executed shortly. Check your email for confirmation.';
                        
                        // Disable further submissions
                        this.tokenExpired = true;
                        
                    } catch (err) {
                        console.error('Error submitting confirmation:', err);
                        alert(\`Failed to submit confirmation: \${err.message}\\n\\nPlease try again or contact support.\`);
                    } finally {
                        this.submitting = false;
                    }
                },
                
                // Deny changes
                async denyChanges() {
                    if (this.tokenExpired) {
                        alert('Token has expired.');
                        return;
                    }
                    
                    if (!confirm('Deny all proposed changes? No trades will be executed.')) {
                        return;
                    }
                    
                    this.submitting = true;
                    
                    try {
                        await this.submitAction('DENY', {
                            sells: [],
                            holds: this.holdings.map(h => h.symbol),
                            buys: [],
                            skips: this.recommendations.map(r => r.symbol)
                        });
                        
                        this.successMessage = '✅ All changes denied. No trades will be executed.';
                        
                        // Disable further submissions
                        this.tokenExpired = true;
                        
                    } catch (err) {
                        console.error('Error submitting denial:', err);
                        alert(\`Failed to submit denial: \${err.message}\\n\\nPlease try again or contact support.\`);
                    } finally {
                        this.submitting = false;
                    }
                },
                
                // Submit action to worker API
                async submitAction(action, selections) {
                    const payload = {
                        token: this.data.token,
                        date: this.data.date,
                        action: action,
                        ...selections
                    };
                    
                    console.log('Submitting to API:', payload);
                    
                    const response = await fetch('/api/confirm', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify(payload)
                    });
                    
                    if (!response.ok) {
                        const errorData = await response.json().catch(() => ({}));
                        throw new Error(errorData.error || \`Request failed: \${response.status}\`);
                    }
                    
                    const result = await response.json();
                    console.log('Submission successful:', result);
                    
                    return result;
                },
                
                // Start expiry countdown
                startExpiryCountdown() {
                    if (this.countdownInterval) {
                        clearInterval(this.countdownInterval);
                    }
                    
                    this.countdownInterval = setInterval(() => {
                        const now = new Date();
                        const expires = new Date(this.data.expires);
                        const diff = expires - now;
                        
                        if (diff <= 0) {
                            this.expiryCountdown = 'EXPIRED';
                            this.tokenExpired = true;
                            clearInterval(this.countdownInterval);
                            return;
                        }
                        
                        const hours = Math.floor(diff / (1000 * 60 * 60));
                        const minutes = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60));
                        const seconds = Math.floor((diff % (1000 * 60)) / 1000);
                        
                        this.expiryCountdown = \`\${hours}h \${minutes}m \${seconds}s\`;
                    }, 1000);
                }
            }
        }
    </script>
</body>
</html>`;

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Set CORS headers for browser requests
 */
function setCorsHeaders(response) {
  response.headers.set('Access-Control-Allow-Origin', '*');
  response.headers.set('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  response.headers.set('Access-Control-Allow-Headers', 'Content-Type');
  return response;
}

/**
 * Handle OPTIONS request for CORS preflight
 */
function handleOptions() {
  return setCorsHeaders(new Response(null, {
    status: 204
  }));
}

/**
 * Create JSON response
 */
function jsonResponse(data, status = 200) {
  return setCorsHeaders(new Response(JSON.stringify(data), {
    status: status,
    headers: {
      'Content-Type': 'application/json'
    }
  }));
}

/**
 * Create error response
 */
function errorResponse(message, status = 400) {
  return jsonResponse({ error: message, success: false }, status);
}

/**
 * Validate security token
 */
async function validateToken(env, date, token) {
  try {
    // Get token path from environment or use default
    const tokenPath = env.TOKEN_PATH || DEFAULT_TOKEN_PATH;
    
    // Fetch token data from GitHub
    const tokenUrl = `https://raw.githubusercontent.com/${env.GITHUB_OWNER}/${env.GITHUB_REPO}/main/${tokenPath}/${date}.json`;
    
    const response = await fetch(tokenUrl, {
      headers: {
        'Authorization': `token ${env.GITHUB_PAT}`,
        'Accept': 'application/vnd.github.v3.raw'
      }
    });
    
    if (!response.ok) {
      return { valid: false, error: 'Token file not found' };
    }
    
    const tokenData = await response.json();
    
    // Check token matches
    if (tokenData.token !== token) {
      return { valid: false, error: 'Invalid token' };
    }
    
    // Check not expired
    const expiresDate = new Date(tokenData.expires);
    if (expiresDate < new Date()) {
      return { valid: false, error: 'Token expired' };
    }
    
    // Check not already used
    if (tokenData.used === true) {
      return { valid: false, error: 'Token already used' };
    }
    
    return { valid: true, tokenData };
    
  } catch (error) {
    console.error('Token validation error:', error);
    return { valid: false, error: 'Token validation failed' };
  }
}

/**
 * Fetch data from private GitHub repository
 */
async function fetchFromGitHub(env, path) {
  const url = `https://raw.githubusercontent.com/${env.GITHUB_OWNER}/${env.GITHUB_REPO}/main/${path}`;
  
  const response = await fetch(url, {
    headers: {
      'Authorization': `token ${env.GITHUB_PAT}`,
      'Accept': 'application/vnd.github.v3.raw'
    }
  });
  
  if (!response.ok) {
    throw new Error(`GitHub fetch failed: ${response.status} ${response.statusText}`);
  }
  
  return response;
}

/**
 * Trigger GitHub workflow via repository_dispatch
 */
async function triggerGitHubWorkflow(env, eventType, clientPayload) {
  const url = `https://api.github.com/repos/${env.GITHUB_OWNER}/${env.GITHUB_REPO}/dispatches`;
  
  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Authorization': `token ${env.GITHUB_PAT}`,
      'Accept': 'application/vnd.github.v3+json',
      'Content-Type': 'application/json',
      'User-Agent': 'Cloudflare-Worker-Portfolio-Dashboard'
    },
    body: JSON.stringify({
      event_type: eventType,
      client_payload: clientPayload
    })
  });
  
  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`GitHub workflow trigger failed: ${response.status} - ${errorText}`);
  }
  
  return true;
}

// ============================================================================
// Route Handlers
// ============================================================================

/**
 * Serve dashboard HTML
 */
async function handleDashboard() {
  return new Response(DASHBOARD_HTML, {
    headers: {
      'Content-Type': 'text/html;charset=UTF-8',
      'Cache-Control': 'no-cache'
    }
  });
}

/**
 * Health check endpoint
 */
async function handleHealth() {
  return jsonResponse({
    status: 'healthy',
    timestamp: new Date().toISOString(),
    version: '1.0.0'
  });
}

/**
 * Fetch portfolio data for a specific date
 */
async function handleGetData(env, request, date) {
  try {
    // Extract token from query parameter
    const url = new URL(request.url);
    const token = url.searchParams.get('token');
    
    if (!token) {
      return errorResponse('Token parameter required', 400);
    }
    
    // Validate token
    const validation = await validateToken(env, date, token);
    if (!validation.valid) {
      return errorResponse(validation.error, 403);
    }
    
    // Get data path from environment or use default
    const dataPath = env.DATA_PATH || DEFAULT_DATA_PATH;
    
    // Fetch dashboard data from GitHub
    const fullDataPath = `${dataPath}/${date}.json`;
    const response = await fetchFromGitHub(env, fullDataPath);
    const data = await response.json();
    
    // Return data
    return jsonResponse(data);
    
  } catch (error) {
    console.error('Error fetching data:', error);
    return errorResponse(`Failed to fetch data: ${error.message}`, 500);
  }
}

/**
 * Handle confirmation/denial submission
 */
async function handleConfirm(env, request) {
  try {
    // Parse request body
    const body = await request.json();
    const { token, date, action, sells, buys, holds, skips } = body;
    
    // Validate required fields
    if (!token || !date || !action) {
      return errorResponse('Missing required fields: token, date, action', 400);
    }
    
    if (action !== 'CONFIRM' && action !== 'DENY') {
      return errorResponse('Action must be CONFIRM or DENY', 400);
    }
    
    // Validate token
    const validation = await validateToken(env, date, token);
    if (!validation.valid) {
      return errorResponse(validation.error, 403);
    }
    
    // Prepare payload for GitHub workflow
    const clientPayload = {
      token,
      date,
      action,
      sells: Array.isArray(sells) ? sells : [],
      buys: Array.isArray(buys) ? buys : [],
      holds: Array.isArray(holds) ? holds : [],
      skips: Array.isArray(skips) ? skips : []
    };
    
    // Trigger GitHub workflow
    await triggerGitHubWorkflow(env, 'dashboard_confirmation', clientPayload);
    
    return jsonResponse({
      success: true,
      message: 'Workflow triggered successfully',
      action: action
    });
    
  } catch (error) {
    console.error('Error processing confirmation:', error);
    return errorResponse(`Failed to process confirmation: ${error.message}`, 500);
  }
}

// ============================================================================
// Main Request Handler
// ============================================================================

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);
    const path = url.pathname;
    const method = request.method;
    
    // Handle CORS preflight
    if (method === 'OPTIONS') {
      return handleOptions();
    }
    
    try {
      // Route: GET / or /dashboard -> Serve dashboard HTML
      if ((path === '/' || path === '/dashboard') && method === 'GET') {
        return handleDashboard();
      }
      
      // Route: GET /api/health -> Health check
      if (path === '/api/health' && method === 'GET') {
        return handleHealth();
      }
      
      // Route: GET /api/data/:date -> Fetch portfolio data
      const dataMatch = path.match(/^\/api\/data\/([0-9]{4}-[0-9]{2}-[0-9]{2})$/);
      if (dataMatch && method === 'GET') {
        const date = dataMatch[1];
        return handleGetData(env, request, date);
      }
      
      // Route: POST /api/confirm -> Handle confirmation
      if (path === '/api/confirm' && method === 'POST') {
        return handleConfirm(env, request);
      }
      
      // 404 for unknown routes
      return errorResponse('Not found', 404);
      
    } catch (error) {
      console.error('Worker error:', error);
      return errorResponse(`Internal server error: ${error.message}`, 500);
    }
  }
};
