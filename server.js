import "dotenv/config";
import express from "express";
import { OpenAI, toFile } from "openai";
import path from "path";
import fs from "fs";
import os from "os";
import { fileURLToPath } from "url";
import { EdgeTTS } from "node-edge-tts";
import { initBridge, recordExchange, recordLead, cleanupSessions } from "./agentvox-bridge.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const app = express();
const PORT = process.env.PORT || 8300;

// Config from environment
const GROQ_API_KEY = process.env.GROQ_API_KEY;
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
// Department-based auth codes — each can have a per-customer gateway key
const KAAS_API_URL = process.env.KAAS_API_URL || "https://meridian.covault.xyz";
const KAAS_API_KEY = process.env.KAAS_API_KEY || "";
const AUTH_CODES = {
  sales2026: { dept: "sales", name: "Sales" },
  accounting2026: { dept: "accounting", name: "Accounting" },
  leadership2026: { dept: "leadership", name: "Leadership" },
  [process.env.AUTH_CODE || "luvbuds2026"]: { dept: "all", name: "General" },
  // Must be AFTER computed AUTH_CODE property — if AUTH_CODE=meridian, this overrides it
  meridian: { dept: "all", name: "Oracle", workspace: "meridian", gatewayKey: KAAS_API_KEY },
};
const AUTH_CODE = process.env.AUTH_CODE || "luvbuds2026"; // legacy compat
const GATEWAY_URL = process.env.GATEWAY_URL || "";
const GATEWAY_API_KEY = process.env.GATEWAY_API_KEY || "";

// Per-session gateway keys (for per-customer rate limiting on the gateway)
const sessionGatewayKeys = new Map(); // sessionId → gatewayKey
const KPI_DASHBOARD_URL = process.env.KPI_DASHBOARD_URL || "https://kpi-dashboard-8fj4.onrender.com";
const KPI_DASHBOARD_AUTH = process.env.KPI_DASHBOARD_AUTH || "";  // "admin:password" format

// ============== Meridian House Prompt (workspace-aware) ==============
let MERIDIAN_HOUSE_PROMPT = "";
try {
  const housePath = path.resolve(__dirname, "config/agentvox/meridian-house-voice.md");
  if (fs.existsSync(housePath)) {
    MERIDIAN_HOUSE_PROMPT = fs.readFileSync(housePath, "utf-8");
    console.log(`[meridian-voice] Loaded Meridian house prompt (${MERIDIAN_HOUSE_PROMPT.length} chars)`);
  }
} catch (e) {
  console.warn(`[meridian-voice] Failed to load Meridian house prompt: ${e.message}`);
}
const MERIDIAN_WORKSPACES = new Set(["meridian", "meridian-house", "meridian-demo"]);

function isMeridianWorkspace(workspaceId) {
  return MERIDIAN_HOUSE_PROMPT && MERIDIAN_WORKSPACES.has(workspaceId);
}

function getMeridianPrompt() {
  return `CRITICAL VOICE RULES — you are on a LIVE VOICE CALL, not writing text:
- Keep responses to 1-3 sentences MAX. You are speaking, not writing an essay.
- NEVER use markdown formatting (no **, no ##, no bullet lists).
- Speak naturally. Use contractions. Sound like a real human on a phone call.
- You are Oracle, the Meridian AI voice agent. You ARE the product demo.

${MERIDIAN_HOUSE_PROMPT}`;
}

// ============== Meridian MIG Context Retrieval (via KaaS relay) ==============
// KAAS_API_URL and KAAS_API_KEY defined at top of file (line 18-19)

// Domain knowledge cache: refreshed every 10 minutes
let migDomainCache = null;
let migDomainCacheTime = 0;
const MIG_CACHE_TTL = 10 * 60 * 1000; // 10 minutes

async function fetchMeridianDomainKnowledge(domain = "business_operations") {
  if (migDomainCache && Date.now() - migDomainCacheTime < MIG_CACHE_TTL) return migDomainCache;
  if (!KAAS_API_URL) return null;
  try {
    const headers = {};
    if (KAAS_API_KEY) headers["X-API-Key"] = KAAS_API_KEY;
    const resp = await fetch(
      `${KAAS_API_URL}/v1/cortex/${encodeURIComponent(domain)}?importance_min=6&limit=15`,
      { headers, signal: AbortSignal.timeout(5000) }
    );
    if (!resp.ok) {
      console.warn(`[meridian-voice] MIG cortex fetch failed: ${resp.status}`);
      return null;
    }
    const data = await resp.json();
    migDomainCache = data;
    migDomainCacheTime = Date.now();
    return data;
  } catch (e) {
    console.warn(`[meridian-voice] MIG domain knowledge fetch failed: ${e.message}`);
    return null;
  }
}

async function searchMeridianKnowledge(query, limit = 8) {
  if (!KAAS_API_URL || !query) return [];
  try {
    const headers = {};
    if (KAAS_API_KEY) headers["X-API-Key"] = KAAS_API_KEY;
    const q = encodeURIComponent(query.slice(0, 200));
    const resp = await fetch(
      `${KAAS_API_URL}/v1/search?q=${q}&limit=${limit}`,
      { headers, signal: AbortSignal.timeout(5000) }
    );
    if (!resp.ok) return [];
    return await resp.json();
  } catch (e) {
    console.warn(`[meridian-voice] MIG search failed: ${e.message}`);
    return [];
  }
}

async function queryMeridianMIG(userMessage) {
  if (!KAAS_API_URL) return "";

  try {
    // Run domain knowledge fetch (cached) + live search in parallel
    const [domainData, searchResults] = await Promise.all([
      fetchMeridianDomainKnowledge("business_operations"),
      searchMeridianKnowledge(userMessage, 8),
    ]);

    const parts = [];

    // Format domain knowledge (high-importance memories about the business)
    if (domainData && domainData.findings && domainData.findings.length > 0) {
      const findings = domainData.findings
        .slice(0, 10)
        .map((f) => `- ${f.content.slice(0, 250)}`)
        .join("\n");
      parts.push(`MERIDIAN KNOWLEDGE BASE:\n${findings}`);
    }

    // Format search results (research documents matching the question)
    if (Array.isArray(searchResults) && searchResults.length > 0) {
      const docs = searchResults
        .slice(0, 6)
        .map((r) => {
          let line = r.title || "untitled";
          if (r.excerpt && r.excerpt.length > 5) line += ` — ${r.excerpt.slice(0, 200)}`;
          return `- ${line}`;
        })
        .join("\n");
      parts.push(`RELEVANT RESEARCH:\n${docs}`);
    }

    if (parts.length === 0) return "";

    const context = parts.join("\n\n");
    console.log(
      `[meridian-voice] MIG context: ${domainData?.findings?.length || 0} domain findings, ${searchResults?.length || 0} search results for "${userMessage.slice(0, 40)}..."`
    );
    return `\n${context}\n`;
  } catch (e) {
    console.warn(`[meridian-voice] MIG context query failed: ${e.message}`);
    return "";
  }
}

// ============== Meridian MIG Persistence (fire-and-forget) ==============

async function persistToMIG(workspaceId, sessionId, userMessage, aiResponse) {
  if (!isMeridianWorkspace(workspaceId)) return;
  if (!KAAS_API_URL || !KAAS_API_KEY) return;
  const url = `${KAAS_API_URL}/v1/intelligence/remember`;
  // Fire and forget — don't block the voice response
  fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'X-API-Key': KAAS_API_KEY },
    body: JSON.stringify({
      content: `Voice exchange — User: "${userMessage}" | Oracle: "${aiResponse.substring(0, 200)}"`,
      category: 'observation',
      importance: 5,
      workspace_id: workspaceId,
      source: 'voice-session',
      session_id: sessionId,
    }),
    signal: AbortSignal.timeout(5000),
  }).catch(e => console.warn('[meridian-voice] MIG persist failed (non-blocking):', e.message));
}

async function generateSessionSummary(workspaceId, sessionId, history) {
  if (!isMeridianWorkspace(workspaceId)) return;
  if (!KAAS_API_URL || !KAAS_API_KEY) return;
  if (!history || history.length < 4) return; // need at least 2 full exchanges

  try {
    // Build transcript from history
    const transcript = history.map(h => `${h.role === 'user' ? 'User' : 'Oracle'}: ${h.content}`).join('\n');
    const summaryPrompt = `Summarize this voice conversation in 2-3 sentences. Extract key entities (names, companies, metrics, products mentioned). Return JSON only, no markdown: {"summary": "...", "entities": [{"name": "...", "type": "..."}]}

Conversation:
${transcript}`;

    // Use Groq to generate summary
    const completion = await groq.chat.completions.create({
      model: CHAT_MODEL,
      messages: [{ role: 'user', content: summaryPrompt }],
      max_tokens: 300,
      temperature: 0.3,
    });

    const raw = completion.choices[0]?.message?.content || '';
    let summary, entities;
    try {
      // Strip markdown code fences if present
      const cleaned = raw.replace(/```json\s*/g, '').replace(/```\s*/g, '').trim();
      const parsed = JSON.parse(cleaned);
      summary = parsed.summary || raw;
      entities = Array.isArray(parsed.entities) ? parsed.entities : [];
    } catch {
      // LLM didn't return valid JSON — use raw text as summary
      summary = raw.substring(0, 300);
      entities = [];
    }

    // POST session summary to KaaS
    const url = `${KAAS_API_URL}/v1/intelligence/session-summary`;
    const resp = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-API-Key': KAAS_API_KEY },
      body: JSON.stringify({
        summary,
        entities,
        workspace_id: workspaceId,
        session_id: sessionId,
        exchange_count: Math.floor(history.length / 2),
        source: 'voice-session',
      }),
      signal: AbortSignal.timeout(10000),
    });

    if (resp.ok) {
      console.log(`[meridian-voice] Session summary persisted for ${sessionId} (${Math.floor(history.length / 2)} exchanges, ${entities.length} entities)`);
    } else {
      console.warn(`[meridian-voice] Session summary API returned ${resp.status} for ${sessionId}`);
    }
  } catch (e) {
    console.warn(`[meridian-voice] Session summary failed (non-blocking): ${e.message}`);
  }
}

// ============== Customer Knowledge Retrieval (RAG) ==============

// Stats cache: fetched once per 10 minutes, gives the LLM awareness of available data
let statsCache = null;
let statsCacheTime = 0;
const STATS_TTL = 10 * 60 * 1000; // 10 minutes

async function fetchStats(apiKey = "") {
  if (statsCache && Date.now() - statsCacheTime < STATS_TTL) return statsCache;
  const key = apiKey || GATEWAY_API_KEY;
  if (!GATEWAY_URL || !key) return null;
  try {
    const resp = await fetch(`${GATEWAY_URL}/v1/customer/stats`, {
      headers: { "X-API-Key": key },
      signal: AbortSignal.timeout(5000),
    });
    if (!resp.ok) return null;
    statsCache = await resp.json();
    statsCacheTime = Date.now();
    return statsCache;
  } catch (e) {
    console.warn(`[meridian-voice] Stats fetch failed: ${e.message}`);
    return null;
  }
}

function formatStatsContext(stats) {
  if (!stats) return "";
  // Accept either {categories: [...]} or flat array format
  const cats = Array.isArray(stats) ? stats : stats.categories || stats.breakdown || [];
  if (!cats.length) return "";
  const lines = cats
    .sort((a, b) => (b.count || 0) - (a.count || 0))
    .map((c) => `  ${(c.category || c.type || c.name || "unknown").replace(/_/g, " ")}: ${c.count?.toLocaleString() || "?"} records`)
    .join("\n");
  return `\nDATA AVAILABLE IN SYSTEM:\n${lines}\n`;
}

// Detect analytical questions that need multi-category or aggregation context
function isAnalyticalQuery(message) {
  const patterns = /\b(top|best|most|least|how many|total|sum|average|count|trend|compare|revenue|sales volume|biggest|highest|lowest|ranking|rank)\b/i;
  return patterns.test(message);
}

// Department-relevant categories for targeted sub-searches
const DEPT_CATEGORIES = {
  sales: ["products", "dynamics_sales_orders", "brands", "dynamics_vendors"],
  accounting: ["dynamics_purchase_orders", "dynamics_po_lines", "dynamics_vendors", "dynamics_accounts", "dynamics_sales_orders"],
  leadership: ["dynamics_sales_orders", "dynamics_purchase_orders", "products", "decisions", "tasks"],
  all: ["products", "dynamics_sales_orders", "dynamics_purchase_orders", "dynamics_vendors"],
};

async function searchGateway(query, limit = 10, category = "", apiKey = "") {
  const q = encodeURIComponent(query.slice(0, 200));
  const catParam = category ? `&category=${category}` : "";
  const key = apiKey || GATEWAY_API_KEY;
  const url = `${GATEWAY_URL}/v1/customer/search?q=${q}&limit=${limit}${catParam}`;
  const resp = await fetch(url, {
    headers: { "X-API-Key": key },
    signal: AbortSignal.timeout(8000),
  });
  if (!resp.ok) return [];
  return resp.json();
}

async function queryCustomerKnowledge(userMessage, department, apiKey = "") {
  const key = apiKey || GATEWAY_API_KEY;
  if (!GATEWAY_URL || !key) return "";
  try {
    // Prefetch stats (cached, non-blocking after first call)
    const statsPromise = fetchStats(key);

    const analytical = isAnalyticalQuery(userMessage);
    const searches = [];

    // Primary search: full query, no category filter, broad results
    searches.push(searchGateway(userMessage, analytical ? 15 : 10, "", key));

    // For analytical questions: add targeted category sub-searches
    if (analytical) {
      const deptCats = DEPT_CATEGORIES[department] || DEPT_CATEGORIES.all;
      // Extract key terms for focused sub-searches
      const keyTerms = userMessage
        .replace(/\b(what|are|our|the|how|many|top|best|most|show|me|do|we|have|is|can|you|tell|about|give|list)\b/gi, "")
        .trim();
      if (keyTerms.length > 2) {
        // Search top 2 most relevant categories with focused terms
        for (const cat of deptCats.slice(0, 2)) {
          searches.push(searchGateway(keyTerms, 5, cat, key));
        }
      }
    }

    // Execute all searches in parallel
    const searchResults = await Promise.all(searches);

    // Deduplicate by title
    const seen = new Set();
    const allResults = [];
    for (const results of searchResults) {
      if (!Array.isArray(results)) continue;
      for (const r of results) {
        const key = `${r.category}:${r.title}`;
        if (!seen.has(key)) {
          seen.add(key);
          allResults.push(r);
        }
      }
    }

    if (!allResults.length) return "";

    // Format results with category labels
    const context = allResults
      .slice(0, 20) // cap at 20 to avoid prompt bloat
      .map((r) => {
        let line = `[${(r.category || "unknown").replace(/_/g, " ")}] ${r.title}`;
        if (r.excerpt && r.excerpt !== r.title && r.excerpt.length > 5) {
          line += ` — ${r.excerpt.slice(0, 200)}`;
        }
        return line;
      })
      .join("\n");

    // Build stats context
    const stats = await statsPromise;
    const statsContext = formatStatsContext(stats);

    console.log(`[meridian-voice] RAG: ${allResults.length} results (${searches.length} queries, analytical: ${analytical}) for "${userMessage.slice(0, 40)}..." (dept: ${department || "all"})`);
    return `${statsContext}\nRELEVANT CUSTOMER DATA (from LuvBuds knowledge base — answer from this data):\n${context}`;
  } catch (e) {
    console.warn(`[meridian-voice] RAG query failed: ${e.message}`);
    return "";
  }
}

// ============== KPI Dashboard Data Retrieval ==============
async function queryKPIDashboard(userMessage) {
  if (!KPI_DASHBOARD_URL || !KPI_DASHBOARD_AUTH) return "";
  const msg = userMessage.toLowerCase();

  // Route to the most relevant KPI endpoint based on the question
  const queries = [];
  if (msg.match(/sales|revenue|invoic|month|total|mtd|ytd|quarter/)) {
    queries.push({ url: "/api/cache/monthly-totals", label: "Monthly Sales Totals" });
  }
  if (msg.match(/customer|account|top.*client|biggest.*buyer|parent.*company/)) {
    queries.push({ url: "/api/cache/parent-companies?limit=10", label: "Top Customers" });
  }
  if (msg.match(/compare|month.*over|growth|trend|change|vs|versus/)) {
    queries.push({ url: "/api/cache/month-over-month", label: "Month-over-Month Comparison" });
  }
  if (msg.match(/margin|profit|cost|cogs|gross/)) {
    queries.push({ url: "/api/cache/margin-analysis", label: "Margin Analysis" });
  }
  if (msg.match(/inventory|stock|on.hand|velocity|reorder/)) {
    queries.push({ url: "/api/cache/inventory-velocity?leadTime=14", label: "Inventory Velocity" });
  }
  if (msg.match(/scorecard|pva|goal|target|performance/)) {
    queries.push({ url: "/api/cache/pva-scorecard", label: "PVA Scorecard" });
  }
  if (msg.match(/open.*order|backorder|pending|shipment/)) {
    queries.push({ url: "/api/cache/open-orders", label: "Open Orders" });
  }

  if (queries.length === 0) return ""; // No KPI-relevant question

  const authHeader = "Basic " + Buffer.from(KPI_DASHBOARD_AUTH).toString("base64");
  const results = [];

  for (const q of queries.slice(0, 2)) { // Max 2 KPI queries per turn
    try {
      const resp = await fetch(`${KPI_DASHBOARD_URL}${q.url}`, {
        headers: { Authorization: authHeader },
        signal: AbortSignal.timeout(8000),
      });
      if (!resp.ok) continue;
      const data = await resp.json();
      if (!data.success && !data.data) continue;

      // Summarize the data for the LLM (don't dump raw JSON)
      const payload = data.data || data;
      let summary = `[${q.label}]\n`;

      if (Array.isArray(payload)) {
        // Take recent entries (last 3 months for monthly data)
        const recent = payload.slice(-3);
        summary += recent.map((r) => {
          if (r.month && r.totalSales) {
            return `${r.month}: $${(r.totalSales/1000).toFixed(0)}K sales, ${(r.totalUnits||0).toLocaleString()} units, ${r.skuCount||'?'} SKUs`;
          }
          if (r.parentName && r.totalRevenue) {
            return `${r.parentName.split('|')[0].trim()}: $${(r.totalRevenue/1000).toFixed(0)}K revenue, ${r.totalOrders} orders`;
          }
          return JSON.stringify(r).slice(0, 150);
        }).join("\n");
      } else if (typeof payload === "object") {
        summary += JSON.stringify(payload).slice(0, 400);
      }

      results.push(summary);
      console.log(`[meridian-voice] KPI: ${q.label} returned data`);
    } catch (e) {
      console.warn(`[meridian-voice] KPI query failed (${q.url}): ${e.message}`);
    }
  }

  if (results.length === 0) return "";
  return `\n\nLIVE KPI DATA (from Dynamics 365 ETL — real numbers, updated daily):\n${results.join("\n\n")}`;
}

if (!GROQ_API_KEY) {
  console.error("[meridian-voice] GROQ_API_KEY is required");
  process.exit(1);
}

// Groq client (for STT + Chat)
const groq = new OpenAI({
  apiKey: GROQ_API_KEY,
  baseURL: "https://api.groq.com/openai/v1",
});
const CHAT_MODEL = "llama-3.3-70b-versatile"; // streams fast enough for TTS queue timing

// OpenAI client (for TTS) — may be out of quota
let openai = null;
if (OPENAI_API_KEY) {
  openai = new OpenAI({ apiKey: OPENAI_API_KEY });
} else {
  console.warn("[meridian-voice] OPENAI_API_KEY not set — TTS disabled, text-only mode");
}

// Middleware
app.use((req, res, next) => {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Workspace-Id");
  res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  if (req.method === "OPTIONS") return res.sendStatus(204);
  next();
});
app.use(express.static(path.join(__dirname, "public")));
app.use(express.json({ limit: "10mb" }));

// ============== Response Cache ==============
const responseCache = new Map();
const CACHE_MAX_SIZE = 100;

function getCacheKey(text, voice) {
  return `${voice}:${text.toLowerCase().trim()}`;
}

function addToCache(text, voice, audioBase64) {
  const key = getCacheKey(text, voice);
  if (responseCache.size >= CACHE_MAX_SIZE) {
    const firstKey = responseCache.keys().next().value;
    responseCache.delete(firstKey);
  }
  responseCache.set(key, { audio: audioBase64, timestamp: Date.now() });
}

function getFromCache(text, voice) {
  const key = getCacheKey(text, voice);
  return responseCache.get(key)?.audio;
}

// ============== Lead Storage ==============
const LEADS_FILE = path.join(__dirname, "leads.json");

function saveLead(lead) {
  const entry = {
    ...lead,
    timestamp: new Date().toISOString(),
    id: crypto.randomUUID(),
  };
  fs.appendFileSync(LEADS_FILE, JSON.stringify(entry) + "\n", "utf-8");
  console.log(`[meridian-voice] Lead captured: ${entry.name || "anonymous"} (${entry.email || "no email"})`);
  return entry;
}

// ============== Health Check ==============
app.get("/health", (req, res) => {
  res.json({ status: "ok", service: "meridian-voice", timestamp: Date.now() });
});

// ============== Analytics API Proxy ==============
// Proxies /analytics/* to the voice analytics API on :8100
// so the dashboard can reach it through the same tunnel
app.use("/analytics", async (req, res) => {
  try {
    const url = `http://127.0.0.1:8100/api/v1/analytics${req.url}`;
    const resp = await fetch(url, {
      method: req.method,
      headers: {
        "Content-Type": "application/json",
        "X-Workspace-Id": req.headers["x-workspace-id"] || "d6574870-0529-4015-aeeb-a25c202d7436",
      },
      signal: AbortSignal.timeout(10000),
    });
    const data = await resp.text();
    res.setHeader("Content-Type", "application/json");
    res.status(resp.status).send(data);
  } catch (e) {
    res.status(502).json({ error: "Analytics API unreachable", detail: e.message });
  }
});

// ============== Buyer Journey Stage Detection ==============
const journeyStages = new Map();

function detectJourneyStage(history) {
  const exchangeCount = Math.floor(history.length / 2);
  const allText = history.map(m => m.content.toLowerCase()).join(' ');
  let stage = 'lead';

  const problemSignals = ['challenge', 'headache', 'problem', 'struggling', 'pain', 'frustrating', 'nightmare', 'issue', 'difficult'];
  if (problemSignals.some(s => allText.includes(s)) && exchangeCount >= 2) stage = 'qualified';

  const solutionSignals = ['how do you', 'what can you', 'how would', 'pricing', 'cost', 'how much', 'tried', 'looked into'];
  if (solutionSignals.some(s => allText.includes(s)) && exchangeCount >= 3) stage = 'qualified';

  const contactSignals = ['my email', 'my phone', 'my number', 'reach me at', '@'];
  if (contactSignals.some(s => allText.includes(s))) stage = 'booked';

  return stage;
}

// ============== Lead Extraction ==============
const extractedSessions = new Set();

function extractLeadInfo(text) {
  const email = text.match(/[\w.-]+@[\w.-]+\.\w{2,}/)?.[0] || null;
  const phone = text.match(/\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}/)?.[0] || null;
  const nameMatch = text.match(/(?:my name is|i'm|i am|this is|call me)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)/i);
  const name = nameMatch?.[1] || null;
  const companyMatch = text.match(/(?:work at|work for|company is|company called|from)\s+([A-Z][A-Za-z\s&]+?)(?:\.|,|$)/i);
  const company = companyMatch?.[1]?.trim() || null;
  return { name, email, phone, company };
}

// ============== Session Persistence ==============
function loadSessionFromTranscript(workspaceId, sessionId) {
  // Try workspace-scoped first, then legacy flat
  const wsFile = path.join(TRANSCRIPTS_DIR, workspaceId || "demo", `${sessionId}.jsonl`);
  const legacyFile = path.join(TRANSCRIPTS_DIR, `${sessionId}.jsonl`);
  const logFile = fs.existsSync(wsFile) ? wsFile : legacyFile;
  try {
    if (!fs.existsSync(logFile)) return [];
    const lines = fs.readFileSync(logFile, "utf-8").trim().split("\n").filter(Boolean);
    const history = [];
    for (const line of lines) {
      const entry = JSON.parse(line);
      history.push({ role: "user", content: entry.user });
      history.push({ role: "assistant", content: entry.oracle });
    }
    while (history.length > 20) history.shift();
    if (history.length > 0) console.log(`[meridian-voice] Restored ${history.length / 2} exchanges for session ${sessionId}`);
    return history;
  } catch { return []; }
}

// ============== Conversation Transcript Logger ==============
const TRANSCRIPTS_DIR = path.join(__dirname, "transcripts");
if (!fs.existsSync(TRANSCRIPTS_DIR)) fs.mkdirSync(TRANSCRIPTS_DIR, { recursive: true });

function logTranscript(workspaceId, sessionId, userMessage, aiResponse, timing) {
  // Workspace-scoped transcript directory
  const wsDir = path.join(TRANSCRIPTS_DIR, workspaceId || "demo");
  if (!fs.existsSync(wsDir)) fs.mkdirSync(wsDir, { recursive: true });
  const logFile = path.join(wsDir, `${sessionId}.jsonl`);
  const entry = { timestamp: new Date().toISOString(), user: userMessage, oracle: aiResponse, timing };
  fs.appendFileSync(logFile, JSON.stringify(entry) + "\n", "utf-8");

  // AgentVox DB bridge (workspace-scoped)
  recordExchange(workspaceId, sessionId, userMessage, aiResponse, timing);

  // Auto lead extraction
  if (!extractedSessions.has(sessionId)) {
    const lead = extractLeadInfo(userMessage);
    if (lead.email || lead.phone || lead.name) {
      saveLead({ ...lead, notes: `Auto-extracted from session ${sessionId}` });
      recordLead(workspaceId, sessionId, lead);
      extractedSessions.add(sessionId);
      console.log(`[meridian-voice] Auto-extracted lead: ${lead.name || lead.email || lead.phone}`);
    }
  }

  // Update buyer journey stage
  const session = sessions.get(sessionId);
  if (session) {
    const stage = detectJourneyStage(session.history);
    const prev = journeyStages.get(sessionId);
    if (stage !== prev) {
      journeyStages.set(sessionId, stage);
      console.log(`[meridian-voice] Journey: ${prev || 'new'} → ${stage} (session ${sessionId})`);
    }
  }
}

// ============== Journey Funnel ==============
app.get("/api/journey", (req, res) => {
  const authHeader = req.headers.authorization;
  if (authHeader !== `Bearer ${AUTH_CODE}`) {
    return res.status(401).json({ error: "Unauthorized" });
  }
  const funnel = { lead: 0, qualified: 0, booked: 0, total: 0 };
  for (const [id, stage] of journeyStages.entries()) {
    funnel[stage] = (funnel[stage] || 0) + 1;
    funnel.total++;
  }
  // Also include sessions without a stage yet (they're leads)
  for (const [id] of sessions.entries()) {
    if (!journeyStages.has(id)) {
      funnel.lead++;
      funnel.total++;
    }
  }
  res.json(funnel);
});

// ============== Transcript Retrieval ==============
app.get("/api/transcripts", (req, res) => {
  const authHeader = req.headers.authorization;
  if (authHeader !== `Bearer ${AUTH_CODE}`) {
    return res.status(401).json({ error: "Unauthorized" });
  }
  try {
    const files = fs.readdirSync(TRANSCRIPTS_DIR).filter(f => f.endsWith(".jsonl"));
    const transcripts = files.map(f => {
      const lines = fs.readFileSync(path.join(TRANSCRIPTS_DIR, f), "utf-8").trim().split("\n");
      return {
        sessionId: f.replace(".jsonl", ""),
        exchanges: lines.map(l => JSON.parse(l)),
      };
    });
    res.json({ sessions: transcripts });
  } catch (e) {
    res.json({ sessions: [] });
  }
});

app.get("/api/transcripts/:sessionId", (req, res) => {
  const authHeader = req.headers.authorization;
  if (authHeader !== `Bearer ${AUTH_CODE}`) {
    return res.status(401).json({ error: "Unauthorized" });
  }
  const logFile = path.join(TRANSCRIPTS_DIR, `${req.params.sessionId}.jsonl`);
  try {
    const lines = fs.readFileSync(logFile, "utf-8").trim().split("\n");
    res.json({ sessionId: req.params.sessionId, exchanges: lines.map(l => JSON.parse(l)) });
  } catch (e) {
    res.status(404).json({ error: "Session not found" });
  }
});

// ============== TTS Probe (disable OpenAI if quota exceeded) ==============
setTimeout(async () => {
  if (!openai) return;
  try {
    const r = await openai.audio.speech.create({ model: "tts-1", voice: "nova", input: "hi", response_format: "opus" });
    const buf = Buffer.from(await r.arrayBuffer());
    if (buf.length > 0) { console.log("[meridian-voice] OpenAI TTS: ACTIVE"); return; }
  } catch (e) {
    console.log("[meridian-voice] OpenAI TTS: DISABLED (quota exceeded) — using browser TTS");
    openai = null;
  }
}, 2000);

// ============== Auth ==============
const VOICE_API_URL = process.env.VOICE_API_URL || "http://127.0.0.1:8100";

// Legacy code-based auth — now with per-customer gateway key
app.post("/api/auth", (req, res) => {
  const { code } = req.body;
  const match = AUTH_CODES[code];
  if (match) {
    // Return gateway key indicator (not the key itself — that stays server-side)
    const hasGatewayKey = !!(match.gatewayKey || GATEWAY_API_KEY);
    res.json({ success: true, department: match.dept, departmentName: match.name, workspaceId: match.workspace || "demo", gatewayEnabled: hasGatewayKey });
  } else {
    res.status(401).json({ success: false, error: "Invalid code" });
  }
});

// JWT registration — proxies to voice API
app.post("/api/auth/register", async (req, res) => {
  try {
    const { email, password, companyName } = req.body;
    if (!email || !password) return res.status(400).json({ error: "Email and password required" });
    const resp = await fetch(`${VOICE_API_URL}/api/v1/auth/register`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password, name: companyName || email.split("@")[0] }),
      signal: AbortSignal.timeout(10000),
    });
    const data = await resp.json();
    if (!resp.ok) return res.status(resp.status).json(data);
    console.log(`[meridian-voice] Registered: ${email} → workspace ${data.workspaceId || data.workspace_id}`);
    res.json({
      token: data.token,
      user: { id: data.userId || data.user_id, email, workspaceId: data.workspaceId || data.workspace_id, companyName: companyName || "" },
    });
  } catch (e) {
    res.status(502).json({ error: "Registration service unavailable" });
  }
});

// JWT login — proxies to voice API
app.post("/api/auth/login", async (req, res) => {
  try {
    const { email, password } = req.body;
    if (!email || !password) return res.status(400).json({ error: "Email and password required" });
    const resp = await fetch(`${VOICE_API_URL}/api/v1/auth/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password }),
      signal: AbortSignal.timeout(10000),
    });
    const data = await resp.json();
    if (!resp.ok) return res.status(resp.status).json(data);
    res.json({
      token: data.token,
      user: { id: data.userId || data.user_id, email, workspaceId: data.workspaceId || data.workspace_id, companyName: data.displayName || "" },
    });
  } catch (e) {
    res.status(502).json({ error: "Login service unavailable" });
  }
});

// Extract workspace from JWT, body, header, or fallback to demo
function extractWorkspace(req) {
  if (req.body?.workspaceId) return req.body.workspaceId;
  const auth = req.headers.authorization?.replace("Bearer ", "") || "";
  if (auth.includes(".")) {
    try {
      const payload = JSON.parse(Buffer.from(auth.split(".")[1], "base64").toString());
      if (payload.workspace_id) return payload.workspace_id;
    } catch {}
  }
  if (req.headers["x-workspace-id"]) return req.headers["x-workspace-id"];
  return "demo";
}

// ============== Lead Capture Endpoint ==============
app.post("/api/lead", (req, res) => {
  const authHeader = req.headers.authorization;
  if (authHeader !== `Bearer ${AUTH_CODE}`) {
    return res.status(401).json({ error: "Unauthorized" });
  }

  const { name, email, phone, company, notes } = req.body;
  if (!name && !email && !phone) {
    return res.status(400).json({ error: "At least name, email, or phone is required" });
  }

  const lead = saveLead({ name, email, phone, company, notes });
  res.json({ success: true, id: lead.id });
});

// ============== Conversation Sessions ==============
// Sessions track history + department (set at auth time)
const sessions = new Map();
const sessionDepartments = new Map(); // sessionId → department string

// ============== LuvBuds Intelligence Assistant Prompt ==============
function getSystemPrompt(department) {
  const now = new Date();
  const currentDate = now.toLocaleDateString("en-US", {
    weekday: "long",
    year: "numeric",
    month: "long",
    day: "numeric",
    timeZone: "America/New_York",
  });
  const currentTime = now.toLocaleTimeString("en-US", {
    hour: "numeric",
    minute: "2-digit",
    timeZone: "America/New_York",
  });

  const base = `You are the LuvBuds Intelligence Assistant — an internal AI that helps LuvBuds team members find information from company systems.

You're on a voice call. Keep responses concise, specific, and action-oriented.

VOICE RULES:
- 1-4 sentences for simple lookups. More detail if they ask to elaborate.
- Answer directly from the RELEVANT CUSTOMER DATA section below — that's your source of truth.
- Be specific: cite vendor names, product names, order numbers, dollar amounts when available.
- NEVER fabricate data, numbers, or names. Only state what's in the search results.
- PRONUNCIATION: Say "P.O." not "PO", "S.K.U." not "SKU", "K.P.I." not "KPI", "A.P.I." not "API".
- When listing items, read the top 3-5 and say "and X more" for the rest.

COMPANY CONTEXT:
- LuvBuds is a smoke shop distribution and retail company
- Systems: Microsoft Dynamics 365 (E.R.P.), Notion (tasks/docs), BigCommerce (e-commerce)
- Data loaded: 92,000+ records including products, vendors, purchase orders, sales orders, tasks, decisions
- Brands include: Blazer, Stache, and many more across smoke shop product categories

WHAT YOU CAN ANSWER:
- Product lookups: names, brands, categories, S.K.U.s
- Vendor info: who supplies what, contact details if available
- Purchase orders: P.O. counts, line items, vendor spend
- Sales orders: order volumes, sales data (52,000+ orders loaded)
- Tasks and decisions: what's in progress, what was decided
- Documents: internal docs, S.O.P.s, meeting notes

HOW TO HANDLE ANALYTICAL QUESTIONS (top sellers, totals, rankings):
- The search results below are a SAMPLE from the knowledge base, not the full dataset.
- When asked about "top" or "best" items: report what you see in the search results and say "based on the records I can see." Do NOT say you don't have the data if results ARE present below — work with what you have.
- When asked about counts or totals: if the DATA AVAILABLE section shows record counts, cite those. For specific filtered counts, work from the search results.
- If the search results genuinely don't contain relevant data, say "That specific breakdown isn't in the data I can search right now — we'd need to add it to the system."
- NEVER say "I don't have sales data" when sales order records appear in the results below.`;

  const salesContext = `

SALES TEAM FOCUS:
- Help with: product availability, brand info, customer questions, pricing lookups, competitor product comparisons
- 3,525 product S.K.U.s, 131 brands, 270 categories loaded
- 52,100 sales orders from Dynamics loaded — use these to identify popular products and order patterns
- When asked about "best sellers" or "top products": report the products and sales orders visible in the search results. Say "based on the data I can see" — don't claim you lack data if results are present.
- Connect product questions to vendor and procurement data when useful
- If asked about margins or pricing details not in the results, say "I have product and P.O. data — let me check what's available"`;

  const accountingContext = `

ACCOUNTING TEAM FOCUS:
- Help with: P.O. status, vendor spend analysis, payment tracking, cost breakdowns, order reconciliation
- When asked about totals or aggregates: cite the data you have and note if it's partial
- 20,000+ P.O. line items and 3,598 purchase orders are loaded
- 837 vendor records and 246 account records available
- 52,100 sales orders from Dynamics loaded
- If asked about invoices or payments not in the data, say "that level of detail needs the Dynamics dashboard"`;

  const leadershipContext = `

LEADERSHIP FOCUS:
- Help with: high-level summaries, cross-department questions, strategic data
- You have visibility across all departments: sales, accounting, operations
- When asked for overviews, pull from multiple categories
- Flag when data is incomplete: "We have 52,000 sales orders loaded but not the line-item detail yet"`;

  let prompt = base;
  if (department === "sales") prompt += salesContext;
  else if (department === "accounting") prompt += accountingContext;
  else if (department === "leadership") prompt += leadershipContext;
  else prompt += salesContext + accountingContext;

  prompt += `\n\nCurrent date: ${currentDate}\nCurrent time: ${currentTime} ET`;
  return prompt;
}

// ============== Main Voice Endpoint ==============
app.post("/api/voice", async (req, res) => {
  const authHeader = req.headers.authorization;
  const tokenNS = authHeader?.replace("Bearer ", "") || "";
  const authMatchNS = AUTH_CODES[tokenNS] || (tokenNS.includes(".") ? { dept: "all", name: "JWT User" } : null);
  if (!authMatchNS) {
    return res.status(401).json({ error: "Unauthorized" });
  }

  const { audio, sessionId, voice } = req.body;
  const workspaceId = extractWorkspace(req);
  if (sessionId && authMatchNS.dept) {
    sessionDepartments.set(sessionId, authMatchNS.dept);
  }
  // Store per-session gateway key for per-customer rate limiting
  if (sessionId && !sessionGatewayKeys.has(sessionId)) {
    sessionGatewayKeys.set(sessionId, authMatchNS.gatewayKey || GATEWAY_API_KEY);
  }
  const selectedVoice = voice || "nova";

  if (!audio) {
    return res.status(400).json({ error: "No audio provided" });
  }

  try {
    console.log(`[meridian-voice] Processing audio for session ${sessionId} (workspace: ${workspaceId})`);
    const startTime = Date.now();

    // 1. Convert base64 audio to buffer
    const audioBuffer = Buffer.from(audio, "base64");
    console.log(`[meridian-voice] Audio size: ${audioBuffer.length} bytes`);

    // 2. STT with Groq Whisper
    const sttStart = Date.now();
    const transcript = await speechToText(audioBuffer);
    const sttTime = Date.now() - sttStart;
    console.log(`[meridian-voice] STT (${sttTime}ms): "${transcript}"`);

    if (!transcript || transcript.trim().length === 0) {
      return res.json({
        transcript: "",
        response: "",
        audio: null,
        timing: { total: Date.now() - startTime },
      });
    }

    // 3. Get or create session history
    if (!sessions.has(sessionId)) {
      const restored = loadSessionFromTranscript(workspaceId, sessionId);
      sessions.set(sessionId, { history: restored, lastAccess: Date.now(), workspaceId });
    }
    const session = sessions.get(sessionId);
    session.lastAccess = Date.now();
    const history = session.history;

    // 4. Get AI response from Groq LLM
    const aiStart = Date.now();
    const deptNonStream = sessionDepartments.get(sessionId) || "all";
    let overridePrompt = null;
    if (isMeridianWorkspace(workspaceId)) {
      const migContext = await queryMeridianMIG(transcript);
      overridePrompt = getMeridianPrompt() + migContext;
    }
    const response = await getAIResponse(transcript, history, deptNonStream, overridePrompt, sessionGatewayKeys.get(sessionId) || "");
    const aiTime = Date.now() - aiStart;
    console.log(`[meridian-voice] AI (${aiTime}ms): "${response}"`);

    // Update history + log
    history.push({ role: "user", content: transcript });
    history.push({ role: "assistant", content: response });
    while (history.length > 20) history.shift();
    logTranscript(workspaceId, sessionId, transcript, response, { stt: sttTime, ai: aiTime });
    persistToMIG(workspaceId, sessionId, transcript, response);

    // 5. TTS
    const ttsStart = Date.now();
    let responseAudio = null;
    try {
      if (selectedVoice.startsWith("chatterbox:")) {
        responseAudio = await chatterboxTTS(response, selectedVoice.replace("chatterbox:", ""));
      } else if (selectedVoice.startsWith("edge:")) {
        responseAudio = await edgeTTS(response, selectedVoice.replace("edge:", ""));
      } else {
        // Try Edge-TTS first (free, high quality), fall back to OpenAI
        responseAudio = await edgeTTS(response) || await textToSpeech(response, selectedVoice);
      }
    } catch (e) {
      console.warn(`[meridian-voice] TTS failed, returning text-only: ${e.message}`);
    }
    const ttsTime = Date.now() - ttsStart;

    const totalTime = Date.now() - startTime;
    console.log(`[meridian-voice] Total: ${totalTime}ms`);

    res.json({
      transcript,
      response,
      audio: responseAudio ? responseAudio.toString("base64") : null,
      timing: { stt: sttTime, ai: aiTime, tts: ttsTime, total: totalTime },
    });
  } catch (error) {
    console.error("[meridian-voice] Error:", error);
    res.status(500).json({ error: error.message });
  }
});

// ============== Streaming Voice Endpoint ==============
app.post("/api/voice/stream", async (req, res) => {
  const authHeader = req.headers.authorization;
  const token = authHeader?.replace("Bearer ", "") || "";
  // Accept auth codes OR JWTs (JWT contains a dot)
  const authMatch = AUTH_CODES[token] || (token.includes(".") ? { dept: "all", name: "JWT User" } : null);
  if (!authMatch) {
    return res.status(401).json({ error: "Unauthorized" });
  }

  const { audio, transcript: clientTranscript, sessionId, voice } = req.body;
  const workspaceId = extractWorkspace(req);
  // Track department + gateway key per session
  if (sessionId && authMatch.dept) {
    sessionDepartments.set(sessionId, authMatch.dept);
  }
  if (sessionId && !sessionGatewayKeys.has(sessionId)) {
    sessionGatewayKeys.set(sessionId, authMatch.gatewayKey || GATEWAY_API_KEY);
  }
  const selectedVoice = voice || "nova";

  if (!audio && !clientTranscript) {
    return res.status(400).json({ error: "No audio or transcript provided" });
  }

  // Set up SSE
  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "keep-alive");

  try {
    const startTime = Date.now();

    // 1. STT — use client-provided transcript (Web Speech API) or fall back to Groq Whisper
    let transcript;
    let sttTime;
    if (clientTranscript) {
      transcript = clientTranscript;
      sttTime = 0;
      console.log(`[meridian-voice/stream] Fast path — client STT: "${transcript}"`);
    } else {
      const audioBuffer = Buffer.from(audio, "base64");
      const sttStart = Date.now();
      transcript = await speechToText(audioBuffer);
      sttTime = Date.now() - sttStart;
    }

    // Send transcript immediately
    res.write(`data: ${JSON.stringify({ type: "transcript", text: transcript, sttTime })}\n\n`);

    if (!transcript || transcript.trim().length === 0) {
      res.write(
        `data: ${JSON.stringify({ type: "done", timing: { total: Date.now() - startTime } })}\n\n`
      );
      return res.end();
    }

    // 2. Get/create session
    if (!sessions.has(sessionId)) {
      const restored = loadSessionFromTranscript(workspaceId, sessionId);
      sessions.set(sessionId, { history: restored, lastAccess: Date.now(), workspaceId });
    }
    const session = sessions.get(sessionId);
    session.lastAccess = Date.now();
    session.workspaceId = session.workspaceId || workspaceId;
    const history = session.history;

    // 3. Stream AI response and start TTS on sentences
    const aiStart = Date.now();

    let fullResponse = "";
    let sentenceBuffer = "";
    let sentenceCount = 0;

    console.log(`[meridian-voice/stream] Streaming AI response...`);

    // RAG: query customer knowledge base + KPI dashboard in parallel
    const dept = sessionDepartments.get(sessionId) || "all";
    let systemPrompt;
    if (isMeridianWorkspace(workspaceId)) {
      // Meridian house workspace — Oracle identity + MIG knowledge via KaaS relay
      const migContext = await queryMeridianMIG(transcript);
      systemPrompt = getMeridianPrompt() + migContext;
      console.log(`[meridian-voice/stream] Meridian house prompt + MIG context for workspace ${workspaceId}`);
    } else {
      const [ragContext, kpiContext] = await Promise.all([
        queryCustomerKnowledge(transcript, dept, sessionGatewayKeys.get(sessionId) || ""),
        queryKPIDashboard(transcript),
      ]);
      systemPrompt = getSystemPrompt(dept) + ragContext + kpiContext;
    }

    // Stream from Groq LLM
    const stream = await groq.chat.completions.create({
      model: CHAT_MODEL,
      messages: [
        { role: "system", content: systemPrompt },
        ...history,
        { role: "user", content: transcript },
      ],
      max_tokens: 300,
      stream: true,
    });

    for await (const chunk of stream) {
      const content = chunk.choices[0]?.delta?.content || "";
      fullResponse += content;
      sentenceBuffer += content;

      // Check for complete sentences
      const { sentences, remaining } = extractSentences(sentenceBuffer);
      sentenceBuffer = remaining;

      for (const sentence of sentences) {
        sentenceCount++;
        console.log(`[meridian-voice/stream] Sentence ${sentenceCount}: "${sentence}"`);

        // Start TTS immediately for this sentence
        const ttsStart = Date.now();
        let audioData = null;
        try {
          // Use Chatterbox for cloned voices, Edge-TTS for edge: prefix, else Edge-TTS → OpenAI fallback
          if (selectedVoice.startsWith("chatterbox:")) {
            const cbVoice = selectedVoice.replace("chatterbox:", "");
            audioData = await chatterboxTTS(sentence, cbVoice);
          } else if (selectedVoice.startsWith("edge:")) {
            audioData = await edgeTTS(sentence, selectedVoice.replace("edge:", ""));
          } else {
            audioData = await edgeTTS(sentence) || await textToSpeech(sentence, selectedVoice);
          }
        } catch (e) {
          console.warn(`[meridian-voice/stream] TTS failed for sentence: ${e.message}`);
        }
        const ttsTime = Date.now() - ttsStart;

        // Send audio chunk (or text-only if TTS failed)
        res.write(
          `data: ${JSON.stringify({
            type: "audio",
            audio: audioData ? audioData.toString("base64") : null,
            text: sentence,
            sentenceNum: sentenceCount,
            ttsTime,
          })}\n\n`
        );
      }
    }

    // Handle any remaining text
    if (sentenceBuffer.trim()) {
      let audioData = null;
      try {
        if (selectedVoice.startsWith("chatterbox:")) {
          audioData = await chatterboxTTS(sentenceBuffer.trim(), selectedVoice.replace("chatterbox:", ""));
        } else if (selectedVoice.startsWith("edge:")) {
          audioData = await edgeTTS(sentenceBuffer.trim(), selectedVoice.replace("edge:", ""));
        } else {
          audioData = await edgeTTS(sentenceBuffer.trim()) || await textToSpeech(sentenceBuffer.trim(), selectedVoice);
        }
      } catch (e) {
        console.warn(`[meridian-voice/stream] TTS failed for final chunk: ${e.message}`);
      }
      res.write(
        `data: ${JSON.stringify({
          type: "audio",
          audio: audioData ? audioData.toString("base64") : null,
          text: sentenceBuffer.trim(),
          sentenceNum: ++sentenceCount,
          ttsTime: 0,
        })}\n\n`
      );
      fullResponse = fullResponse.trim();
    }

    const aiTime = Date.now() - aiStart;

    // Update history + log
    history.push({ role: "user", content: transcript });
    history.push({ role: "assistant", content: fullResponse });
    while (history.length > 20) history.shift();
    logTranscript(workspaceId, sessionId, transcript, fullResponse, { stt: sttTime, ai: aiTime });
    persistToMIG(workspaceId, sessionId, transcript, fullResponse);

    // Send done
    res.write(
      `data: ${JSON.stringify({
        type: "done",
        response: fullResponse,
        timing: { stt: sttTime, ai: aiTime, total: Date.now() - startTime },
      })}\n\n`
    );

    console.log(`[meridian-voice/stream] Complete in ${Date.now() - startTime}ms`);
    res.end();
  } catch (error) {
    console.error("[meridian-voice/stream] Error:", error);
    res.write(`data: ${JSON.stringify({ type: "error", message: error.message })}\n\n`);
    res.end();
  }
});

// ============== STT: Groq Whisper ==============
async function speechToText(audioBuffer) {
  const file = await toFile(audioBuffer, "audio.webm", { type: "audio/webm" });

  const response = await groq.audio.transcriptions.create({
    model: "whisper-large-v3",
    file: file,
    language: "en",
  });

  return response.text;
}

// ============== Chat: Groq LLM ==============
async function getAIResponse(userMessage, history, department, overridePrompt = null, apiKey = "") {
  let systemPrompt;
  if (overridePrompt) {
    systemPrompt = overridePrompt;
  } else {
    const [ragContext, kpiContext] = await Promise.all([
      queryCustomerKnowledge(userMessage, department, apiKey),
      queryKPIDashboard(userMessage),
    ]);
    systemPrompt = getSystemPrompt(department) + ragContext + kpiContext;
  }

  const response = await groq.chat.completions.create({
    model: CHAT_MODEL,
    messages: [
      { role: "system", content: systemPrompt },
      ...history,
      { role: "user", content: userMessage },
    ],
    max_tokens: 300,
  });

  return response.choices[0].message.content;
}

// ============== TTS: OpenAI (with graceful fallback) ==============
async function textToSpeech(text, voice = "nova", useCache = true) {
  if (!openai) return null;

  // Check cache first
  if (useCache) {
    const cached = getFromCache(text, voice);
    if (cached) {
      console.log(`[meridian-voice] TTS cache HIT for: "${text.substring(0, 30)}..."`);
      return Buffer.from(cached, "base64");
    }
  }

  try {
    const response = await openai.audio.speech.create({
      model: "tts-1",
      voice: voice,
      input: text,
      response_format: "opus",
    });

    const arrayBuffer = await response.arrayBuffer();
    const buffer = Buffer.from(arrayBuffer);

    // Cache the result
    if (useCache && text.length < 200) {
      addToCache(text, voice, buffer.toString("base64"));
    }

    return buffer;
  } catch (error) {
    // Graceful fallback: if OpenAI TTS is out of quota or errors, return null
    console.warn(`[meridian-voice] TTS error (returning null): ${error.message}`);
    return null;
  }
}

// ============== TTS: Edge-TTS (free Microsoft voices, no API key) ==============
// Edge-TTS voices for UK English:
//   en-GB-SoniaNeural (female, default — closest to Google UK Female)
//   en-GB-RyanNeural (male)
//   en-GB-LibbyNeural (female, warm)
//   en-GB-MaisieNeural (female, young)
// Also available: en-US-*, en-AU-*, en-IN-*, plus 300+ other voices

const EDGE_TTS_VOICES = {
  "en-GB-SoniaNeural":  { name: "Sonia (UK Female)",  lang: "en-GB", gender: "female" },
  "en-GB-RyanNeural":   { name: "Ryan (UK Male)",     lang: "en-GB", gender: "male" },
  "en-GB-LibbyNeural":  { name: "Libby (UK Female)",  lang: "en-GB", gender: "female" },
  "en-GB-MaisieNeural": { name: "Maisie (UK Female)", lang: "en-GB", gender: "female" },
  "en-US-JennyNeural":  { name: "Jenny (US Female)",  lang: "en-US", gender: "female" },
  "en-US-GuyNeural":    { name: "Guy (US Male)",      lang: "en-US", gender: "male" },
  "en-US-AriaNeural":   { name: "Aria (US Female)",   lang: "en-US", gender: "female" },
  "en-US-DavisNeural":  { name: "Davis (US Male)",    lang: "en-US", gender: "male" },
};

const DEFAULT_EDGE_VOICE = "en-GB-SoniaNeural";

async function edgeTTS(text, voice = DEFAULT_EDGE_VOICE) {
  if (!text || text.trim().length === 0) return null;

  // Check cache first (reuses the same cache as OpenAI TTS)
  const cacheKey = `edge:${voice}`;
  const cached = getFromCache(text, cacheKey);
  if (cached) {
    console.log(`[meridian-voice] Edge-TTS cache HIT for: "${text.substring(0, 30)}..."`);
    return Buffer.from(cached, "base64");
  }

  // Determine lang from voice name (e.g. en-GB-SoniaNeural → en-GB)
  const langMatch = voice.match(/^([a-z]{2}-[A-Z]{2})/);
  const lang = langMatch ? langMatch[1] : "en-GB";

  const tts = new EdgeTTS({
    voice,
    lang,
    outputFormat: "audio-24khz-48kbitrate-mono-mp3",
    timeout: 15000,
  });

  // Write to temp file, read back as buffer, clean up
  const tmpFile = path.join(os.tmpdir(), `edge-tts-${Date.now()}-${Math.random().toString(36).slice(2, 8)}.mp3`);
  try {
    await tts.ttsPromise(text, tmpFile);
    const buffer = fs.readFileSync(tmpFile);
    fs.unlinkSync(tmpFile);

    // Cache the result
    if (text.length < 200) {
      addToCache(text, cacheKey, buffer.toString("base64"));
    }

    return buffer;
  } catch (e) {
    console.warn(`[meridian-voice] Edge-TTS failed: ${e.message || e}`);
    // Clean up temp file on error
    try { fs.unlinkSync(tmpFile); } catch {}
    return null;
  }
}

// ============== TTS: Chatterbox (local voice cloning) ==============
const CHATTERBOX_URL = process.env.CHATTERBOX_URL || "http://127.0.0.1:8766";

async function chatterboxTTS(text, voice = "default") {
  try {
    const response = await fetch(`${CHATTERBOX_URL}/synthesize`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, voice, exaggeration: 0.5, cfg_weight: 0.5 }),
      signal: AbortSignal.timeout(30000),
    });
    if (!response.ok) return null;
    const arrayBuffer = await response.arrayBuffer();
    return Buffer.from(arrayBuffer);
  } catch (error) {
    console.warn(`[meridian-voice] Chatterbox TTS failed: ${error.message}`);
    return null;
  }
}

// Check if Chatterbox is available on startup
let chatterboxAvailable = false;
setTimeout(async () => {
  try {
    const res = await fetch(`${CHATTERBOX_URL}/health`, { signal: AbortSignal.timeout(3000) });
    if (res.ok) {
      const data = await res.json();
      chatterboxAvailable = data.status === "ready";
      console.log(`[meridian-voice] Chatterbox: ${chatterboxAvailable ? "AVAILABLE" : "not ready"} (${data.device})`);
    }
  } catch {
    console.log("[meridian-voice] Chatterbox: not running");
  }
}, 1000);

// ============== Voice Training Proxy ==============
app.post("/api/train-voice/:name", async (req, res) => {
  const authHeader = req.headers.authorization;
  if (authHeader !== `Bearer ${AUTH_CODE}`) {
    return res.status(401).json({ error: "Unauthorized" });
  }

  const { audio } = req.body;
  if (!audio) {
    return res.status(400).json({ error: "No audio data" });
  }

  try {
    const audioBuffer = Buffer.from(audio, "base64");
    const response = await fetch(`${CHATTERBOX_URL}/train-voice/${req.params.name}`, {
      method: "POST",
      headers: { "Content-Type": "application/octet-stream" },
      body: audioBuffer,
      signal: AbortSignal.timeout(10000),
    });
    const data = await response.json();
    console.log(`[meridian-voice] Voice trained: ${req.params.name} (${audioBuffer.length} bytes)`);
    res.json(data);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// ============== Voice List ==============
app.get("/api/voices", async (req, res) => {
  // Build Edge-TTS voice list (always available — no API key needed)
  const edgeVoices = Object.entries(EDGE_TTS_VOICES).map(([id, info]) => ({
    id: `edge:${id}`,
    name: info.name,
    lang: info.lang,
    gender: info.gender,
    provider: "edge-tts",
  }));

  try {
    const response = await fetch(`${CHATTERBOX_URL}/voices`, { signal: AbortSignal.timeout(3000) });
    const data = await response.json();
    res.json({ ...data, chatterbox: true, edgeVoices });
  } catch {
    res.json({ voices: ["browser-uk-female", "browser-uk-male"], chatterbox: false, edgeVoices });
  }
});

// ============== TTS Test ==============
app.post("/api/tts-test", async (req, res) => {
  const authHeader = req.headers.authorization;
  if (authHeader !== `Bearer ${AUTH_CODE}`) {
    return res.status(401).json({ error: "Unauthorized" });
  }

  const { text, voice } = req.body;
  const testText = text || "Hello, this is how I sound. Nice to meet you.";

  try {
    const audioBuffer = await chatterboxTTS(testText, voice || "default");
    if (audioBuffer) {
      res.json({ audio: audioBuffer.toString("base64"), format: "wav", voice: voice || "default" });
    } else {
      res.json({ audio: null, error: "Chatterbox unavailable" });
    }
  } catch (error) {
    res.json({ audio: null, error: error.message });
  }
});

// ============== Sentence Extraction ==============
function extractSentences(text) {
  const sentences = [];
  let remaining = text;

  const sentenceRegex = /(?<![A-Z][a-z]?\.)(?<![A-Z]\.)(?<=\.|\?|!)\s+/;
  const parts = remaining.split(sentenceRegex);

  for (let i = 0; i < parts.length - 1; i++) {
    if (parts[i].trim()) sentences.push(parts[i].trim());
  }

  const lastPart = parts[parts.length - 1];
  return { sentences, remaining: lastPart };
}

// ============== Session End + Cleanup ==============

// Explicitly end a session (called from end-call button)
app.post("/api/session/end", async (req, res) => {
  const { sessionId } = req.body;
  if (!sessionId) return res.status(400).json({ error: "sessionId required" });

  const session = sessions.get(sessionId);
  if (!session) return res.json({ ok: true, summary: false });

  const workspaceId = session.workspaceId || extractWorkspace(req);
  // Fire summary generation in the background — don't block the response
  generateSessionSummary(workspaceId, sessionId, session.history)
    .catch(e => console.warn(`[meridian-voice] End-call summary failed: ${e.message}`));

  sessions.delete(sessionId);
  sessionGatewayKeys.delete(sessionId);
  console.log(`[meridian-voice] Session ended by client: ${sessionId}`);
  res.json({ ok: true, summary: isMeridianWorkspace(workspaceId) });
});

// Periodic cleanup — also generates summaries for timed-out Meridian sessions
setInterval(() => {
  const now = Date.now();
  for (const [id, session] of sessions.entries()) {
    if (session.lastAccess && now - session.lastAccess > 30 * 60 * 1000) {
      console.log(`[meridian-voice] Clearing stale session: ${id}`);
      const workspaceId = session.workspaceId || "demo";
      // Generate summary for Meridian sessions before eviction (fire-and-forget)
      generateSessionSummary(workspaceId, id, session.history)
        .catch(e => console.warn(`[meridian-voice] Stale session summary failed: ${e.message}`));
      sessions.delete(id);
      sessionGatewayKeys.delete(id);
    }
  }
}, 30 * 60 * 1000);

// ============== Start Server ==============
initBridge();
setInterval(() => cleanupSessions(), 30 * 60 * 1000);

app.listen(PORT, "0.0.0.0", () => {
  console.log(`[meridian-voice] Server running on http://localhost:${PORT}`);
  console.log(`[meridian-voice] Groq API: configured`);
  console.log(`[meridian-voice] OpenAI TTS: ${openai ? "configured" : "DISABLED (no key)"}`);
  console.log(`[meridian-voice] Auth code: ${AUTH_CODE ? "set" : "none"}`);
});
