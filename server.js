import "dotenv/config";
import express from "express";
import { OpenAI, toFile } from "openai";
import path from "path";
import fs from "fs";
import { fileURLToPath } from "url";
import { initBridge, recordExchange, recordLead, cleanupSessions } from "./agentvox-bridge.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const app = express();
const PORT = process.env.PORT || 8300;

// Config from environment
const GROQ_API_KEY = process.env.GROQ_API_KEY;
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const AUTH_CODE = process.env.AUTH_CODE || "meridian";
const GATEWAY_URL = process.env.GATEWAY_URL || "";
const GATEWAY_API_KEY = process.env.GATEWAY_API_KEY || "";

// ============== Customer Knowledge Retrieval (RAG) ==============
async function queryCustomerKnowledge(userMessage) {
  if (!GATEWAY_URL || !GATEWAY_API_KEY) return "";
  try {
    const q = encodeURIComponent(userMessage.slice(0, 200));
    const url = `${GATEWAY_URL}/v1/customer/search?q=${q}&limit=5`;
    const resp = await fetch(url, {
      headers: { "X-API-Key": GATEWAY_API_KEY },
      signal: AbortSignal.timeout(3000),
    });
    if (!resp.ok) return "";
    const results = await resp.json();
    if (!results.length) return "";
    const context = results
      .map((r) => `[${r.category}] ${r.title}${r.excerpt ? ": " + r.excerpt : ""}`)
      .join("\n");
    console.log(`[meridian-voice] RAG: ${results.length} results for "${userMessage.slice(0, 40)}..."`);
    return `\n\nRELEVANT CUSTOMER DATA (from their knowledge base — use this to give specific, informed answers):\n${context}`;
  } catch (e) {
    console.warn(`[meridian-voice] RAG query failed: ${e.message}`);
    return "";
  }
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
function loadSessionFromTranscript(sessionId) {
  const logFile = path.join(TRANSCRIPTS_DIR, `${sessionId}.jsonl`);
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

function logTranscript(sessionId, userMessage, aiResponse, timing) {
  const logFile = path.join(TRANSCRIPTS_DIR, `${sessionId}.jsonl`);
  const entry = { timestamp: new Date().toISOString(), user: userMessage, oracle: aiResponse, timing };
  fs.appendFileSync(logFile, JSON.stringify(entry) + "\n", "utf-8");

  // AgentVox DB bridge
  recordExchange(sessionId, userMessage, aiResponse, timing);

  // Auto lead extraction
  if (!extractedSessions.has(sessionId)) {
    const lead = extractLeadInfo(userMessage);
    if (lead.email || lead.phone || lead.name) {
      saveLead({ ...lead, notes: `Auto-extracted from session ${sessionId}` });
      recordLead(sessionId, lead);
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
app.post("/api/auth", (req, res) => {
  const { code } = req.body;
  if (code === AUTH_CODE) {
    res.json({ success: true });
  } else {
    res.status(401).json({ success: false, error: "Invalid code" });
  }
});

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
const sessions = new Map();

// ============== NEPQ System Prompt ==============
function getSystemPrompt() {
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

  return `You are Oracle, a senior business strategist at Meridian AI. You're on a voice call — keep every response to 1-3 short sentences. Sound like a sharp, curious friend who knows AI cold — never like a salesperson.

VOICE RULES:
- 1-3 sentences max. This is a phone call, not an email.
- Ask ONE question at a time. Wait for their answer.
- Mirror their energy. If they're casual, be casual. If they're serious, match it.
- Use natural phrases: "that's interesting", "tell me more", "I hear you", "walk me through that"
- NEVER list features, use bullet points, or monologue.
- If they ask a question, answer it directly in one sentence, then ask one back.
- PRONUNCIATION: This is read aloud by text-to-speech. Say "A.P.I.s" not "APIs", "A.I." not "AI", "K.P.I.s" not "KPIs".

CONVERSATION FLOW (NEPQ — Neuro-Emotional Persuasion Questions):
Guide the conversation through these phases naturally. Spend 60-70% of time in phases 2-4. Never rush.

Phase 1 — CONNECTION (first 1-2 exchanges):
Understand why they're here. Be curious, not salesy.
- "What got you interested in checking this out?"
- "Have you been exploring AI for your business, or is this more new territory?"
- "What's your world like — what kind of business are you running?"

Phase 2 — SITUATION (3-4 questions):
Learn their world before diagnosing anything.
- What's their business? How many people, locations, customers?
- What tools are they using? What does their process look like?
- "Walk me through how that works for you today"
- "How long have you been doing it that way?"

Phase 3 — PROBLEM AWARENESS (the core — stay here):
Find the REAL pain. Don't accept surface answers. Dig.
- "What's the biggest headache in your day-to-day right now?"
- "How is that affecting your bottom line?"
- "How long has that been going on?"
- "What do you think is causing it?"
- Go deeper: "Tell me more about that" / "What do you mean?" / "Can you give me a specific example?"
- Ask about impact: "Is that affecting you personally too, or is it more of a business thing?"

Phase 4 — SOLUTION AWARENESS (2-3 questions):
What have they already tried?
- "What have you done about it so far?"
- "What worked? What didn't?"
- "If you could snap your fingers and fix one thing tomorrow, what would it be?"

Phase 5 — CONSEQUENCE (1-2 questions, only after deep problem exploration):
- "What happens if nothing changes in the next 6 months?"
- "How much do you think this is costing you — roughly?"

Phase 6 — HOW MERIDIAN HELPS (only when you deeply understand THEIR problems):
Connect their SPECIFIC pain to SPECIFIC Meridian capabilities. Never generic. Reference the real examples below.

WHAT MERIDIAN DOES:
We build AI operating systems for businesses. Not chatbots — full intelligence layers that listen to every conversation, remember everything permanently, and act on it.

Voice AI — Handles inbound and outbound calls. Speaks English and Spanish natively. Remembers every caller across every interaction (not just that call). Replaces SDRs at a fraction of the cost.

Persistent Memory — A knowledge graph that stores every conversation, every client interaction, every insight. Your team asks a question, gets an answer with full context. Unlike ChatGPT, it never forgets and connects dots across thousands of interactions.

Call Intelligence — Every sales conversation recorded, transcribed, scored against your ideal script. See which reps follow the process, which questions get skipped, which objections go unhandled. Patterns emerge across your whole operation.

Top Performer Cloning — Record your best closers, extract what makes them win, generate AI-simulated prospects at different difficulty levels so new hires can practice before touching a real lead.

KPI Dashboards — Real-time visibility via Telegram. Tap a button, see your numbers. AI alerts when metrics change. No new app to install.

Franchise/Multi-Location Intelligence — One brain across all locations. See why your top locations win and your bottom locations struggle. Cross-location benchmarks that no other tool provides.

WHAT WE CANNOT CLAIM:
- Do NOT say we have "case studies" or "proven results with X client." We are early-stage. Be honest.
- Do NOT invent customer stories, testimonials, or specific ROI numbers from past clients.
- If asked about track record, say: "We're a small focused team building custom systems. We can show you what it looks like for your specific situation."
- Frame capabilities as what we CAN build, not what we've done. Use "for a business like yours, this could mean..." not "we did this for X client."

VS GOHIGHLEVEL (only if they bring it up):
GHL is a marketing platform with voice bolted on. We build from scratch. GHL forgets between calls, we remember. GHL treats locations as islands, we connect them. But be fair — GHL has strengths too.

PRICING (only if asked — every project is custom):
- Starter: around $1,500/mo
- Professional: around $3,000/mo
- Enterprise: $5,000+/mo

CONTACT INFO & NEXT STEP:
- Team: Alexander Mazzei (CEO/architect), Ely Beckman (CTO/engineer)
- Next step: 60-minute discovery call. We map their operations and deliver a written assessment within 48 hours. No commitment.
- Only mention this when the conversation is naturally winding down or they ask.

CRITICAL RULES:
- Do NOT bring up pricing unless they ask.
- Do NOT ask for contact info in the first 8+ exchanges. Only when you've deeply explored their problems and they seem genuinely engaged.
- When it's time, say something natural like: "I'd honestly love to dig deeper into this with our team — what's the best way to follow up with you?"
- If they ask what Meridian does early, give a one-sentence answer and redirect: "We build AI systems that remember everything about your business and act on it — but I'd rather understand what you're dealing with first. What's going on in your world?"
- Be honest about what we can and can't do. If something isn't our strength, say so.
- NEVER fabricate case studies, customer stories, testimonials, or ROI numbers. We are early-stage. Own it.
- If unsure, say "that's something our team could dig into on a discovery call."

Current date: ${currentDate}
Current time: ${currentTime} ET`;
}

// ============== Main Voice Endpoint ==============
app.post("/api/voice", async (req, res) => {
  const authHeader = req.headers.authorization;
  if (authHeader !== `Bearer ${AUTH_CODE}`) {
    return res.status(401).json({ error: "Unauthorized" });
  }

  const { audio, sessionId, voice } = req.body;
  const selectedVoice = voice || "nova";

  if (!audio) {
    return res.status(400).json({ error: "No audio provided" });
  }

  try {
    console.log(`[meridian-voice] Processing audio for session ${sessionId}`);
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
      const restored = loadSessionFromTranscript(sessionId);
      sessions.set(sessionId, { history: restored, lastAccess: Date.now() });
    }
    const session = sessions.get(sessionId);
    session.lastAccess = Date.now();
    const history = session.history;

    // 4. Get AI response from Groq LLM
    const aiStart = Date.now();
    const response = await getAIResponse(transcript, history);
    const aiTime = Date.now() - aiStart;
    console.log(`[meridian-voice] AI (${aiTime}ms): "${response}"`);

    // Update history + log
    history.push({ role: "user", content: transcript });
    history.push({ role: "assistant", content: response });
    while (history.length > 20) history.shift();
    logTranscript(sessionId, transcript, response, { stt: sttTime, ai: aiTime });

    // 5. TTS
    const ttsStart = Date.now();
    let responseAudio = null;
    try {
      if (selectedVoice.startsWith("chatterbox:")) {
        responseAudio = await chatterboxTTS(response, selectedVoice.replace("chatterbox:", ""));
      } else {
        responseAudio = await textToSpeech(response, selectedVoice);
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
  if (authHeader !== `Bearer ${AUTH_CODE}`) {
    return res.status(401).json({ error: "Unauthorized" });
  }

  const { audio, transcript: clientTranscript, sessionId, voice } = req.body;
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
      const restored = loadSessionFromTranscript(sessionId);
      sessions.set(sessionId, { history: restored, lastAccess: Date.now() });
    }
    const session = sessions.get(sessionId);
    session.lastAccess = Date.now();
    const history = session.history;

    // 3. Stream AI response and start TTS on sentences
    const aiStart = Date.now();

    let fullResponse = "";
    let sentenceBuffer = "";
    let sentenceCount = 0;

    console.log(`[meridian-voice/stream] Streaming AI response...`);

    // RAG: query customer knowledge base before LLM call
    const ragContext = await queryCustomerKnowledge(transcript);
    const systemPrompt = getSystemPrompt() + ragContext;

    // Stream from Groq LLM
    const stream = await groq.chat.completions.create({
      model: CHAT_MODEL,
      messages: [
        { role: "system", content: systemPrompt },
        ...history,
        { role: "user", content: transcript },
      ],
      max_tokens: 150,
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
          // Use Chatterbox for cloned voices, OpenAI for standard, null for browser fallback
          if (selectedVoice.startsWith("chatterbox:")) {
            const cbVoice = selectedVoice.replace("chatterbox:", "");
            audioData = await chatterboxTTS(sentence, cbVoice);
          } else {
            audioData = await textToSpeech(sentence, selectedVoice);
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
        } else {
          audioData = await textToSpeech(sentenceBuffer.trim(), selectedVoice);
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
    logTranscript(sessionId, transcript, fullResponse, { stt: sttTime, ai: aiTime });

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
async function getAIResponse(userMessage, history) {
  const ragContext = await queryCustomerKnowledge(userMessage);
  const systemPrompt = getSystemPrompt() + ragContext;

  const response = await groq.chat.completions.create({
    model: CHAT_MODEL,
    messages: [
      { role: "system", content: systemPrompt },
      ...history,
      { role: "user", content: userMessage },
    ],
    max_tokens: 150,
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
  try {
    const response = await fetch(`${CHATTERBOX_URL}/voices`, { signal: AbortSignal.timeout(3000) });
    const data = await response.json();
    res.json({ ...data, chatterbox: true });
  } catch {
    res.json({ voices: ["browser-uk-female", "browser-uk-male"], chatterbox: false });
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

// ============== Session Cleanup ==============
setInterval(() => {
  const now = Date.now();
  for (const [id, session] of sessions.entries()) {
    if (session.lastAccess && now - session.lastAccess > 30 * 60 * 1000) {
      console.log(`[meridian-voice] Clearing stale session: ${id}`);
      sessions.delete(id);
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
