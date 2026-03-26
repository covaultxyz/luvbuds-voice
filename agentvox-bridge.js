/**
 * AgentVox Bridge — writes voice demo data into agentvox.db
 * so the analytics API + dashboard pick it up automatically.
 *
 * Data flow: voice-demo/server.js → this bridge → agentvox.db ← analytics API ← dashboard
 */

import Database from "better-sqlite3";
import path from "path";
import { fileURLToPath } from "url";
import crypto from "crypto";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DB_PATH = path.join(__dirname, "..", "agentvox.db");

// Use existing workspace from the DB
const WORKSPACE_ID = "d6574870-0529-4015-aeeb-a25c202d7436";
const ORACLE_AGENT_ID = "oracle-voice-demo-001";
const ANONYMOUS_CONTACT_ID = "anonymous-voice-demo-001";

let db = null;

export function initBridge() {
  try {
    db = new Database(DB_PATH);
    db.pragma("journal_mode = WAL");
    db.pragma("busy_timeout = 5000");

    // Ensure Oracle agent exists
    const agentExists = db
      .prepare("SELECT id FROM agents WHERE id = ?")
      .get(ORACLE_AGENT_ID);

    if (!agentExists) {
      db.prepare(`
        INSERT INTO agents (id, workspace_id, name, system_prompt, voice_id, language, provider, type, greeting, max_duration_seconds, end_call_phrases, voicemail_detection, transfer_rules, status, metadata_json, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'))
      `).run(
        ORACLE_AGENT_ID,
        WORKSPACE_ID,
        "Oracle (Voice Demo)",
        "NEPQ sales consultant — see server.js getSystemPrompt()",
        "browser-tts",
        "en",
        "groq",
        "inbound",
        "Hey there, welcome to Meridian AI. What can I help you with?",
        1800, // 30 min max
        "[]",
        0, // no voicemail detection
        "[]",
        "active",
        JSON.stringify({ source: "voice-demo", model: "llama-3.3-70b-versatile" })
      );
      console.log("[agentvox-bridge] Created Oracle agent in agentvox.db");
    }

    // Ensure anonymous contact exists (for callers who haven't given info)
    const contactExists = db
      .prepare("SELECT id FROM contacts WHERE id = ?")
      .get(ANONYMOUS_CONTACT_ID);

    if (!contactExists) {
      db.prepare(`
        INSERT INTO contacts (id, workspace_id, name, phone, tags, metadata_json, call_count, created_at, updated_at, stage)
        VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'), ?)
      `).run(
        ANONYMOUS_CONTACT_ID,
        WORKSPACE_ID,
        "Voice Demo Caller",
        "web",
        "[]",
        "{}",
        0,
        "lead"
      );
      console.log("[agentvox-bridge] Created anonymous contact in agentvox.db");
    }

    console.log(`[agentvox-bridge] Connected to ${DB_PATH}`);
    return true;
  } catch (error) {
    console.warn(`[agentvox-bridge] Failed to connect: ${error.message}`);
    console.warn("[agentvox-bridge] Analytics integration disabled — voice demo still works standalone");
    db = null;
    return false;
  }
}

// Track per-session state for cost accumulation
const sessionState = new Map();

/**
 * Record an exchange (user spoke + Oracle responded) into agentvox.db
 */
export function recordExchange(sessionId, userMessage, aiResponse, timing) {
  if (!db) return;

  try {
    const now = new Date().toISOString();

    // Get or create session state
    if (!sessionState.has(sessionId)) {
      sessionState.set(sessionId, {
        startedAt: now,
        turns: [],
        sttCost: 0,
        llmCost: 0,
        ttsCost: 0,
        exchangeCount: 0,
      });
    }
    const state = sessionState.get(sessionId);

    // Accumulate costs (per exchange estimates)
    // STT: ~$0.00185/min, avg utterance ~8 sec = $0.000247
    const sttCostEst = 0.000247;
    // LLM: ~1700 input tokens × $0.59/M + ~80 output tokens × $0.79/M
    const llmCostEst = (1700 * 0.59 + 80 * 0.79) / 1_000_000;
    // TTS: browser = free
    const ttsCostEst = 0;

    state.sttCost += sttCostEst;
    state.llmCost += llmCostEst;
    state.ttsCost += ttsCostEst;
    state.exchangeCount += 1;

    // Add to turns
    state.turns.push(
      { role: "user", content: userMessage, timestamp: now },
      { role: "assistant", content: aiResponse, timestamp: now }
    );

    // Calculate duration from first exchange to now
    const startMs = new Date(state.startedAt).getTime();
    const durationSec = Math.round((Date.now() - startMs) / 1000);
    const totalCost = state.sttCost + state.llmCost + state.ttsCost;

    // Build full transcript text
    const fullText = state.turns
      .map((t) => `${t.role === "user" ? "Caller" : "Oracle"}: ${t.content}`)
      .join("\n");

    // Upsert call record
    db.prepare(`
      INSERT INTO calls (id, workspace_id, agent_id, contact_id, contact_phone, direction, duration, outcome, started_at, ended_at, stt_cost, llm_cost, tts_cost, twilio_cost, total_cost)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      ON CONFLICT(id) DO UPDATE SET
        duration = excluded.duration,
        ended_at = excluded.ended_at,
        stt_cost = excluded.stt_cost,
        llm_cost = excluded.llm_cost,
        tts_cost = excluded.tts_cost,
        total_cost = excluded.total_cost
    `).run(
      sessionId,
      WORKSPACE_ID,
      ORACLE_AGENT_ID,
      ANONYMOUS_CONTACT_ID,
      "web",
      "inbound",
      durationSec,
      "completed",
      state.startedAt,
      now,
      state.sttCost,
      state.llmCost,
      state.ttsCost,
      0, // no twilio cost
      totalCost
    );

    // Upsert transcript
    const transcriptId = `transcript-${sessionId}`;
    db.prepare(`
      INSERT INTO transcripts (id, call_id, text, turns, extraction_json)
      VALUES (?, ?, ?, ?, ?)
      ON CONFLICT(call_id) DO UPDATE SET
        text = excluded.text,
        turns = excluded.turns
    `).run(
      transcriptId,
      sessionId,
      fullText,
      JSON.stringify(state.turns),
      "{}"
    );

    // Update contact call count
    db.prepare(`
      UPDATE contacts SET call_count = call_count + 1, last_activity = ?, updated_at = ?
      WHERE id = ? AND last_activity IS NULL OR last_activity < ?
    `).run(now, now, ANONYMOUS_CONTACT_ID, now);

  } catch (error) {
    console.warn(`[agentvox-bridge] Write failed: ${error.message}`);
  }
}

/**
 * Record lead capture (name/email/phone from conversation)
 */
export function recordLead(sessionId, leadInfo) {
  if (!db) return null;

  try {
    const now = new Date().toISOString();
    const contactId = crypto.randomUUID();

    db.prepare(`
      INSERT INTO contacts (id, workspace_id, name, phone, email, company, tags, metadata_json, call_count, created_at, updated_at, stage)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `).run(
      contactId,
      WORKSPACE_ID,
      leadInfo.name || "Unknown",
      leadInfo.phone || "web",
      leadInfo.email || null,
      leadInfo.company || null,
      JSON.stringify(["voice-demo", "oracle"]),
      JSON.stringify({ sessionId, source: "voice-demo" }),
      1,
      now,
      now,
      "lead"
    );

    // Update the call record to point to the real contact
    db.prepare(`
      UPDATE calls SET contact_id = ? WHERE id = ?
    `).run(contactId, sessionId);

    console.log(`[agentvox-bridge] Lead saved: ${leadInfo.name || "anonymous"} → ${contactId}`);
    return contactId;
  } catch (error) {
    console.warn(`[agentvox-bridge] Lead save failed: ${error.message}`);
    return null;
  }
}

/**
 * Clean up stale session state (called periodically)
 */
export function cleanupSessions(maxAgeMs = 30 * 60 * 1000) {
  const now = Date.now();
  for (const [id, state] of sessionState.entries()) {
    const startMs = new Date(state.startedAt).getTime();
    if (now - startMs > maxAgeMs) {
      sessionState.delete(id);
    }
  }
}
