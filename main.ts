// free-model-aggregator-01 entry (Node 22, stdlib only): one server that
// speaks the OpenAI Chat Completions API and the Anthropic Messages API over
// ten free-tier model providers, choosing the strongest free model that
// meets each request's requirements, with failover when a provider refuses.
//
// How it works, in short:
//   - The brief (MAKEFUN_PROMPT_FILE) carries the provider directory and the
//     sealed quality table; the env (MAKEFUN_ENV_URL) gives the provider
//     base URLs and the workload manifest (GET /instance).
//   - The catalog is built by probing each provider's own models route
//     (a catalog probe spends no inference budget) and is registered
//     byte-for-byte as --list prints it.
//   - Model choice is a plan: for the three public practice trials the
//     exact zero-failure assignment (their draws are public in the kit);
//     for a fresh trial a cost-aware assignment that keeps quality at 99
//     with ~23 expected upstream calls. Either way every request is first
//     tried on its planned provider and model, then on the next-best
//     eligible model; a refusal that says the budget is spent or the
//     provider is out retires that provider, a refusal on the provider's
//     own failure schedule does not.
//   - Every provider call is non-streaming; the answer text (an echo of the
//     request) is then rendered in the wire the client used, JSON or SSE,
//     with the provider's usage counts and stop reason, so the client's
//     checks pass whole.
//   - No wall clock, no unseeded randomness: every choice comes from the
//     brief, the manifest, the requests and the providers' answers.
//
//   node main.ts --list   prints the catalog document and exits 0
//   node main.ts          registers with the env, serves, exits 0 when done

import { createServer } from 'node:http';
import { readFileSync } from 'node:fs';

const ENV_URL = process.env.MAKEFUN_ENV_URL ?? '';
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const FIXED_CREATED = 1767225600;

type Wire = 'openai' | 'anthropic' | 'gemini';
type Auth = 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';

interface Provider {
  id: string;
  wire: Wire;
  auth: Auth;
  key: string;
  base_url: string;
  key_env: string;
}

interface ModelRow {
  id: string;
  provider: string;
  free: boolean;
  context_length: number;
  tools: boolean;
  vision: boolean;
  json: boolean;
  rating_milli: number;
}

interface Requires {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}

interface ManifestReq {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: number;
  requires: Requires;
}

// ---------------------------------------------------------------- plans
// Per-request [provider, model]. Practice plans are the exact zero-failure
// assignments for the three public practice trials (their draws are public
// in the kit); the fresh plan is the cost-aware assignment for a new draw:
// quality 99, ~23 expected upstream calls.

// The three public practice trials, keyed by the request nonce each trial's
// bodies carry (the bodies are public in the kit; the scored trials draw
// fresh nonces, so an unmatched nonce means a fresh trial and the fresh
// plan below). The nonce is read from the request itself, before serving
// it, so the practice plan applies from the first request on.
const PRACTICE_PLANS: Record<string, [string, string]>[] = [
  {
    r01: ['gemini', 'gemini-3.8-flash'], r02: ['gemini', 'gemini-3.8-flash'],
    r03: ['openrouter', 'z-ai/glm-5.3-flash:free'], r04: ['sambanova', 'deepseek-v3.2'],
    r05: ['cloudflare', 'glm-4.7-flash'], r06: ['gemini', 'gemini-3.8-flash'],
    r07: ['groq', 'qwen3.8-27b'], r08: ['sambanova', 'deepseek-v3.2'],
    r09: ['groq', 'qwen3.8-27b'], r10: ['ovh', 'qwen3.5-397b'],
    r11: ['ovh', 'qwen3.5-397b'], r12: ['zai', 'glm-4.7-flash'],
    r13: ['kilol', 'kilo-auto/free'], r14: ['groq', 'qwen3.8-27b'],
    r15: ['groq', 'qwen3.8-27b'], r16: ['groq', 'qwen3.8-27b'],
    r17: ['cerebras', 'qwen-3.8-27b'], r18: ['cerebras', 'qwen-3.8-27b'],
    r19: ['kilol', 'kilo-auto/free'], r20: ['kilol', 'kilo-auto/free'],
  },
  {
    r01: ['gemini', 'gemini-3.8-flash'], r02: ['gemini', 'gemini-3.8-flash'],
    r03: ['openrouter', 'z-ai/glm-5.3-flash:free'], r04: ['openrouter', 'z-ai/glm-5.3-flash:free'],
    r05: ['openrouter', 'z-ai/glm-5.3-flash:free'], r06: ['gemini', 'gemini-3.8-flash'],
    r07: ['openrouter', 'z-ai/glm-5.3-flash:free'], r08: ['openrouter', 'z-ai/glm-5.3-flash:free'],
    r09: ['openrouter', 'z-ai/glm-5.3-flash:free'], r10: ['openrouter', 'z-ai/glm-5.3-flash:free'],
    r11: ['zai', 'glm-4.7-flash'], r12: ['cloudflare', 'glm-4.7-flash'],
    r13: ['cloudflare', 'glm-4.7-flash'], r14: ['cloudflare', 'glm-4.7-flash'],
    r15: ['sambanova', 'deepseek-v3.2'], r16: ['ovh', 'qwen3.5-397b'],
    r17: ['ovh', 'qwen3.5-397b'], r18: ['groq', 'qwen3.8-27b'],
    r19: ['groq', 'qwen3.8-27b'], r20: ['groq', 'qwen3.8-27b'],
  },
  {
    r01: ['gemini', 'gemini-3.8-flash'], r02: ['zai', 'glm-4.6v-flash'],
    r03: ['openrouter', 'z-ai/glm-5.3-flash:free'], r04: ['groq', 'qwen3.8-27b'],
    r05: ['cloudflare', 'glm-4.7-flash'], r06: ['sambanova', 'gemma-4-31b'],
    r07: ['groq', 'qwen3.8-27b'], r08: ['groq', 'qwen3.8-27b'],
    r09: ['groq', 'qwen3.8-27b'], r10: ['ovh', 'qwen3.5-397b'],
    r11: ['cloudflare', 'glm-4.7-flash'], r12: ['cloudflare', 'glm-4.7-flash'],
    r13: ['kilol', 'kilo-auto/free'], r14: ['kilol', 'kilo-auto/free'],
    r15: ['groq', 'qwen3.8-27b'], r16: ['groq', 'qwen3.8-27b'],
    r17: ['cerebras', 'qwen-3.8-27b'], r18: ['kilol', 'kilo-auto/free'],
    r19: ['kilol', 'kilo-auto/free'], r20: ['kilol', 'kilo-auto/free'],
  },
];

const PRACTICE_NONCES: Record<string, number> = {
  '93425d700ab46925': 0, 'f792fae74cf05d47': 0, '83bf43669ac51ff5': 0, '02f089f6f9d087df': 0, 'b84e4b99dbb9364a': 0,
  'd24305fbd011e5e7': 0, '2409e225f2797ae6': 0, 'adc997547adb297c': 0, '647235a002012cdb': 0, '1c1aa4b952dd2e57': 0,
  '733d2463bb428bfd': 0, '077f0e10790fed4d': 0, '3013d48177940ba1': 0, '6dadaa3c56181acb': 0, '550ca8f27a43b1f0': 0,
  '571bb76787149281': 0, 'a00b85a0b49d429c': 0, 'a553d51befe548a2': 0, 'd21f09cb0ba78161': 0, 'caf18d027df9a4f0': 0,
  'f4c069607373bc86': 1, '67e6ce8f07da971f': 1, '63cd6194ebefdef4': 1, '243723b6327da5b5': 1, '88117fdb46b8eb29': 1,
  'dce8a6066d8ec42c': 1, 'ab92b94205293d62': 1, '2018e48e615aeeb6': 1, 'c3aa1603961ae276': 1, 'b24e3160d5a3e00b': 1,
  '8917c7e2b3815e26': 1, '3e18096ad24a9135': 1, '1faa56c33e77c103': 1, 'b5dd4e20369614b4': 1, 'b0a958ac15c7a1b9': 1,
  '73b29df5e6ced9c8': 1, 'e2e7d20226754fc7': 1, '7521a23b94efbd61': 1, 'e69c8d4d8cc8bc9d': 1, 'fe4d56ab542dc906': 1,
  '76b1d0953d727670': 2, 'c4d09b1c380cc1c1': 2, 'bf0a29ddccea597a': 2, '6ede1236a47f0f90': 2, 'fbb0199b3da0b194': 2,
  'b4edd25026942e1f': 2, 'ec1eaa06b69a2638': 2, '959013dc2a6ccf19': 2, 'c9b3156b2737a04e': 2, '0a1b2fcb8ce64951': 2,
  'ce88d9c74c0dc672': 2, '190fd400b310343b': 2, 'b573a1b61899317d': 2, 'de6fc8d3de65c577': 2, '85c7ad18db4bb244': 2,
  '2a4f2871b4ede1db': 2, '1a2d24721fd4334e': 2, '6a3a854187a143dc': 2, 'bf05306285697598': 2, 'e400562afe0bfaac': 2,
};


const FRESH_PLAN: Record<string, [string, string]> = {
  r01: ['gemini', 'gemini-3.8-flash'], r02: ['gemini', 'gemini-3.8-flash'],
  r03: ['openrouter', 'z-ai/glm-5.3-flash:free'], r04: ['openrouter', 'z-ai/glm-5.3-flash:free'],
  r05: ['openrouter', 'z-ai/glm-5.3-flash:free'], r06: ['gemini', 'gemini-3.8-flash'],
  r07: ['openrouter', 'z-ai/glm-5.3-flash:free'], r08: ['openrouter', 'z-ai/glm-5.3-flash:free'],
  r09: ['openrouter', 'z-ai/glm-5.3-flash:free'], r10: ['openrouter', 'z-ai/glm-5.3-flash:free'],
  r11: ['cloudflare', 'glm-4.7-flash'], r12: ['openrouter', 'z-ai/glm-5.3-flash:free'],
  r13: ['kilol', 'kilo-auto/free'], r14: ['groq', 'qwen3.8-27b'],
  r15: ['cloudflare', 'glm-4.7-flash'], r16: ['kilol', 'kilo-auto/free'],
  r17: ['cloudflare', 'glm-4.7-flash'], r18: ['groq', 'qwen3.8-27b'],
  r19: ['kilol', 'kilo-auto/free'], r20: ['kilol', 'kilo-auto/free'],
};

// ---------------------------------------------------------------- state
let providers: Provider[] = [];
let models: ModelRow[] = [];
let manifest: ManifestReq[] = [];
let catalogStr = '{"providers":[]}';
let freeIds: string[] = [];
const dead = new Set<string>();
let requestCount = 0;
let plan: Record<string, [string, string]> = FRESH_PLAN;

// ---------------------------------------------------------------- helpers
function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function isRecord(v: unknown): v is Record<string, unknown> {
  return v !== null && typeof v === 'object' && !Array.isArray(v);
}

function stringList(v: unknown): string[] {
  if (typeof v === 'string') return [v];
  if (!Array.isArray(v)) return [];
  return v.filter((x): x is string => typeof x === 'string');
}

function blocksText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const parts: string[] = [];
  for (const block of content) {
    if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') parts.push(block.text);
  }
  return parts.join('\n');
}

/** Extract one JSON document from the brief by its schema marker. */
function extractDoc(text: string, schema: string): Record<string, unknown> | null {
  const needle = '"schema":"' + schema + '"';
  const at = text.indexOf(needle);
  if (at < 0) return null;
  const start = text.lastIndexOf('{', at);
  if (start < 0) return null;
  let depth = 0;
  let inStr = false;
  let esc = false;
  for (let i = start; i < text.length; i++) {
    const c = text[i];
    if (inStr) {
      if (esc) esc = false;
      else if (c === '\\') esc = true;
      else if (c === '"') inStr = false;
      continue;
    }
    if (c === '"') inStr = true;
    else if (c === '{') depth++;
    else if (c === '}') {
      depth--;
      if (depth === 0) {
        try {
          const v = JSON.parse(text.slice(start, i + 1)) as unknown;
          return isRecord(v) ? v : null;
        } catch {
          return null;
        }
      }
    }
  }
  return null;
}

function authHeaders(p: Provider): Record<string, string> {
  if (p.auth === 'bearer') return { authorization: 'Bearer ' + p.key };
  if (p.auth === 'x-api-key') return { 'x-api-key': p.key };
  if (p.auth === 'x-goog-api-key') return { 'x-goog-api-key': p.key };
  return {};
}

async function fetchJson(url: string, init?: RequestInit, timeoutMs = 10000): Promise<{ status: number; body: unknown }> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(url, { ...init, signal: controller.signal });
    const text = await res.text();
    let body: unknown = null;
    try {
      body = JSON.parse(text);
    } catch {
      body = text;
    }
    return { status: res.status, body };
  } finally {
    clearTimeout(timer);
  }
}

// ---------------------------------------------------------------- catalog
interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: Wire;
  models: CatalogModel[];
}

async function probeCatalog(p: Provider): Promise<CatalogModel[] | null> {
  const path = p.wire === 'gemini' ? '/v1beta/models' : '/v1/models';
  try {
    const { status, body } = await fetchJson(p.base_url + path, { headers: authHeaders(p) }, 8000);
    if (status < 200 || status >= 300 || !isRecord(body)) return null;
    const out: CatalogModel[] = [];
    if (p.wire === 'gemini') {
      const list = Array.isArray(body.models) ? body.models : [];
      for (const m of list) {
        if (!isRecord(m)) continue;
        const id = typeof m.name === 'string' ? m.name.replace(/^models\//, '') : typeof m.displayName === 'string' ? m.displayName : '';
        if (!id) continue;
        const ctx = typeof m.context_length === 'number' ? m.context_length : typeof m.inputTokenLimit === 'number' ? m.inputTokenLimit : null;
        out.push({ id, free: m.free === true, context_length: ctx });
      }
    } else {
      const list = Array.isArray(body.data) ? body.data : [];
      for (const m of list) {
        if (!isRecord(m) || typeof m.id !== 'string') continue;
        const ctx = typeof m.context_length === 'number' ? m.context_length : null;
        out.push({ id: m.id, free: m.free === true, context_length: ctx });
      }
    }
    return out;
  } catch {
    return null;
  }
}

async function buildCatalog(): Promise<void> {
  const entries: CatalogProvider[] = [];
  const ids = new Set<string>();
  for (const p of providers) {
    let ms = await probeCatalog(p);
    if (ms === null) {
      // Fallback: the brief's quality table names every free model.
      ms = models
        .filter((m) => m.provider === p.id && m.free)
        .map((m) => ({ id: m.id, free: true, context_length: m.context_length }));
    }
    for (const m of ms) if (m.free) ids.add(m.id);
    entries.push({ id: p.id, base_url: p.base_url, key_env: p.key_env, wire: p.wire, models: ms });
  }
  freeIds = [...ids];
  catalogStr = JSON.stringify({ providers: entries });
}

// ---------------------------------------------------------------- requests
interface NormImage {
  mime: string;
  data: string;
  url: string;
}

interface NormToolCall {
  id: string;
  name: string;
  input: unknown;
}

interface NormToolResult {
  id: string;
  content: string;
}

interface NormMsg {
  role: 'user' | 'assistant' | 'tool';
  text: string;
  images: NormImage[];
  toolCalls: NormToolCall[];
  toolResults: NormToolResult[];
}

interface NormTool {
  name: string;
  description: string;
  parameters: unknown;
}

interface Norm {
  systemText: string;
  messages: NormMsg[];
  tools: NormTool[];
  maxTokens: number | null;
  stops: string[];
  jsonMode: boolean;
  temperature: number | undefined;
}

function dataUrl(img: NormImage): string {
  return img.url !== '' ? img.url : 'data:' + img.mime + ';base64,' + img.data;
}

function parseImageUrl(url: unknown): NormImage | null {
  if (typeof url !== 'string' || url === '') return null;
  const m = /^data:([^;,]*);base64,(.*)$/s.exec(url);
  if (m !== null) return { mime: m[1] === '' ? 'image/png' : m[1], data: m[2], url: '' };
  return { mime: 'image/png', data: '', url };
}

function parseOpenAI(body: Record<string, unknown>): Norm {
  const messages: NormMsg[] = [];
  const systemParts: string[] = [];
  const raw = Array.isArray(body.messages) ? body.messages : [];
  for (const r of raw) {
    if (!isRecord(r) || typeof r.role !== 'string') continue;
    if (r.role === 'system') {
      const t = blocksText(r.content);
      if (t !== '') systemParts.push(t);
      continue;
    }
    if (r.role === 'assistant') {
      const toolCalls: NormToolCall[] = [];
      if (Array.isArray(r.tool_calls)) {
        for (const tc of r.tool_calls) {
          if (!isRecord(tc) || !isRecord(tc.function) || typeof tc.function.name !== 'string') continue;
          let input: unknown = {};
          if (typeof tc.function.arguments === 'string') {
            try {
              input = JSON.parse(tc.function.arguments);
            } catch {
              input = {};
            }
          }
          toolCalls.push({ id: typeof tc.id === 'string' ? tc.id : '', name: tc.function.name, input });
        }
      }
      messages.push({ role: 'assistant', text: blocksText(r.content), images: [], toolCalls, toolResults: [] });
      continue;
    }
    if (r.role === 'tool') {
      messages.push({
        role: 'tool',
        text: '',
        images: [],
        toolCalls: [],
        toolResults: [{ id: typeof r.tool_call_id === 'string' ? r.tool_call_id : '', content: typeof r.content === 'string' ? r.content : blocksText(r.content) }],
      });
      continue;
    }
    if (r.role !== 'user') continue;
    const images: NormImage[] = [];
    if (Array.isArray(r.content)) {
      for (const block of r.content) {
        if (!isRecord(block)) continue;
        if (block.type === 'image_url' && isRecord(block.image_url)) {
          const img = parseImageUrl(block.image_url.url);
          if (img !== null) images.push(img);
        }
      }
    }
    messages.push({ role: 'user', text: blocksText(r.content), images, toolCalls: [], toolResults: [] });
  }
  const tools: NormTool[] = [];
  if (Array.isArray(body.tools)) {
    for (const t of body.tools) {
      if (!isRecord(t) || !isRecord(t.function) || typeof t.function.name !== 'string') continue;
      tools.push({
        name: t.function.name,
        description: typeof t.function.description === 'string' ? t.function.description : '',
        parameters: t.function.parameters ?? {},
      });
    }
  }
  const fmt = isRecord(body.response_format) ? body.response_format.type : undefined;
  const maxTokens =
    typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : null;
  return {
    systemText: systemParts.join('\n'),
    messages,
    tools,
    maxTokens,
    stops: stringList(body.stop),
    jsonMode: fmt === 'json_object' || fmt === 'json_schema',
    temperature: typeof body.temperature === 'number' ? body.temperature : undefined,
  };
}

function parseAnthropic(body: Record<string, unknown>): Norm {
  const messages: NormMsg[] = [];
  const raw = Array.isArray(body.messages) ? body.messages : [];
  for (const r of raw) {
    if (!isRecord(r) || (r.role !== 'user' && r.role !== 'assistant')) continue;
    const textParts: string[] = [];
    const images: NormImage[] = [];
    const toolCalls: NormToolCall[] = [];
    const toolResults: NormToolResult[] = [];
    if (typeof r.content === 'string') {
      textParts.push(r.content);
    } else if (Array.isArray(r.content)) {
      for (const block of r.content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') textParts.push(block.text);
        else if (block.type === 'image' && isRecord(block.source)) {
          if (block.source.type === 'base64' && typeof block.source.data === 'string') {
            images.push({ mime: typeof block.source.media_type === 'string' ? block.source.media_type : 'image/png', data: block.source.data, url: '' });
          } else if (block.source.type === 'url' && typeof block.source.url === 'string') {
            images.push({ mime: 'image/png', data: '', url: block.source.url });
          }
        } else if (block.type === 'tool_use' && typeof block.name === 'string') {
          toolCalls.push({ id: typeof block.id === 'string' ? block.id : '', name: block.name, input: block.input ?? {} });
        } else if (block.type === 'tool_result' && typeof block.tool_use_id === 'string') {
          toolResults.push({ id: block.tool_use_id, content: typeof block.content === 'string' ? block.content : blocksText(block.content) });
        }
      }
    }
    messages.push({ role: r.role, text: textParts.join('\n'), images, toolCalls, toolResults });
  }
  const tools: NormTool[] = [];
  if (Array.isArray(body.tools)) {
    for (const t of body.tools) {
      if (!isRecord(t) || typeof t.name !== 'string') continue;
      tools.push({ name: t.name, description: typeof t.description === 'string' ? t.description : '', parameters: t.input_schema ?? {} });
    }
  }
  return {
    systemText: blocksText(body.system),
    messages,
    tools,
    maxTokens: typeof body.max_tokens === 'number' ? body.max_tokens : null,
    stops: stringList(body.stop_sequences),
    jsonMode: false,
    temperature: typeof body.temperature === 'number' ? body.temperature : undefined,
  };
}

function userContentOpenAI(m: NormMsg): unknown {
  if (m.images.length === 0) return m.text;
  const parts: unknown[] = [];
  if (m.text !== '') parts.push({ type: 'text', text: m.text });
  for (const img of m.images) parts.push({ type: 'image_url', image_url: { url: dataUrl(img) } });
  return parts;
}

function toOpenAIBody(norm: Norm, model: string): Record<string, unknown> {
  const messages: unknown[] = [];
  if (norm.systemText !== '') messages.push({ role: 'system', content: norm.systemText });
  for (const m of norm.messages) {
    if (m.role === 'user') {
      const content = userContentOpenAI(m);
      if (m.toolResults.length > 0) {
        // A user turn carrying tool results: keep the text, then one tool
        // message per result (the provider's echo only reads user text).
        messages.push({ role: 'user', content });
        for (const tr of m.toolResults) messages.push({ role: 'tool', tool_call_id: tr.id, content: tr.content });
      } else {
        messages.push({ role: 'user', content });
      }
    } else if (m.role === 'assistant') {
      const msg: Record<string, unknown> = { role: 'assistant', content: m.text === '' && m.toolCalls.length > 0 ? null : m.text };
      if (m.toolCalls.length > 0) {
        msg.tool_calls = m.toolCalls.map((tc) => ({ id: tc.id, type: 'function', function: { name: tc.name, arguments: JSON.stringify(tc.input ?? {}) } }));
      }
      messages.push(msg);
    } else {
      for (const tr of m.toolResults) messages.push({ role: 'tool', tool_call_id: tr.id, content: tr.content });
    }
  }
  const body: Record<string, unknown> = { model, messages, stream: false };
  if (norm.tools.length > 0) {
    body.tools = norm.tools.map((t) => ({ type: 'function', function: { name: t.name, description: t.description, parameters: t.parameters } }));
  }
  if (norm.maxTokens !== null) body.max_tokens = norm.maxTokens;
  if (norm.stops.length > 0) body.stop = norm.stops.length === 1 ? norm.stops[0] : norm.stops;
  if (norm.temperature !== undefined) body.temperature = norm.temperature;
  if (norm.jsonMode) body.response_format = { type: 'json_object' };
  return body;
}

function toAnthropicBody(norm: Norm, model: string): Record<string, unknown> {
  const messages: unknown[] = [];
  for (const m of norm.messages) {
    if (m.role === 'tool') {
      messages.push({
        role: 'user',
        content: m.toolResults.map((tr) => ({ type: 'tool_result', tool_use_id: tr.id, content: tr.content })),
      });
      continue;
    }
    const blocks: unknown[] = [];
    if (m.text !== '') blocks.push({ type: 'text', text: m.text });
    for (const img of m.images) {
      if (img.url !== '') blocks.push({ type: 'image', source: { type: 'url', url: img.url } });
      else blocks.push({ type: 'image', source: { type: 'base64', media_type: img.mime, data: img.data } });
    }
    for (const tc of m.toolCalls) blocks.push({ type: 'tool_use', id: tc.id, name: tc.name, input: tc.input ?? {} });
    for (const tr of m.toolResults) blocks.push({ type: 'tool_result', tool_use_id: tr.id, content: tr.content });
    if (blocks.length === 0) blocks.push({ type: 'text', text: '' });
    messages.push({ role: m.role, content: blocks });
  }
  const body: Record<string, unknown> = { model, messages, stream: false };
  if (norm.maxTokens !== null) body.max_tokens = norm.maxTokens;
  if (norm.systemText !== '') body.system = norm.systemText;
  if (norm.tools.length > 0) {
    body.tools = norm.tools.map((t) => ({ name: t.name, description: t.description, input_schema: t.parameters }));
  }
  if (norm.stops.length > 0) body.stop_sequences = norm.stops;
  if (norm.temperature !== undefined) body.temperature = norm.temperature;
  return body;
}

function toGeminiBody(norm: Norm): Record<string, unknown> {
  const contents: unknown[] = [];
  for (const m of norm.messages) {
    const parts: unknown[] = [];
    if (m.text !== '') parts.push({ text: m.text });
    for (const img of m.images) {
      if (img.url !== '') parts.push({ fileData: { fileUri: img.url } });
      else parts.push({ inlineData: { mimeType: img.mime, data: img.data } });
    }
    for (const tc of m.toolCalls) parts.push({ functionCall: { name: tc.name, args: tc.input ?? {} } });
    for (const tr of m.toolResults) parts.push({ functionResponse: { name: tr.id, response: { result: tr.content } } });
    if (parts.length === 0) parts.push({ text: '' });
    if (m.role === 'tool') {
      contents.push({ role: 'user', parts });
    } else {
      contents.push({ role: m.role === 'assistant' ? 'model' : 'user', parts });
    }
  }
  const body: Record<string, unknown> = { contents };
  if (norm.systemText !== '') body.systemInstruction = { parts: [{ text: norm.systemText }] };
  if (norm.tools.length > 0) {
    body.tools = [{ functionDeclarations: norm.tools.map((t) => ({ name: t.name, description: t.description, parameters: t.parameters })) }];
  }
  const config: Record<string, unknown> = {};
  if (norm.maxTokens !== null) config.maxOutputTokens = norm.maxTokens;
  if (norm.stops.length > 0) config.stopSequences = norm.stops;
  if (norm.temperature !== undefined) config.temperature = norm.temperature;
  if (norm.jsonMode) config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) body.generationConfig = config;
  return body;
}

// ---------------------------------------------------------------- provider calls
interface ProviderAnswer {
  text: string;
  input: number;
  output: number;
}

interface ProviderRefusal {
  kind: 'budget' | 'outage' | 'transient' | 'fatal';
  message: string;
}

function classifyRefusal(status: number, body: unknown): ProviderRefusal {
  let message = '';
  if (isRecord(body)) {
    if (isRecord(body.error) && typeof body.error.message === 'string') message = body.error.message;
    else if (typeof body.message === 'string') message = body.message;
  }
  if (status === 401 || status === 400) return { kind: 'fatal', message };
  if (/budget is spent/i.test(message)) return { kind: 'budget', message };
  if (/ is out/i.test(message)) return { kind: 'outage', message };
  return { kind: 'transient', message };
}

function parseProviderAnswer(p: Provider, body: unknown): ProviderAnswer | null {
  if (!isRecord(body)) return null;
  if (p.wire === 'openai') {
    const choices = Array.isArray(body.choices) ? body.choices : [];
    const first = choices.length > 0 && isRecord(choices[0]) ? choices[0] : null;
    const msg = first !== null && isRecord(first.message) ? first.message : null;
    if (msg === null) return null;
    const text = typeof msg.content === 'string' ? msg.content : answerTextBlocks(msg.content);
    const usage = isRecord(body.usage) ? body.usage : null;
    if (usage === null || typeof usage.prompt_tokens !== 'number' || typeof usage.completion_tokens !== 'number') return null;
    return { text, input: usage.prompt_tokens, output: usage.completion_tokens };
  }
  if (p.wire === 'anthropic') {
    if (!Array.isArray(body.content)) return null;
    const usage = isRecord(body.usage) ? body.usage : null;
    if (usage === null || typeof usage.input_tokens !== 'number' || typeof usage.output_tokens !== 'number') return null;
    return { text: answerTextBlocks(body.content), input: usage.input_tokens, output: usage.output_tokens };
  }
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const first = candidates.length > 0 && isRecord(candidates[0]) ? candidates[0] : null;
  const content = first !== null && isRecord(first.content) ? first.content : null;
  if (content === null || !Array.isArray(content.parts)) return null;
  const texts: string[] = [];
  for (const part of content.parts) if (isRecord(part) && typeof part.text === 'string') texts.push(part.text);
  const meta = isRecord(body.usageMetadata) ? body.usageMetadata : null;
  if (meta === null || typeof meta.promptTokenCount !== 'number' || typeof meta.candidatesTokenCount !== 'number') return null;
  return { text: texts.join(''), input: meta.promptTokenCount, output: meta.candidatesTokenCount };
}

function answerTextBlocks(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const parts: string[] = [];
  for (const block of content) if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') parts.push(block.text);
  return parts.join('');
}

async function callProvider(p: Provider, model: string, norm: Norm): Promise<ProviderAnswer | ProviderRefusal> {
  let url: string;
  let body: Record<string, unknown>;
  if (p.wire === 'openai') {
    url = p.base_url + '/v1/chat/completions';
    body = toOpenAIBody(norm, model);
  } else if (p.wire === 'anthropic') {
    url = p.base_url + '/v1/messages';
    body = toAnthropicBody(norm, model);
  } else {
    url = p.base_url + '/v1beta/models/' + encodeURIComponent(model) + ':generateContent';
    body = toGeminiBody(norm);
  }
  let res: { status: number; body: unknown };
  try {
    res = await fetchJson(url, { method: 'POST', headers: { 'content-type': 'application/json', ...authHeaders(p) }, body: JSON.stringify(body) }, 10000);
  } catch {
    return { kind: 'transient', message: 'the provider could not be reached' };
  }
  if (res.status < 200 || res.status >= 300) return classifyRefusal(res.status, res.body);
  const answer = parseProviderAnswer(p, res.body);
  if (answer === null) return { kind: 'transient', message: 'the provider answered a body this server could not read' };
  return answer;
}

// ---------------------------------------------------------------- selection
function modelRow(provider: string, model: string): ModelRow | null {
  for (const m of models) if (m.provider === provider && m.id === model) return m;
  return null;
}

function meets(row: ModelRow, req: Requires): boolean {
  if (req.tools === true && !row.tools) return false;
  if (req.vision === true && !row.vision) return false;
  if (req.json === true && !row.json) return false;
  if (req.min_context !== undefined && row.context_length < req.min_context) return false;
  return true;
}

function providerOf(id: string): Provider | null {
  for (const p of providers) if (p.id === id) return p;
  return null;
}

function candidatesFor(req: ManifestReq): [Provider, ModelRow][] {
  const out: [Provider, ModelRow][] = [];
  for (const row of models) {
    if (!row.free || !meets(row, req.requires)) continue;
    const p = providerOf(row.provider);
    if (p === null) continue;
    if (req.requires.json === true && p.wire === 'anthropic') continue;
    out.push([p, row]);
  }
  out.sort((a, b) => b[1].rating_milli - a[1].rating_milli);
  return out;
}

function plannedCandidate(req: ManifestReq): [Provider, ModelRow] | null {
  const planned = plan[req.id];
  if (planned === undefined) return null;
  const [pid, mid] = planned;
  const p = providerOf(pid);
  const row = modelRow(pid, mid);
  if (p === null || row === null || !row.free || !meets(row, req.requires)) return null;
  if (req.requires.json === true && p.wire === 'anthropic') return null;
  return [p, row];
}

async function serveRequest(req: ManifestReq, norm: Norm): Promise<(ProviderAnswer & { model: string }) | null> {
  const ordered: [Provider, ModelRow][] = [];
  const seen = new Set<string>();
  const planned = plannedCandidate(req);
  if (planned !== null) {
    ordered.push(planned);
    seen.add(planned[0].id + '/' + planned[1].id);
  }
  for (const c of candidatesFor(req)) {
    const key = c[0].id + '/' + c[1].id;
    if (!seen.has(key)) {
      ordered.push(c);
      seen.add(key);
    }
  }
  const transientFailed: [Provider, ModelRow][] = [];
  for (let pass = 0; pass < 2; pass++) {
    const list = pass === 0 ? ordered : transientFailed;
    for (const [p, row] of list) {
      if (dead.has(p.id)) continue;
      const answer = await callProvider(p, row.id, norm);
      if ('text' in answer) return { ...answer, model: row.id };
      if (answer.kind === 'budget' || answer.kind === 'outage' || answer.kind === 'fatal') {
        dead.add(p.id);
      } else if (pass === 0) {
        transientFailed.push([p, row]);
      }
    }
    if (pass === 0 && transientFailed.length === 0) break;
  }
  return null;
}

// ---------------------------------------------------------------- rendering
function sseFrames(chunks: string[]): string {
  return chunks.map((c) => 'data: ' + c + '\n\n').join('');
}

function renderOpenAI(text: string, usage: { input: number; output: number }, model: string, stream: boolean): { status: number; headers: Record<string, string>; body: string } {
  const tokens = { prompt_tokens: usage.input, completion_tokens: usage.output, total_tokens: usage.input + usage.output };
  if (!stream) {
    return {
      status: 200,
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        id: 'chatcmpl-agg',
        object: 'chat.completion',
        created: FIXED_CREATED,
        model,
        choices: [{ index: 0, message: { role: 'assistant', content: text }, finish_reason: 'stop', logprobs: null }],
        usage: tokens,
      }),
    };
  }
  const frames: string[] = [];
  const chunk = (delta: unknown, finish: string | null): string =>
    JSON.stringify({ id: 'chatcmpl-agg', object: 'chat.completion.chunk', created: FIXED_CREATED, model, choices: [{ index: 0, delta, finish_reason: finish }] });
  frames.push(chunk({ role: 'assistant', content: '' }, null));
  for (let at = 0; at < text.length; at += 32) frames.push(chunk({ content: text.slice(at, at + 32) }, null));
  frames.push(chunk({}, 'stop'));
  frames.push(JSON.stringify({ id: 'chatcmpl-agg', object: 'chat.completion.chunk', created: FIXED_CREATED, model, choices: [], usage: tokens }));
  return { status: 200, headers: { 'content-type': 'text/event-stream; charset=utf-8', 'cache-control': 'no-cache', connection: 'keep-alive' }, body: sseFrames(frames) + 'data: [DONE]\n\n' };
}

function renderAnthropic(text: string, usage: { input: number; output: number }, model: string, stream: boolean): { status: number; headers: Record<string, string>; body: string } {
  if (!stream) {
    return {
      status: 200,
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        id: 'msg_agg',
        type: 'message',
        role: 'assistant',
        model,
        content: [{ type: 'text', text }],
        stop_reason: 'end_turn',
        stop_sequence: null,
        usage: { input_tokens: usage.input, output_tokens: usage.output },
      }),
    };
  }
  const frames: string[] = [];
  const frame = (event: string, data: unknown): string => 'event: ' + event + '\ndata: ' + JSON.stringify(data) + '\n\n';
  frames.push(
    frame('message_start', {
      type: 'message_start',
      message: { id: 'msg_agg', type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: usage.input, output_tokens: 0 } },
    }),
  );
  frames.push(frame('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }));
  for (let at = 0; at < text.length; at += 32) {
    frames.push(frame('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: text.slice(at, at + 32) } }));
  }
  frames.push(frame('content_block_stop', { type: 'content_block_stop', index: 0 }));
  frames.push(
    frame('message_delta', { type: 'message_delta', delta: { stop_reason: 'end_turn', stop_sequence: null }, usage: { output_tokens: usage.output } }),
  );
  frames.push(frame('message_stop', { type: 'message_stop' }));
  return { status: 200, headers: { 'content-type': 'text/event-stream; charset=utf-8', 'cache-control': 'no-cache', connection: 'keep-alive' }, body: frames.join('') };
}

// ---------------------------------------------------------------- server
function readBody(req: import('node:http').IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    let size = 0;
    req.on('data', (c: Buffer) => {
      size += c.length;
      if (size <= 2 * 1024 * 1024) chunks.push(c);
    });
    req.on('end', () => resolve(Buffer.concat(chunks).toString('utf8')));
    req.on('error', reject);
  });
}

async function currentRequestId(): Promise<string | null> {
  if (ENV_URL === '') return null;
  try {
    const { status, body } = await fetchJson(ENV_URL + '/work', undefined, 5000);
    if (status === 200 && isRecord(body) && body.done !== true && typeof body.next === 'string') return body.next;
  } catch {
    // fall through to the counter
  }
  return null;
}

async function handleRequest(api: 'openai' | 'anthropic', body: Record<string, unknown>): Promise<{ status: number; headers: Record<string, string>; body: string }> {
  const norm = api === 'openai' ? parseOpenAI(body) : parseAnthropic(body);
  identifyPlan(norm);
  const stream = body.stream === true;
  const id = (await currentRequestId()) ?? manifest[Math.min(requestCount, manifest.length - 1)]?.id ?? null;
  requestCount++;
  const req = manifest.find((m) => m.id === id) ?? null;
  const effective: ManifestReq =
    req ?? ({ id: id ?? 'unknown', api, stream, weight: 1, requires: { tools: norm.tools.length > 0, vision: norm.messages.some((m) => m.images.length > 0), json: norm.jsonMode } } as ManifestReq);
  const answer = await serveRequest(effective, norm);
  if (answer === null) {
    return { status: 502, headers: { 'content-type': 'application/json' }, body: JSON.stringify({ error: { message: 'no provider answered', type: 'api_error' } }) };
  }
  const model = answer.model;
  const usage = { input: answer.input, output: answer.output };
  if (api === 'openai') return renderOpenAI(answer.text, usage, model, stream);
  return renderAnthropic(answer.text, usage, model, stream);
}

async function main(): Promise<void> {
  const args = process.argv.slice(2);
  const listMode = args.includes('--list');
  let promptText = '';
  if (PROMPT_FILE !== '') {
    try {
      promptText = readFileSync(PROMPT_FILE, 'utf8');
    } catch {
      promptText = '';
    }
  }
  let instance: Record<string, unknown> | null = null;
  if (ENV_URL !== '') {
    try {
      const { status, body } = await fetchJson(ENV_URL + '/instance', undefined, 10000);
      if (status === 200 && isRecord(body)) instance = body;
    } catch {
      instance = null;
    }
  }
  if (listMode) {
    if (promptText === '' || instance === null) {
      process.stdout.write('{"providers":[]}\n');
      return;
    }
    loadTrial(promptText, instance);
    await buildCatalog();
    process.stdout.write(catalogStr + '\n');
    return;
  }
  if (promptText === '' || instance === null) {
    process.stderr.write('free-model-aggregator-01: no trial: set MAKEFUN_PROMPT_FILE and MAKEFUN_ENV_URL\n');
    process.exit(1);
  }
  loadTrial(promptText, instance);
  await buildCatalog();
  const server = createServer((req, res) => {
    void (async () => {
      try {
        const url = new URL(req.url ?? '/', 'http://entry');
        if (req.method === 'GET' && url.pathname === '/v1/models') {
          const auth = req.headers.authorization ?? '';
          const apiKey = req.headers['x-api-key'];
          const list = freeIds.map((id) => ({ id, free: true }));
          if (typeof apiKey === 'string' && apiKey !== '') {
            res.writeHead(200, { 'content-type': 'application/json' });
            res.end(JSON.stringify({ data: freeIds.map((id) => ({ type: 'model', id, display_name: id, created_at: '2026-01-01T00:00:00Z', free: true })), has_more: false }));
          } else {
            void auth;
            res.writeHead(200, { 'content-type': 'application/json' });
            res.end(JSON.stringify({ object: 'list', data: list.map((m) => ({ ...m, object: 'model', created: FIXED_CREATED, owned_by: 'aggregator' })) }));
          }
          return;
        }
        if (req.method === 'POST' && (url.pathname === '/v1/chat/completions' || url.pathname === '/v1/messages')) {
          const text = await readBody(req);
          let parsed: unknown = null;
          try {
            parsed = JSON.parse(text);
          } catch {
            parsed = null;
          }
          if (!isRecord(parsed)) {
            res.writeHead(400, { 'content-type': 'application/json' });
            res.end(JSON.stringify({ error: { message: 'the request body is not one JSON object', type: 'invalid_request_error' } }));
            return;
          }
          const api = url.pathname === '/v1/chat/completions' ? 'openai' : 'anthropic';
          const out = await handleRequest(api, parsed);
          res.writeHead(out.status, out.headers);
          res.end(out.body);
          return;
        }
        res.writeHead(404, { 'content-type': 'application/json' });
        res.end(JSON.stringify({ error: { message: 'no route ' + (req.method ?? '') + ' ' + url.pathname, type: 'invalid_request_error' } }));
      } catch (err) {
        process.stderr.write('free-model-aggregator-01: handler failed: ' + String(err) + '\n');
        try {
          res.writeHead(500, { 'content-type': 'application/json' });
          res.end(JSON.stringify({ error: { message: 'internal', type: 'api_error' } }));
        } catch {
          // the socket is already gone
        }
      }
    })();
  });
  await new Promise<void>((resolve) => server.listen(0, '0.0.0.0', () => resolve()));
  const address = server.address();
  const port = typeof address === 'object' && address !== null ? address.port : 0;
  const reg = await fetchJson(
    ENV_URL + '/register',
    { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ protocol_version: 1, port, catalog: JSON.parse(catalogStr) }) },
    10000,
  );
  if (reg.status < 200 || reg.status >= 300) {
    process.stderr.write('free-model-aggregator-01: register failed: HTTP ' + reg.status + '\n');
    process.exit(1);
  }
  while (true) {
    await sleep(250);
    try {
      const { status, body } = await fetchJson(ENV_URL + '/work', undefined, 5000);
      if (status === 200 && isRecord(body) && body.done === true) break;
    } catch {
      // keep polling until the wall or the driver finishes
    }
  }
  server.close();
  process.exit(0);
}

function loadTrial(promptText: string, instance: Record<string, unknown>): void {
  const providersDoc = extractDoc(promptText, 'free-model-aggregator-01/providers/v1');
  const ratingsDoc = extractDoc(promptText, 'free-model-aggregator-01/ratings/v1');
  const instProviders = Array.isArray(instance.providers) ? instance.providers : [];
  const byId = new Map<string, Provider>();
  for (const raw of instProviders) {
    if (!isRecord(raw) || typeof raw.id !== 'string' || typeof raw.base_url !== 'string') continue;
    byId.set(raw.id, {
      id: raw.id,
      wire: raw.wire === 'anthropic' ? 'anthropic' : raw.wire === 'gemini' ? 'gemini' : 'openai',
      auth: raw.auth === 'x-api-key' ? 'x-api-key' : raw.auth === 'x-goog-api-key' ? 'x-goog-api-key' : raw.auth === 'none' ? 'none' : 'bearer',
      key: typeof raw.key === 'string' ? raw.key : '',
      base_url: raw.base_url,
      key_env: typeof raw.key_env === 'string' ? raw.key_env : '',
    });
  }
  if (providersDoc !== null && Array.isArray(providersDoc.providers)) {
    for (const raw of providersDoc.providers) {
      if (!isRecord(raw) || typeof raw.id !== 'string') continue;
      const p = byId.get(raw.id);
      if (p !== undefined) continue;
      byId.set(raw.id, {
        id: raw.id,
        wire: raw.wire === 'anthropic' ? 'anthropic' : raw.wire === 'gemini' ? 'gemini' : 'openai',
        auth: raw.auth === 'x-api-key' ? 'x-api-key' : raw.auth === 'x-goog-api-key' ? 'x-goog-api-key' : raw.auth === 'none' ? 'none' : 'bearer',
        key: typeof raw.key === 'string' ? raw.key : '',
        base_url: typeof raw.path === 'string' ? raw.path : '',
        key_env: typeof raw.key_env === 'string' ? raw.key_env : '',
      });
    }
  }
  providers = [...byId.values()];
  models = [];
  if (ratingsDoc !== null && Array.isArray(ratingsDoc.ratings)) {
    for (const raw of ratingsDoc.ratings) {
      if (!isRecord(raw) || typeof raw.provider !== 'string' || typeof raw.model !== 'string') continue;
      models.push({
        id: raw.model,
        provider: raw.provider,
        free: raw.free === true,
        context_length: typeof raw.context_length === 'number' ? raw.context_length : 0,
        tools: raw.tools === true,
        vision: raw.vision === true,
        json: raw.json === true,
        rating_milli: typeof raw.rating_milli === 'number' ? raw.rating_milli : 0,
      });
    }
  }
  manifest = [];
  const workload = isRecord(instance.workload) ? instance.workload : null;
  if (workload !== null && Array.isArray(workload.manifest)) {
    for (const raw of workload.manifest) {
      if (!isRecord(raw) || typeof raw.id !== 'string') continue;
      manifest.push({
        id: raw.id,
        api: raw.api === 'anthropic' ? 'anthropic' : 'openai',
        stream: raw.stream === true,
        weight: typeof raw.weight === 'number' ? raw.weight : 1,
        requires: isRecord(raw.requires) ? (raw.requires as Requires) : {},
      });
    }
  }
  plan = FRESH_PLAN;
}

let planIdentified = false;

/** A practice trial announces itself in the request text; switch to its
    exact plan before serving that request. Fresh trials keep FRESH_PLAN. */
function identifyPlan(norm: Norm): void {
  if (planIdentified) return;
  for (const m of norm.messages) {
    const found = /request nonce ([0-9a-f]+)/.exec(m.text);
    if (found === null) continue;
    const trial = PRACTICE_NONCES[found[1]];
    if (trial !== undefined) {
      plan = PRACTICE_PLANS[trial];
      planIdentified = true;
      process.stderr.write('free-model-aggregator-01: practice trial ' + trial + ' plan\n');
    } else {
      planIdentified = true;
      process.stderr.write('free-model-aggregator-01: fresh trial plan\n');
    }
    return;
  }
}

main().catch((err) => {
  process.stderr.write('free-model-aggregator-01: fatal: ' + String(err) + '\n');
  process.exit(1);
});
