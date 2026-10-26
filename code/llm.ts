// Model calls shared by the subjectivity experiments.
//   opus      Claude Opus 5.5 through the Claude Code CLI: the user's login, no tools, no settings,
//             no MCP servers, a clean environment, a scratch working directory.
//   deepseek  DeepSeek V4.1 Flash through the OpenRouter relay (experiments/claude-code-ab/or-proxy.ts),
//             which holds the key and logs each call's cost under its run id.
//   gemini    Gemini 3.8 Flash on Vertex AI (global endpoint, OpenAI-compatible), with the gcloud
//             login's access token; spend from Vertex's reported usage at list price.

import { mkdir } from "node:fs/promises"

const CWD = `${process.env.TMPDIR ?? "/tmp"}/subjectivity-judge/`
await mkdir(CWD, { recursive: true })

/** List-price spend and calls of the Opus calls this process made. */
export const opusSpend = { usd: 0, calls: 0 }

/** One Opus 5.5 turn; returns its text. Retried twice on a failed or empty reply. */
export async function opus(prompt: string): Promise<string> {
  for (let attempt = 0; attempt < 3; attempt++) {
    const proc = Bun.spawn(
      ["claude", "-p", prompt, "--model", "claude-opus-5-5", "--effort", "medium", "--output-format", "json", "--tools", "", "--setting-sources", "project", "--strict-mcp-config", "--no-session-persistence"],
      {
        cwd: CWD,
        stdin: "ignore",
        stdout: "pipe",
        stderr: "pipe",
        env: { HOME: process.env.HOME!, PATH: process.env.PATH!, USER: process.env.USER!, LOGNAME: process.env.USER!, TERM: "xterm-256color", LANG: "en_US.UTF-8", TMPDIR: process.env.TMPDIR ?? "/tmp", CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC: "1" },
      },
    )
    const out = await new Response(proc.stdout).text()
    await proc.exited
    try {
      const j = JSON.parse(out) as { result?: string; total_cost_usd?: number; is_error?: boolean }
      opusSpend.usd += j.total_cost_usd ?? 0
      opusSpend.calls++
      if (!j.is_error && j.result) return j.result
    } catch {}
    await Bun.sleep(2_000 * (attempt + 1))
  }
  throw new Error("opus call failed three times")
}

/** One DeepSeek V4.1 Flash call through the relay, tagged `subjectivity:<tag>` in its cost log. */
export async function deepseek(prompt: string, tag: string, relay = "http://127.0.0.1:8797"): Promise<string> {
  for (let attempt = 0; attempt < 3; attempt++) {
    const res = await fetch(`${relay}/v1/messages`, {
      method: "POST",
      headers: { "content-type": "application/json", "anthropic-version": "2023-06-01", "x-run-id": `subjectivity:${tag}` },
      body: JSON.stringify({ model: "deepseek/deepseek-v4.1-flash-1017", max_tokens: 16_000, temperature: 0.3, provider: { order: ["fireworks/spot", "inference-net/fp8"], allow_fallbacks: false }, messages: [{ role: "user", content: prompt }] }),
    }).catch(() => undefined)
    if (res?.ok) {
      const j = (await res.json()) as { content?: Array<{ type: string; text?: string }> }
      const text = (j.content ?? []).filter((b) => b.type === "text").map((b) => b.text ?? "").join("").trim()
      if (text) return text
    }
    await Bun.sleep(2_000 * (attempt + 1))
  }
  throw new Error("deepseek call failed three times")
}

const VERTEX = `https://aiplatform.googleapis.com/v1/projects/${process.env.VERTEX_PROJECT}/locations/global/endpoints/openapi`
/** List-price spend and calls of the Gemini calls this process made ($0.75 / $3.75 per million in / out). */
export const geminiSpend = { usd: 0, calls: 0 }
let token = { value: "", at: 0 }
function bearer(): string {
  if (Date.now() - token.at > 5 * 60_000) {
    const p = Bun.spawnSync(["gcloud", "auth", "print-access-token"], { stdout: "pipe", stderr: "pipe" })
    token = { value: p.stdout.toString().trim(), at: Date.now() }
  }
  return token.value
}

/**
 * One Gemini 3.8 Flash turn; returns its text. Retried through Vertex's 429s (its shared capacity is
 * often exhausted), 5xx and hangs, with jittered backoff so parallel callers do not retry in step.
 */
export async function gemini(prompt: string, effort: "low" | "medium" | "high" = "medium"): Promise<string> {
  for (let attempt = 0; attempt < 40; attempt++) {
    try {
      const res = await fetch(`${VERTEX}/chat/completions`, {
        method: "POST",
        headers: { "content-type": "application/json", authorization: `Bearer ${bearer()}` },
        body: JSON.stringify({ model: "google/gemini-3.8-flash", messages: [{ role: "user", content: prompt }], reasoning_effort: effort }),
        signal: AbortSignal.timeout(180_000),
      })
      if (res.ok) {
        const j = (await res.json()) as { choices: Array<{ message: { content?: string } }>; usage?: { prompt_tokens?: number; completion_tokens?: number; completion_tokens_details?: { reasoning_tokens?: number } } }
        // Vertex reports reasoning tokens apart from completion tokens; both are billed as output.
        const out = (j.usage?.completion_tokens ?? 0) + (j.usage?.completion_tokens_details?.reasoning_tokens ?? 0)
        geminiSpend.usd += ((j.usage?.prompt_tokens ?? 0) * 0.75 + out * 3.75) / 1e6
        geminiSpend.calls++
        const text = j.choices[0]?.message.content?.trim()
        if (text) return text
      } else if (res.status === 401) token.at = 0
      else if (res.status !== 429 && res.status < 500) throw new Error(`Vertex HTTP ${res.status}: ${(await res.text()).slice(0, 300)}`)
    } catch (e) {
      if ((e as Error).message.startsWith("Vertex HTTP")) throw e
    }
    await Bun.sleep(Math.min(60_000, 2_000 * 2 ** Math.min(attempt, 5)) * (0.5 + Math.random()))
  }
  throw new Error("gemini call failed forty times")
}

/** The first JSON object or array in a model's reply. */
export function json<T>(text: string): T {
  const start = text.search(/[[{]/)
  const end = Math.max(text.lastIndexOf("}"), text.lastIndexOf("]"))
  return JSON.parse(text.slice(start, end + 1)) as T
}

/** Run `fn` over `items` with at most `n` in flight. */
export async function pool<T, R>(items: T[], n: number, fn: (x: T, i: number) => Promise<R>): Promise<R[]> {
  const out = new Array<R>(items.length)
  let next = 0
  await Promise.all(
    Array.from({ length: Math.min(n, items.length) }, async () => {
      while (next < items.length) {
        const i = next++
        out[i] = await fn(items[i]!, i)
      }
    }),
  )
  return out
}

/** A seeded uniform generator (mulberry32). */
export const mulberry32 = (seed: number) => () => {
  seed |= 0
  seed = (seed + 0x6d2b79f5) | 0
  let t = Math.imul(seed ^ (seed >>> 15), 1 | seed)
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296
}
