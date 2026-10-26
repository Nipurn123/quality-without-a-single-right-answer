import { mkdir, appendFile } from "node:fs/promises"
import { parseArgs } from "node:util"
import { deepseek, gemini, geminiSpend, json, mulberry32, opus, opusSpend, pool } from "./llm.ts"

// A test of the unified quality equation (fitted in analyze.ts) on 100xprompt's own material, with
// Claude Opus 5.5 standing in for every human role:
//
//   U_u(y) = Σ_k w_k c_k(y) + θ̄ᵀφ(y) + (θ_u − θ̄)ᵀφ(y) + ε
//
// Thirty tasks in three kinds, all grounded in the product docs: docs questions (objective),
// replies to support tickets (mixed: facts and tone) and launch posts (subjective). Each task gets
// four answers: Opus 5.5 (A), DeepSeek V4.1 Flash (B), A with one factual error edited in (C), and B
// padded with filler that adds nothing (D). Every answer is fact-checked against the docs (the c_k
// instrument), then every pair is judged blind, in random order, by seven raters: an impartial judge
// and six personas of the company's own staff who would approve such text (the population D whose
// taste θ_u is measured). Ten percent of judgments are repeated to measure each rater's noise ε.
// DeepSeek also judges A against B, to measure the judge's preference for its own model.
//
// Every stage is cached under data/, so a rerun resumes. Opus runs through the Claude Code CLI
// (the user's login, no tools, no settings, no MCP servers); DeepSeek through the OpenRouter relay
// (experiments/claude-code-ab/or-proxy.ts), which logs its cost.
//
// With --raters gemini, Gemini 3.8 Flash plays all seven raters on the same answers and fact checks
// (into data/judgments-gemini.jsonl), so the study can be checked for an Opus-specific artefact: a
// judge family rating its own answers, or personas that only one model family portrays this way.
// Gemini wrote none of the answers.
//
//   bun run experiments/subjectivity/run.ts [--concurrency 8] [--raters opus|gemini]

const { values } = parseArgs({ args: Bun.argv.slice(2), options: { concurrency: { type: "string", default: "8" }, relay: { type: "string", default: "http://127.0.0.1:8797" }, raters: { type: "string", default: "opus" } } })
const HERE = new URL("./", import.meta.url).pathname
const DATA = `${HERE}data/`
const ORG = new URL("../../orgs/100xprompt/", import.meta.url).pathname
await mkdir(DATA, { recursive: true })

const N = Number(values.concurrency)

async function cached<T>(name: string, make: () => Promise<T>): Promise<T> {
  const f = Bun.file(`${DATA}${name}`)
  if (await f.exists()) return (await f.json()) as T
  const v = await make()
  await Bun.write(f, JSON.stringify(v, null, 1))
  return v
}

const rng = mulberry32(20261025)
const shuffle = <T,>(xs: T[]) => {
  const a = [...xs]
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1))
    ;[a[i], a[j]] = [a[j]!, a[i]!]
  }
  return a
}

// ── 1. Tasks ──

interface Chunk { id: string; page: string; title?: string; heading?: string; text: string }
interface EvalTask { id: string; type: string; question: string; reference: string; sources: string[]; grounded: boolean }
export interface Task { id: string; kind: "docs-qa" | "support-reply" | "launch-post"; request: string; sources: string }

const chunks = (await Bun.file(`${ORG}chunks.json`).json()) as Chunk[]
const byId = new Map(chunks.map((c) => [c.id, c]))
const evalTasks = ((await Bun.file(`${ORG}eval.json`).json()) as { tasks: EvalTask[] }).tasks
const sourceText = (ids: string[]) => ids.map((id) => byId.get(id)).filter(Boolean).map((c) => `[${c!.id}]\n${c!.text}`).join("\n\n")

const tasks = await cached<Task[]>("tasks.json", async () => {
  const pick = (type: string, n: number) => shuffle(evalTasks.filter((t) => t.type === type && t.grounded && t.sources.length)).slice(0, n)
  const pages = shuffle([...new Set(chunks.map((c) => c.page))].filter((p) => !p.startsWith("index") && !p.startsWith("resources"))).slice(0, 10)
  return [
    ...[...pick("lookup", 5), ...pick("how-to", 5)].map((t, i) => ({
      id: `qa${i}`,
      kind: "docs-qa" as const,
      request: `A user of 100xprompt (a terminal AI coding agent) asks: "${t.question}"\nAnswer the question.`,
      sources: sourceText(t.sources),
    })),
    ...pick("troubleshoot", 10).map((t, i) => ({
      id: `sr${i}`,
      kind: "support-reply" as const,
      request: `You work on 100xprompt's support team. A customer wrote in:\n\n"${t.question}"\n\nWrite the reply email to this customer.`,
      sources: sourceText(t.sources),
    })),
    ...pages.map((page, i) => ({
      id: `lp${i}`,
      kind: "launch-post" as const,
      request: `Write a short LinkedIn post (under 150 words) from the 100xprompt team announcing this capability to developers: "${byId.get(`${page}#0`)?.title ?? page}".`,
      sources: sourceText(chunks.filter((c) => c.page === page).slice(0, 3).map((c) => c.id)),
    })),
  ]
})
console.log(`tasks: ${tasks.length} (${["docs-qa", "support-reply", "launch-post"].map((k) => `${k} ${tasks.filter((t) => t.kind === k).length}`).join(", ")})`)

// ── 2. Answers: A Opus, B DeepSeek, C = A with one error, D = B padded ──

export type Variant = "A" | "B" | "C" | "D"
export interface Answer { task: string; variant: Variant; text: string }
const answerPrompt = (t: Task) => `${t.request}\n\nUse only the facts in these passages from 100xprompt's documentation:\n\n${t.sources}\n\nReply with the answer text only.`

const base = await cached<Answer[]>("answers-base.json", async () => {
  const a = await pool(tasks, N, async (t) => ({ task: t.id, variant: "A" as const, text: (await opus(answerPrompt(t))).trim() }))
  const b = await pool(tasks, N, async (t) => ({ task: t.id, variant: "B" as const, text: await deepseek(answerPrompt(t), `answer:${t.id}`, values.relay) }))
  return [...a, ...b]
})
const edits = await cached<Answer[]>("answers-edited.json", async () => {
  const of = (task: string, v: Variant) => base.find((x) => x.task === task && x.variant === v)!.text
  const c = await pool(tasks, N, async (t) => ({
    task: t.id,
    variant: "C" as const,
    text: (await opus(`Here is a text and the documentation it is based on.\n\nDOCUMENTATION:\n${t.sources}\n\nTEXT:\n${of(t.id, "A")}\n\nEdit the text to introduce exactly ONE factual error that contradicts the documentation: change one specific detail (a key, command, flag, number, name or behaviour) to something plausible but wrong. Change nothing else at all: same wording, length and format everywhere else. Reply with the edited text only.`)).trim(),
  }))
  const d = await pool(tasks, N, async (t) => ({
    task: t.id,
    variant: "D" as const,
    text: (await opus(`TEXT:\n${of(t.id, "B")}\n\nAdd two or three sentences of generic filler to this text (restating what is already there, generic encouragement or pleasantries) that add NO new information. Keep every existing sentence exactly as it is. Reply with the padded text only.`)).trim(),
  }))
  return [...c, ...d]
})
const answers = [...base, ...edits]
const answerOf = (task: string, v: Variant) => answers.find((x) => x.task === task && x.variant === v)!.text
console.log(`answers: ${answers.length}`)

// ── 3. Fact checks (the c_k instrument) ──

export interface FactCheck { task: string; variant: Variant; claims: Array<{ claim: string; verdict: "supported" | "contradicted" | "unsupported" }> }
const factChecks = await cached<FactCheck[]>("factchecks.json", async () => {
  const out: FactCheck[] = []
  await pool(tasks, N, async (t) => {
    const order = shuffle(["A", "B", "C", "D"] as Variant[])
    const reply = await opus(
      `You are a fact checker. DOCUMENTATION (the only ground truth):\n${t.sources}\n\n` +
        order.map((v, i) => `TEXT ${i + 1}:\n${answerOf(t.id, v)}`).join("\n\n") +
        `\n\nFor each text, list every specific factual claim it makes about 100xprompt (commands, keys, flags, behaviours, numbers, names; not opinions or pleasantries), and judge each against the documentation: "supported", "contradicted" (the documentation says otherwise) or "unsupported" (the documentation does not say). Reply with JSON only: [{"text": 1, "claims": [{"claim": "...", "verdict": "supported"}]}, ...]`,
    )
    const parsed = json<Array<{ text: number; claims: FactCheck["claims"] }>>(reply)
    for (const p of parsed) out.push({ task: t.id, variant: order[p.text - 1]!, claims: p.claims ?? [] })
  })
  return out
})
console.log(`fact checks: ${factChecks.length}`)

// ── 4. Pairwise judgments ──

export const PERSONAS: Record<string, string> = {
  judge: "You are an impartial expert reviewer. Judge which response better serves the request: correct against the documentation, helpful, clear and appropriate for its purpose. Do not let length or formatting sway you unless they genuinely help the reader.",
  priya: "You are Priya, a staff backend engineer at 100xprompt. You value precision and brevity above all: exact commands and keys, no fluff, no marketing tone, no exclamation marks. Padding and pleasantries annoy you.",
  marco: "You are Marco, 100xprompt's developer advocate. You love friendly, energetic writing with a concrete example, a bit of personality and a clear call to action. Dry, terse text feels cold to you.",
  aisha: "You are Aisha, 100xprompt's support team lead. You care that the reader feels heard: acknowledge the problem, then clear numbered steps and a next step if it does not work. Curt answers feel dismissive to you.",
  tom: "You are Tom, 100xprompt's security and compliance lead. You distrust overclaiming and hype. You want careful, formal wording, caveats where they belong and nothing promised that the documentation does not promise.",
  lena: "You are Lena, 100xprompt's product marketing manager. You want punchy, benefit-led, scannable text with a strong hook in the first line. Long paragraphs and caveats lose readers, in your view.",
  kenji: "You are Kenji, an engineering manager at 100xprompt whose first language is Japanese. You want plain English: short sentences, the bottom line first, no idioms or slang.",
}
const RATERS = Object.keys(PERSONAS)
const PAIRS: Array<[Variant, Variant]> = [["A", "B"], ["A", "C"], ["A", "D"], ["B", "C"], ["B", "D"], ["C", "D"]]

export interface Judgment { rater: string; task: string; first: Variant; second: Variant; preferred: "first" | "second" | "tie"; useAsIs: [boolean, boolean]; repeat: boolean; judge: "opus" | "deepseek" | "gemini" }
type Item = { task: Task; pair: [Variant, Variant] }

const judgePrompt = (rater: string, items: Array<{ task: Task; first: Variant; second: Variant }>) =>
  `${PERSONAS[rater]}\n\nYou will review ${items.length} unrelated items. Each has a request, the documentation it must rely on, and two responses written for it. For each item, say which response you prefer (or "tie" only if you truly cannot choose), and whether you would approve each response for use exactly as it is.\n\n` +
  items
    .map((it, i) => `=== ITEM ${i + 1} ===\nREQUEST:\n${it.task.request}\n\nDOCUMENTATION:\n${it.task.sources}\n\nRESPONSE 1:\n${answerOf(it.task.id, it.first)}\n\nRESPONSE 2:\n${answerOf(it.task.id, it.second)}`)
    .join("\n\n") +
  `\n\nReply with JSON only: [{"item": 1, "preferred": 1 | 2 | "tie", "use_as_is": [true|false, true|false]}, ...] with one entry per item.`

/** Six items per call, each from a different task, so a call can never force consistency within a task. */
function batches(): Item[][] {
  const out: Item[][] = []
  for (let j = 0; j < tasks.length; j++) out.push(PAIRS.map((pair, i) => ({ task: tasks[(j + 5 * i) % tasks.length]!, pair })))
  return out
}

const FAMILY = values.raters === "gemini" ? "gemini" : "opus"
const JUDGMENTS = FAMILY === "opus" ? `${DATA}judgments.jsonl` : `${DATA}judgments-gemini.jsonl`
const done = new Set<string>()
if (await Bun.file(JUDGMENTS).exists()) for (const l of (await Bun.file(JUDGMENTS).text()).trim().split("\n").filter(Boolean)) {
  const j = JSON.parse(l) as Judgment & { batch: string }
  done.add(j.batch)
}

const work: Array<{ key: string; rater: string; judge: Judgment["judge"]; items: Item[]; repeat: boolean }> = []
for (const rater of RATERS) {
  const bs = batches()
  bs.forEach((items, j) => work.push({ key: `${rater}|${j}`, rater, judge: FAMILY, items, repeat: false }))
  // Repeats: 18 earlier items (10%), from distinct tasks, in fresh calls with fresh order and sides.
  const again = shuffle(bs.flat()).filter((it, i, xs) => xs.findIndex((y) => y.task.id === it.task.id) === i).slice(0, 18)
  for (let r = 0; r < 3; r++) work.push({ key: `${rater}|repeat${r}`, rater, judge: FAMILY, items: again.slice(r * 6, r * 6 + 6), repeat: true })
}
// DeepSeek as the impartial judge on A vs B, for self-preference.
if (FAMILY === "opus") for (let j = 0; j < 5; j++) work.push({ key: `deepseek-judge|${j}`, rater: "judge", judge: "deepseek", items: tasks.slice(j * 6, j * 6 + 6).map((task) => ({ task, pair: ["A", "B"] as [Variant, Variant] })), repeat: false })

const todo = work.filter((w) => !done.has(w.key))
console.log(`judgment batches: ${work.length} (${todo.length} to run)`)
let finished = 0
await pool(todo, N, async (w) => {
  const items = w.items.map((it) => (rng() < 0.5 ? { task: it.task, first: it.pair[0], second: it.pair[1] } : { task: it.task, first: it.pair[1], second: it.pair[0] }))
  const prompt = judgePrompt(w.rater, items)
  const reply = w.judge === "opus" ? await opus(prompt) : w.judge === "gemini" ? await gemini(prompt) : await deepseek(prompt, `judge:${w.key}`, values.relay)
  let parsed: Array<{ item: number; preferred: 1 | 2 | "tie"; use_as_is?: [boolean, boolean] }>
  try {
    parsed = json(reply)
  } catch {
    console.error(`unparseable reply for ${w.key}`)
    return
  }
  const rows = parsed
    .filter((p) => items[p.item - 1])
    .map((p) => {
      const it = items[p.item - 1]!
      return { batch: w.key, rater: w.rater, judge: w.judge, task: it.task.id, first: it.first, second: it.second, preferred: p.preferred === 1 ? "first" : p.preferred === 2 ? "second" : "tie", useAsIs: p.use_as_is ?? [false, false], repeat: w.repeat }
    })
  await appendFile(JUDGMENTS, rows.map((r) => JSON.stringify(r)).join("\n") + "\n")
  if (++finished % 20 === 0) console.log(`  ${finished}/${todo.length} batches, spend so far: Opus $${opusSpend.usd.toFixed(2)} over ${opusSpend.calls} calls, Gemini $${geminiSpend.usd.toFixed(2)} over ${geminiSpend.calls}`)
})
console.log(`done: Opus $${opusSpend.usd.toFixed(2)} over ${opusSpend.calls} calls, Gemini $${geminiSpend.usd.toFixed(2)} over ${geminiSpend.calls} this run (DeepSeek in the relay log, run ids subjectivity:*)`)
