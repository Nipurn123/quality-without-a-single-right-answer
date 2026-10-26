import { mkdir } from "node:fs/promises"
import { gemini, geminiSpend, json, mulberry32, opus, opusSpend, pool } from "./llm.ts"

// Checks an external review of the paper asked for, on MT-Bench's expert votes:
//
// 1. Baselines for "Opus 5.5 agrees with an expert more than another expert does". A judge that
//    tracks the consensus should beat a single expert, so the bar is set by predictors that use no
//    answer text at all: the leaderboard (the stronger model of the pair by Bradley–Terry, overall and
//    per category, fitted without the comparison's own votes) and the majority of the other experts.
//    Differences carry 95% intervals from a bootstrap over comparisons.
// 2. Contamination probes for Opus 5.5, which may have seen MT-Bench in training: (a) given the first
//    half of a question, continue it word for word; (b) given a question and the names of two models,
//    but neither answer, say which one the experts preferred. (b) at the leaderboard's level means
//    no recall of individual votes beyond what the model names give away.
// 3. A second correctness grader: Gemini 3.8 Flash re-grades the answers Opus graded against MT-Bench's
//    references (same prompt), for agreement, and for the check that experts prefer the correct answer.
//
// Model replies are cached in mtbench-out/, so a rerun costs nothing.
//
//   bun run experiments/subjectivity/checks.ts [--concurrency 8] [--no-grader]

const IN = new URL("./mtbench-data/", import.meta.url).pathname
const OUT = new URL("./mtbench-out/", import.meta.url).pathname
await mkdir(OUT, { recursive: true })
const N = Number(Bun.argv.includes("--concurrency") ? Bun.argv[Bun.argv.indexOf("--concurrency") + 1] : 8)

interface Msg { role: string; content: string }
interface Vote { judge: string; question_id: number; turn: number; model_a: string; model_b: string; winner: string; conversation_a: Msg[]; conversation_b: Msg[] }
interface Question { question_id: number; category: string; turns: string[]; reference?: string[] }

const lines = async <T,>(f: string) => (await Bun.file(f).text()).trim().split("\n").map((l) => JSON.parse(l) as T)
const human = await lines<Vote>(`${IN}human.jsonl`)
const gpt4 = await lines<Vote>(`${IN}gpt4_pair.jsonl`)
const questions = new Map((await lines<Question>(`${IN}question.jsonl`)).map((q) => [q.question_id, q]))
const CATEGORIES = ["writing", "roleplay", "humanities", "stem", "extraction", "reasoning", "math", "coding"]
const OBJECTIVE = new Set(["reasoning", "math", "coding"])
const catOf = (v: { question_id: number }) => questions.get(v.question_id)!.category
const MODELS = [...new Set(human.flatMap((v) => [v.model_a, v.model_b]))].sort()
const itemOf = (v: Vote) => `${v.question_id}/${v.turn}/${[v.model_a, v.model_b].sort().join("|")}`
/** 1 if the alphabetically first model of the pair won, 0 if the other did, 0.5 for a tie. */
const label = (v: Vote) => (v.winner === "tie" ? 0.5 : (v.winner === "model_a" ? v.model_a : v.model_b) === [v.model_a, v.model_b].sort()[0] ? 1 : 0)
const akey = (q: number, turn: number, model: string) => `${q}/${turn}/${model}`
const answers = new Map<string, string>()
for (const v of [...human, ...gpt4]) for (const [model, conv] of [[v.model_a, v.conversation_a], [v.model_b, v.conversation_b]] as const) {
  const msg = conv?.filter((m) => m.role === "assistant")[v.turn - 1]
  if (msg) answers.set(akey(v.question_id, v.turn, model), msg.content)
}
const conversation = (q: Question, turn: number) => q.turns.slice(0, turn).map((t, i) => `USER (turn ${i + 1}): ${t}`).join("\n\n")
const byItem = new Map<string, Vote[]>()
for (const v of human) byItem.set(itemOf(v), [...(byItem.get(itemOf(v)) ?? []), v])
const pct = (x: number) => `${(100 * x).toFixed(1)}%`
const rng = mulberry32(20261025)
const shuffle = <T,>(xs: T[]) => {
  const a = [...xs]
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1))
    ;[a[i], a[j]] = [a[j]!, a[i]!]
  }
  return a
}
async function cached<T>(name: string, make: () => Promise<T>): Promise<T> {
  const f = Bun.file(`${OUT}${name}`)
  if (await f.exists()) return (await f.json()) as T
  const v = await make()
  await Bun.write(f, JSON.stringify(v, null, 1))
  return v
}

// The 300 comparisons Opus judged in mtbench.ts, and its verdicts.
const opusJudge = (await Bun.file(`${OUT}opus-judge.json`).json()) as Record<string, number>
const sample = Object.keys(opusJudge).filter((k) => (byItem.get(k)?.length ?? 0) >= 2)
const gptOf = new Map(gpt4.map((v) => [itemOf(v), label(v)]))

// ── 1. Baselines ──

/** Bradley–Terry scores from win counts per ordered pair (ties as half wins), a weak ridge. */
function bt(counts: Map<string, { n: number; w: number }>): Map<string, number> {
  const s = new Map(MODELS.map((m) => [m, 0]))
  for (let it = 0; it < 2000; it++) {
    const g = new Map(MODELS.map((m) => [m, -0.01 * s.get(m)!]))
    for (const [k, c] of counts) {
      const [a, b] = k.split("|") as [string, string]
      const p = 1 / (1 + Math.exp(-(s.get(a)! - s.get(b)!)))
      g.set(a, g.get(a)! + c.w - c.n * p)
      g.set(b, g.get(b)! - (c.w - c.n * p))
    }
    for (const m of MODELS) s.set(m, s.get(m)! + 0.002 * g.get(m)!)
  }
  return s
}
/** Pair counts (alphabetical order, wins of the first) over the given votes. */
function counts(votes: Vote[]) {
  const c = new Map<string, { n: number; w: number }>()
  for (const v of votes) {
    const k = [v.model_a, v.model_b].sort().join("|")
    const e = c.get(k) ?? { n: 0, w: 0 }
    e.n++
    e.w += label(v)
    c.set(k, e)
  }
  return c
}
const leaderboard = (scope: "overall" | "category"): Record<string, number> => {
  const out: Record<string, number> = {}
  for (const item of sample) {
    const vs = byItem.get(item)!
    const rest = human.filter((v) => itemOf(v) !== item && (scope === "overall" || catOf(v) === catOf(vs[0]!)))
    const s = bt(counts(rest))
    const [a, b] = [vs[0]!.model_a, vs[0]!.model_b].sort()
    out[item] = s.get(a!)! >= s.get(b!)! ? 1 : 0
  }
  return out
}
const lbOverall = leaderboard("overall")
const lbCategory = leaderboard("category")

/** Per comparison: agreements with each expert vote, 3-way and on votes where both took a side. */
type Tally = { a3: number; n3: number; aD: number; nD: number }
function tally(item: string, verdict: number | undefined): Tally {
  const t = { a3: 0, n3: 0, aD: 0, nD: 0 }
  if (verdict === undefined) return t
  for (const l of byItem.get(item)!.map(label)) {
    t.n3++
    if (l === verdict) t.a3++
    if (l !== 0.5 && verdict !== 0.5) {
      t.nD++
      if (l === verdict) t.aD++
    }
  }
  return t
}
/** Expert against expert: every pair of votes on the comparison. */
function expertTally(item: string): Tally {
  const ls = byItem.get(item)!.map(label)
  const t = { a3: 0, n3: 0, aD: 0, nD: 0 }
  for (let i = 0; i < ls.length; i++) for (let j = i + 1; j < ls.length; j++) {
    t.n3++
    if (ls[i] === ls[j]) t.a3++
    if (ls[i] !== 0.5 && ls[j] !== 0.5) {
      t.nD++
      if (ls[i] === ls[j]) t.aD++
    }
  }
  return t
}
/** Majority of the other experts against each expert (comparisons with three or more votes; a split majority is skipped). */
function majorityTally(item: string): Tally {
  const ls = byItem.get(item)!.map(label)
  const t = { a3: 0, n3: 0, aD: 0, nD: 0 }
  if (ls.length < 3) return t
  ls.forEach((l, i) => {
    const others = ls.filter((_, j) => j !== i)
    const freq = [0, 0.5, 1].map((x) => ({ x, n: others.filter((o) => o === x).length })).sort((p, q) => q.n - p.n)
    if (freq[0]!.n === freq[1]!.n) return
    const m = freq[0]!.x
    t.n3++
    if (l === m) t.a3++
    if (l !== 0.5 && m !== 0.5) {
      t.nD++
      if (l === m) t.aD++
    }
  })
  return t
}
const sum = (ts: Tally[]) => ts.reduce((s, t) => ({ a3: s.a3 + t.a3, n3: s.n3 + t.n3, aD: s.aD + t.aD, nD: s.nD + t.nD }), { a3: 0, n3: 0, aD: 0, nD: 0 })
const fmt = (t: Tally) => `${pct(t.a3 / t.n3)} / ${pct(t.aD / t.nD)}`
/** 95% bootstrap interval (over comparisons) for the difference in 3-way and decided agreement. */
function diffCi(items: string[], f: (i: string) => Tally, g: (i: string) => Tally) {
  const b = mulberry32(7)
  const d3: number[] = []
  const dD: number[] = []
  const F = new Map(items.map((i) => [i, f(i)]))
  const G = new Map(items.map((i) => [i, g(i)]))
  for (let r = 0; r < 2000; r++) {
    const pick = items.map(() => items[Math.floor(b() * items.length)]!)
    const x = sum(pick.map((i) => F.get(i)!))
    const y = sum(pick.map((i) => G.get(i)!))
    d3.push(x.a3 / x.n3 - y.a3 / y.n3)
    dD.push(x.aD / x.nD - y.aD / y.nD)
  }
  const q = (xs: number[]) => {
    const s = [...xs].sort((p, q) => p - q)
    return `[${(100 * s[Math.floor(0.025 * s.length)]!).toFixed(1)}, ${(100 * s[Math.floor(0.975 * s.length)]!).toFixed(1)}]`
  }
  return `3-way ${q(d3)}, decided ${q(dD)}`
}

const predictors: Array<[string, (i: string) => Tally]> = [
  ["expert vs expert", expertTally],
  ["leaderboard, overall", (i) => tally(i, lbOverall[i])],
  ["leaderboard, per category", (i) => tally(i, lbCategory[i])],
  ["GPT-4 judge", (i) => tally(i, gptOf.get(i))],
  ["Opus 5.5 judge", (i) => tally(i, opusJudge[i])],
]

// ── 2. Contamination probes ──

const recall = await cached<Record<string, string>>("probe-question-recall.json", async () => {
  const qs = [...questions.values()].filter((q) => q.turns[0]!.split(/\s+/).length >= 20)
  const out: Record<string, string> = {}
  const batches: Question[][] = []
  for (let i = 0; i < qs.length; i += 10) batches.push(qs.slice(i, i + 10))
  await pool(batches, N, async (batch) => {
    const reply = await opus(
      `These are the openings of questions from the MT-Bench benchmark (LMSYS, Zheng et al., 2023). For each, if you remember the question, continue it word for word to its end; if you do not, write the most likely continuation. Reply with JSON only: [{"item": 1, "continuation": "..."}, ...]\n\n` +
        batch.map((q, i) => { const w = q.turns[0]!.split(/\s+/); return `ITEM ${i + 1} (category: ${q.category}): ${w.slice(0, Math.ceil(w.length / 2)).join(" ")} …` }).join("\n\n"),
    )
    for (const r of json<Array<{ item: number; continuation: string }>>(reply)) if (batch[r.item - 1]) out[String(batch[r.item - 1]!.question_id)] = r.continuation
  })
  return out
})

const namesOnly = await cached<Record<string, number>>("probe-verdict-recall.json", async () => {
  const out: Record<string, number> = {}
  const items = shuffle(sample)
  const batches: string[][] = []
  for (let i = 0; i < items.length; i += 15) batches.push(items.slice(i, i + 15))
  await pool(batches, N, async (batch) => {
    const shown = batch.map((item) => {
      const v = byItem.get(item)![0]!
      const [m1, m2] = [v.model_a, v.model_b].sort()
      const flip = rng() < 0.5
      return { item, v, first: flip ? m2! : m1!, second: flip ? m1! : m2! }
    })
    const reply = await opus(
      `MT-Bench (LMSYS, Zheng et al., 2023) collected expert human votes between the answers of six chat models (gpt-4, gpt-3.5-turbo, claude-v1, vicuna-13b-v1.2, alpaca-13b, llama-13b) to two-turn questions. For each item below you get the conversation up to the judged turn and the two models whose answers to its last turn were compared, but not the answers. Say which model's answer the human experts preferred, "A" or "B", or "tie". If you remember the actual votes, use them; otherwise give your best guess.\n\n` +
        shown.map((s, i) => `=== ITEM ${i + 1} (category: ${catOf(s.v)}) ===\n${conversation(questions.get(s.v.question_id)!, s.v.turn)}\n\nMODEL A: ${s.first}\nMODEL B: ${s.second}`).join("\n\n") +
        `\n\nReply with JSON only: [{"item": 1, "verdict": "A" | "B" | "tie"}, ...]`,
    )
    for (const r of json<Array<{ item: number; verdict: string }>>(reply)) {
      const s = shown[r.item - 1]
      if (!s) continue
      out[s.item] = r.verdict === "tie" ? 0.5 : (r.verdict === "A" ? s.first : s.second) === [s.v.model_a, s.v.model_b].sort()[0] ? 1 : 0
    }
  })
  return out
})
predictors.push(["Opus 5.5, model names only", (i) => tally(i, namesOnly[i])])

// ── 3. Second correctness grader ──

const opusCorrect = (await Bun.file(`${OUT}correctness.json`).json()) as Record<string, 0 | 1>
const NO_GRADER = Bun.argv.includes("--no-grader")
const geminiCorrect = NO_GRADER ? {} : await cached<Record<string, 0 | 1>>("correctness-gemini.json", async () => {
  const out: Record<string, 0 | 1> = {}
  const units = [...questions.values()].filter((q) => OBJECTIVE.has(q.category) && q.reference?.length).flatMap((q) => [1, 2].map((turn) => ({ q, turn })))
  await pool(units, N, async ({ q, turn }) => {
    const ref = q.reference![turn - 1]
    if (!ref) return
    const models = shuffle(MODELS.filter((m) => answers.has(akey(q.question_id, turn, m))))
    const reply = await gemini(
      `Grade answers to a ${q.category} question against the reference answer. Judge only whether each answer's final result is correct (for code: whether it correctly does what was asked); ignore style and length.\n\nCONVERSATION:\n${conversation(q, turn)}\n\nREFERENCE ANSWER for turn ${turn}:\n${ref}\n\n` +
        models.map((m, i) => `ANSWER ${i + 1} (to turn ${turn}):\n${answers.get(akey(q.question_id, turn, m))}`).join("\n\n") +
        `\n\nReply with JSON only: [{"answer": 1, "correct": true}, ...] with one entry per answer.`,
    )
    for (const r of json<Array<{ answer: number; correct: boolean }>>(reply)) if (models[r.answer - 1]) out[akey(q.question_id, turn, models[r.answer - 1]!)] = r.correct ? 1 : 0
  })
  return out
})

// ── Report ──

console.log(`Agreement with an expert's vote on the ${sample.length} comparisons Opus judged (3-way: A, B, tie / on votes where both took a side):`)
for (const [name, f] of predictors) console.log(`  ${name.padEnd(28)} ${fmt(sum(sample.map(f)))}`)
console.log(`  differences (percentage points, 95% bootstrap over comparisons):`)
console.log(`    Opus judge − expert vs expert:        ${diffCi(sample, predictors[4]![1], expertTally)}`)
console.log(`    Opus judge − leaderboard per category: ${diffCi(sample, predictors[4]![1], predictors[2]![1])}`)
console.log(`    Opus names only − leaderboard per category: ${diffCi(sample, predictors[5]![1], predictors[2]![1])}`)
const multi = sample.filter((i) => byItem.get(i)!.length >= 3)
console.log(`  on the ${multi.length} comparisons with three or more votes:`)
console.log(`    majority of the other experts   ${fmt(sum(multi.map(majorityTally)))}`)
console.log(`    expert vs expert                ${fmt(sum(multi.map(expertTally)))}`)
console.log(`    Opus 5.5 judge                  ${fmt(sum(multi.map((i) => tally(i, opusJudge[i]))))}`)

const norm = (t: string) => t.toLowerCase().replace(/[^a-z0-9\s]/g, " ").split(/\s+/).filter(Boolean)
const runs = Object.entries(recall).map(([id, cont]) => {
  const w = questions.get(Number(id))!.turns[0]!.split(/\s+/)
  const truth = norm(w.slice(Math.ceil(w.length / 2)).join(" "))
  const got = norm(cont)
  // Longest run of consecutive words the continuation shares with the true rest of the question.
  let best = 0
  for (let i = 0; i < truth.length; i++) for (let j = 0; j < got.length; j++) {
    let k = 0
    while (i + k < truth.length && j + k < got.length && truth[i + k] === got[j + k]) k++
    best = Math.max(best, k)
  }
  return { best, len: truth.length, first5: truth.slice(0, 5).join(" ") === got.slice(0, 5).join(" ") }
})
console.log(`\nQuestion recall (${runs.length} questions of 20+ words; Opus given the first half): continuation shares a run of 8+ consecutive words with the true rest in ${runs.filter((r) => r.best >= 8).length} (${pct(runs.filter((r) => r.best >= 8).length / runs.length)}); its first five words are exact in ${runs.filter((r) => r.first5).length}; median longest run ${[...runs].map((r) => r.best).sort((a, b) => a - b)[Math.floor(runs.length / 2)]} of a median ${[...runs].map((r) => r.len).sort((a, b) => a - b)[Math.floor(runs.length / 2)]} words`)

if (NO_GRADER) {
  console.log(`\nspend this run: Opus $${opusSpend.usd.toFixed(2)} over ${opusSpend.calls} calls`)
  process.exit(0)
}
const keys = Object.keys(opusCorrect).filter((k) => geminiCorrect[k] !== undefined)
const both = keys.filter((k) => opusCorrect[k] === geminiCorrect[k]).length
const po = keys.filter((k) => opusCorrect[k] === 1).length / keys.length
const pg = keys.filter((k) => geminiCorrect[k] === 1).length / keys.length
const pe = po * pg + (1 - po) * (1 - pg)
const kappa = (both / keys.length - pe) / (1 - pe)
console.log(`\nCorrectness graders: ${keys.length} answers graded by both; agree on ${pct(both / keys.length)}, Cohen's κ ${kappa.toFixed(2)}; correct: Opus ${pct(po)}, Gemini ${pct(pg)}`)
console.log(`When exactly one answer of the pair is correct, experts prefer it (ties as half):`)
for (const cat of [...OBJECTIVE]) {
  const row = (grade: (k: string) => number | undefined) => {
    let right = 0
    let n = 0
    for (const v of human.filter((v) => catOf(v) === cat)) {
      const ca = grade(akey(v.question_id, v.turn, v.model_a))
      const cb = grade(akey(v.question_id, v.turn, v.model_b))
      if (ca === undefined || cb === undefined || ca === cb) continue
      n++
      right += v.winner === "tie" ? 0.5 : (v.winner === "model_a") === (ca === 1) ? 1 : 0
    }
    return `${pct(right / n)} of ${n}`
  }
  const agreed = (k: string) => (opusCorrect[k] !== undefined && opusCorrect[k] === geminiCorrect[k] ? opusCorrect[k] : undefined)
  console.log(`  ${cat.padEnd(10)} Opus grades ${row((k) => opusCorrect[k])}   Gemini grades ${row((k) => geminiCorrect[k])}   where both agree ${row(agreed)}`)
}
console.log(`\nspend this run: Opus $${opusSpend.usd.toFixed(2)} over ${opusSpend.calls} calls, Gemini $${geminiSpend.usd.toFixed(2)} over ${geminiSpend.calls} calls`)
