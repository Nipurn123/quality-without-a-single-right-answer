import { mkdir } from "node:fs/promises"
import { json, mulberry32, opus, opusSpend, pool } from "./llm.ts"

// The unified quality equation on real people: MT-Bench's human judgments (LMSYS; Zheng et al.,
// "Judging LLM-as-a-Judge", NeurIPS 2023). 3,355 blind pairwise votes by 65 experts on the answers of
// six models (GPT-4, GPT-3.5, Claude v1, Vicuna-13B, Alpaca-13B, LLaMA-13B) to 80 two-turn questions
// in eight categories, each vote tagged with its expert; 961 comparisons were judged by two or more
// experts. GPT-4's own pairwise verdicts on the same comparisons come with it.
//
//   U_u(a) = θ_model(a) + w·c(a) + βᵀφ(a) + η_a  +  (θ_u − θ̄)ᵀφ(a)
//            leaderboard  correct  shared style   answer  personal taste over style and model
//
// Fitted per category in nested steps, each kept only if it predicts held-out votes better:
//   L1 model strength (the leaderboard) → L2 + correctness (math, reasoning, coding: graded by Opus
//   against MT-Bench's reference answers) → L3 + shared quality (style features and a per-answer
//   term) → L4 + each expert's personal taste (deviations on the same style and model features).
// Then Opus 5.5 judges 300 comparisons the experts judged, against how often experts agree with
// each other and with GPT-4.
//
// The downloaded files in mtbench-data/ are read as data only; everything derived is cached in
// mtbench-out/, so a rerun costs nothing.
//
//   bun run experiments/subjectivity/mtbench.ts [--concurrency 8]

const IN = new URL("./mtbench-data/", import.meta.url).pathname
const OUT = new URL("./mtbench-out/", import.meta.url).pathname
await mkdir(OUT, { recursive: true })
const N = Number(Bun.argv.includes("--concurrency") ? Bun.argv[Bun.argv.indexOf("--concurrency") + 1] : 8)

interface Msg { role: string; content: string }
interface Vote { judge: string; question_id: number; turn: number; model_a: string; model_b: string; winner: "model_a" | "model_b" | "tie" | string; conversation_a: Msg[]; conversation_b: Msg[] }
interface Question { question_id: number; category: string; turns: string[]; reference?: string[] }

const lines = async <T,>(f: string) => (await Bun.file(f).text()).trim().split("\n").map((l) => JSON.parse(l) as T)
const human = await lines<Vote>(`${IN}human.jsonl`)
const gpt4 = await lines<Vote>(`${IN}gpt4_pair.jsonl`)
const questions = new Map((await lines<Question>(`${IN}question.jsonl`)).map((q) => [q.question_id, q]))
const CATEGORIES = ["writing", "roleplay", "humanities", "stem", "extraction", "reasoning", "math", "coding"]
const OBJECTIVE = new Set(["reasoning", "math", "coding"])
const catOf = (v: { question_id: number }) => questions.get(v.question_id)!.category

// Every answer, keyed by question, turn and model: the assistant message of that turn.
const answers = new Map<string, string>()
const akey = (q: number, turn: number, model: string) => `${q}/${turn}/${model}`
for (const v of [...human, ...gpt4]) {
  for (const [model, conv] of [[v.model_a, v.conversation_a], [v.model_b, v.conversation_b]] as const) {
    const msg = conv?.filter((m) => m.role === "assistant")[v.turn - 1]
    if (msg) answers.set(akey(v.question_id, v.turn, model), msg.content)
  }
}
const MODELS = [...new Set(human.flatMap((v) => [v.model_a, v.model_b]))].sort()
const itemOf = (v: Vote) => `${v.question_id}/${v.turn}/${[v.model_a, v.model_b].sort().join("|")}`
/** 1 if the alphabetically first model of the pair won, 0 if the other did, 0.5 for a tie. */
const label = (v: Vote) => (v.winner === "tie" ? 0.5 : (v.winner === "model_a" ? v.model_a : v.model_b) === [v.model_a, v.model_b].sort()[0] ? 1 : 0)
console.log(`${human.length} expert votes by ${new Set(human.map((v) => v.judge)).size} experts on ${new Set(human.map(itemOf)).size} comparisons; ${gpt4.length} GPT-4 verdicts; ${answers.size} answers; models ${MODELS.join(", ")}`)

async function cached<T>(name: string, make: () => Promise<T>): Promise<T> {
  const f = Bun.file(`${OUT}${name}`)
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
const conversation = (q: Question, turn: number) => q.turns.slice(0, turn).map((t, i) => `USER (turn ${i + 1}): ${t}`).join("\n\n")

// ── 1. Correctness against MT-Bench's references (the c term) ──

const correct = await cached<Record<string, 0 | 1>>("correctness.json", async () => {
  const out: Record<string, 0 | 1> = {}
  const units = [...questions.values()].filter((q) => OBJECTIVE.has(q.category) && q.reference?.length).flatMap((q) => [1, 2].map((turn) => ({ q, turn })))
  await pool(units, N, async ({ q, turn }) => {
    const ref = q.reference![turn - 1]
    if (!ref) return
    const models = shuffle(MODELS.filter((m) => answers.has(akey(q.question_id, turn, m))))
    const reply = await opus(
      `Grade answers to a ${q.category} question against the reference answer. Judge only whether each answer's final result is correct (for code: whether it correctly does what was asked); ignore style and length.\n\nCONVERSATION:\n${conversation(q, turn)}\n\nREFERENCE ANSWER for turn ${turn}:\n${ref}\n\n` +
        models.map((m, i) => `ANSWER ${i + 1} (to turn ${turn}):\n${answers.get(akey(q.question_id, turn, m))}`).join("\n\n") +
        `\n\nReply with JSON only: [{"answer": 1, "correct": true}, ...] with one entry per answer.`,
    )
    for (const r of json<Array<{ answer: number; correct: boolean }>>(reply)) if (models[r.answer - 1]) out[akey(q.question_id, turn, models[r.answer - 1]!)] = r.correct ? 1 : 0
  })
  return out
})
console.log(`correctness: ${Object.keys(correct).length} answers graded, ${Object.values(correct).filter(Boolean).length} correct`)

// ── 2. Opus 5.5 as a judge on comparisons the experts judged twice or more ──

const byItem = new Map<string, Vote[]>()
for (const v of human) byItem.set(itemOf(v), [...(byItem.get(itemOf(v)) ?? []), v])
const opusVerdicts = await cached<Record<string, number>>("opus-judge.json", async () => {
  // 300 comparisons with at least two expert votes, spread evenly over the categories.
  const sample = CATEGORIES.flatMap((c) => shuffle([...byItem.entries()].filter(([, vs]) => vs.length >= 2 && catOf(vs[0]!) === c)).slice(0, 38)).slice(0, 300)
  // Six per call, from six different questions, each in a random order.
  const batches: Array<typeof sample> = []
  const left = shuffle(sample)
  while (left.length) {
    const batch: typeof sample = []
    for (let i = 0; i < left.length && batch.length < 6; ) {
      if (batch.some(([, vs]) => vs[0]!.question_id === left[i]![1][0]!.question_id)) i++
      else batch.push(...left.splice(i, 1))
    }
    batches.push(batch)
  }
  const out: Record<string, number> = {}
  await pool(batches, N, async (batch) => {
    const shown = batch.map(([item, vs]) => {
      const v = vs[0]!
      const [m1, m2] = [v.model_a, v.model_b].sort()
      const flip = rng() < 0.5
      return { item, v, first: flip ? m2! : m1!, second: flip ? m1! : m2! }
    })
    const reply = await opus(
      `You are judging AI assistants' answers. For each item, read the conversation and decide which assistant's answer to the LAST user turn is better: more helpful, relevant, accurate, deep and well reasoned. Do not let length or order sway you. Answer "A", "B" or "tie".\n\n` +
        shown.map((s, i) => `=== ITEM ${i + 1} ===\n${conversation(questions.get(s.v.question_id)!, s.v.turn)}\n\nASSISTANT A:\n${answers.get(akey(s.v.question_id, s.v.turn, s.first))}\n\nASSISTANT B:\n${answers.get(akey(s.v.question_id, s.v.turn, s.second))}`).join("\n\n") +
        `\n\nReply with JSON only: [{"item": 1, "verdict": "A" | "B" | "tie"}, ...]`,
    )
    for (const r of json<Array<{ item: number; verdict: string }>>(reply)) {
      const s = shown[r.item - 1]
      if (!s) continue
      const sorted = [s.v.model_a, s.v.model_b].sort()[0]
      out[s.item] = r.verdict === "tie" ? 0.5 : (r.verdict === "A" ? s.first : s.second) === sorted ? 1 : 0
    }
  })
  return out
})
console.log(`Opus judged ${Object.keys(opusVerdicts).length} comparisons; Opus spend this run $${opusSpend.usd.toFixed(2)} over ${opusSpend.calls} calls\n`)

// ── 3. The equation, fitted per category ──

const sigmoid = (z: number) => 1 / (1 + Math.exp(-z))
const z = (xs: number[]) => {
  const m = xs.reduce((s, x) => s + x, 0) / xs.length
  const sd = Math.sqrt(xs.reduce((s, x) => s + (x - m) ** 2, 0) / xs.length) || 1
  return xs.map((x) => (x - m) / sd)
}
/** Style features of an answer: length (log characters), markdown structure, code blocks. */
const style = (t: string) => [Math.log(1 + t.length), (t.match(/^\s*(#{1,6} |[-*] |\d+\. )/gm) ?? []).length + (t.match(/\*\*/g) ?? []).length / 2, t.includes("```") ? 1 : 0]

type Obs = { u: string; a: string; b: string; y: number }
type Level = "L1" | "L2" | "L3" | "L4"

function fitter(keys: string[], phi: Map<string, number[]>, hasC: boolean) {
  const F = phi.values().next().value!.length
  const mi = new Map(MODELS.map((m, i) => [m, i]))
  const ai = new Map(keys.map((k, i) => [k, i]))
  const modelOf = (k: string) => mi.get(k.split("/")[2]!)!
  return (obs: Obs[], level: Level, sEta: number, tau: number) => {
    const raters = [...new Set(obs.map((o) => o.u))]
    const ri = new Map(raters.map((r, i) => [r, i]))
    // Parameter layout: models | w | β (F) | η (answers) | γ (raters × F)
    const oM = 0, oW = MODELS.length, oB = oW + 1, oE = oB + F, oG = oE + keys.length
    const P = oG + raters.length * F
    const th = new Float64Array(P), m1 = new Float64Array(P), m2 = new Float64Array(P)
    const on = (p: number) => p < oW || (p === oW ? level !== "L1" && hasC : p < oE ? level === "L3" || level === "L4" : p < oG ? level === "L3" || level === "L4" : level === "L4")
    const prior = (p: number) => (p < oW ? 1 / 4 : p === oW ? 1 / 25 : p < oE ? 1 / 4 : p < oG ? 1 / (sEta * sEta) : 1 / (tau * tau))
    const U = (u: string, k: string) => {
      const f = phi.get(k)!
      let s = th[oM + modelOf(k)]!
      if (on(oW)) s += th[oW]! * (correct[k] ?? 0)
      if (level === "L3" || level === "L4") {
        for (let j = 0; j < F; j++) s += th[oB + j]! * f[j]!
        s += th[oE + ai.get(k)!]!
      }
      if (level === "L4" && ri.has(u)) for (let j = 0; j < F; j++) s += th[oG + ri.get(u)! * F + j]! * f[j]!
      return s
    }
    for (let it = 1; it <= 600; it++) {
      const g = new Float64Array(P)
      for (const o of obs) {
        const r = o.y - sigmoid(U(o.u, o.a) - U(o.u, o.b))
        for (const [k, sgn] of [[o.a, 1], [o.b, -1]] as const) {
          const f = phi.get(k)!
          g[oM + modelOf(k)]! += sgn * r
          if (on(oW)) g[oW]! += sgn * r * (correct[k] ?? 0)
          if (level === "L3" || level === "L4") {
            for (let j = 0; j < F; j++) g[oB + j]! += sgn * r * f[j]!
            g[oE + ai.get(k)!]! += sgn * r
          }
          if (level === "L4") for (let j = 0; j < F; j++) g[oG + ri.get(o.u)! * F + j]! += sgn * r * f[j]!
        }
      }
      for (let p = 0; p < P; p++) {
        if (!on(p)) continue
        const gp = g[p]! - prior(p) * th[p]!
        m1[p] = 0.9 * m1[p]! + 0.1 * gp
        m2[p] = 0.999 * m2[p]! + 0.001 * gp * gp
        th[p]! += (0.05 * (m1[p]! / (1 - 0.9 ** it))) / (Math.sqrt(m2[p]! / (1 - 0.999 ** it)) + 1e-8)
      }
    }
    const shared = (k: string) => U("\u0000none", k)
    return { U, shared, beta: Array.from(th.slice(oB, oE)), w: th[oW]! }
  }
}
const logLik = (U: (u: string, k: string) => number, o: Obs) => {
  const p = Math.min(1 - 1e-6, Math.max(1e-6, sigmoid(U(o.u, o.a) - U(o.u, o.b))))
  return o.y * Math.log(p) + (1 - o.y) * Math.log(1 - p)
}
const meanCi = (xs: number[]) => {
  const m = xs.reduce((s, x) => s + x, 0) / xs.length
  const sd = Math.sqrt(xs.reduce((s, x) => s + (x - m) ** 2, 0) / (xs.length - 1))
  return { m, h: (1.96 * sd) / Math.sqrt(xs.length) }
}
const pct = (x: number) => `${(100 * x).toFixed(0)}%`

const rows: Array<Record<string, unknown>> = []
for (const cat of CATEGORIES) {
  const votes = human.filter((v) => catOf(v) === cat)
  const keys = [...new Set(votes.flatMap((v) => [akey(v.question_id, v.turn, v.model_a), akey(v.question_id, v.turn, v.model_b)]))]
  // φ: z-scored style features plus the model's one-hot (a person's taste for a model's voice).
  const raw = keys.map((k) => style(answers.get(k) ?? ""))
  const cols = raw[0]!.map((_, j) => z(raw.map((r) => r[j]!)))
  const phi = new Map(keys.map((k, i) => [k, [...cols.map((c) => c[i]!), ...MODELS.map((m) => (k.endsWith(`/${m}`) ? 1 : 0))]]))
  const hasC = OBJECTIVE.has(cat)
  const fit = fitter(keys, phi, hasC)
  const obs: Obs[] = votes.map((v) => ({ u: v.judge, a: akey(v.question_id, v.turn, v.model_a), b: akey(v.question_id, v.turn, v.model_b), y: v.winner === "model_a" ? 1 : v.winner === "model_b" ? 0 : 0.5 }))
  // Five-fold cross-validation repeated five times, each vote's held-out log-likelihood averaged over
  // the repeats: one split's luck can decide a borderline test. Folds come from their own seeds, so
  // the fit does not depend on how many random draws the (cached or not) stages above consumed.
  const REPEATS = 5
  const folds = Array.from({ length: REPEATS }, (_, r) => {
    const fr = mulberry32(20261025 + 100 * CATEGORIES.indexOf(cat) + r)
    const keysR = obs.map(() => fr())
    const fold = new Array<number>(obs.length)
    obs.map((_, i) => i).sort((p, q) => keysR[p]! - keysR[q]!).forEach((i, k) => (fold[i] = k % 5))
    return fold
  })
  const cv = (level: Level, sEta: number, tau: number) => {
    const out = new Array<number>(obs.length).fill(0)
    for (const fold of folds) for (let f = 0; f < 5; f++) {
      const m = fit(obs.filter((_, i) => fold[i] !== f), level, sEta, tau)
      obs.forEach((o, i) => {
        if (fold[i] === f) out[i]! += logLik(m.U, o) / REPEATS
      })
    }
    return out
  }
  const sum = (xs: number[]) => xs.reduce((s, x) => s + x, 0)
  const l1 = cv("L1", 1, 1)
  const l2 = hasC ? cv("L2", 1, 1) : l1
  let bestEta = 1
  let l3 = cv("L3", 1, 1)
  for (const s of [0.5, 2]) {
    const l = cv("L3", s, 1)
    if (sum(l) > sum(l3)) [l3, bestEta] = [l, s]
  }
  let bestTau = 0.25
  let l4 = cv("L4", bestEta, 0.25)
  for (const t of [0.1, 0.5, 1]) {
    const l = cv("L4", bestEta, t)
    if (sum(l) > sum(l4)) [l4, bestTau] = [l, t]
  }
  const gain = (a: number[], b: number[]) => meanCi(a.map((x, i) => x - b[i]!))
  // Subjectivity: the personal part's share of the utility differences the full model gives each vote.
  const full = fit(obs, "L4", bestEta, bestTau)
  const sharedD = obs.map((o) => full.shared(o.a) - full.shared(o.b))
  const personalD = obs.map((o, i) => full.U(o.u, o.a) - full.U(o.u, o.b) - sharedD[i]!)
  const variance = (xs: number[]) => {
    const m = xs.reduce((s, x) => s + x, 0) / xs.length
    return xs.reduce((s, x) => s + (x - m) ** 2, 0) / xs.length
  }
  const S = variance(personalD) / (variance(sharedD) + variance(personalD))
  // How often two experts judging the same comparison differ (ties count half).
  let dis = 0
  let pairs = 0
  for (const [, vs] of [...byItem].filter(([, vs]) => catOf(vs[0]!) === cat)) for (let i = 0; i < vs.length; i++) for (let j = i + 1; j < vs.length; j++) {
    dis += Math.abs(label(vs[i]!) - label(vs[j]!))
    pairs++
  }
  // Correctness: when exactly one answer is right, how often do experts prefer it?
  let right = 0
  let decided = 0
  if (hasC) for (const v of votes) {
    const ca = correct[akey(v.question_id, v.turn, v.model_a)]
    const cb = correct[akey(v.question_id, v.turn, v.model_b)]
    if (ca === undefined || cb === undefined || ca === cb) continue
    decided++
    right += v.winner === "tie" ? 0.5 : (v.winner === "model_a") === (ca === 1) ? 1 : 0
  }
  const shared = fit(obs, "L3", bestEta, 1)
  const g21 = gain(l2, l1), g32 = gain(l3, l2), g43 = gain(l4, l3)
  rows.push({ cat, votes: obs.length, l1: meanCi(l1).m, g21, g32, g43, S, disagree: pairs ? dis / pairs : NaN, pairs, lengthEffect: shared.beta[0], right, decided })
}

const f3 = (x: number) => x.toFixed(3)
// Two-sided p-values for each gain (normal approximation), Holm–Bonferroni over all 19 tests:
// correctness in the three categories with references, shared quality and personal taste in all eight.
const erf = (x: number) => {
  const t = 1 / (1 + 0.3275911 * Math.abs(x))
  const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x)
  return x >= 0 ? y : -y
}
const pValue = (g: { m: number; h: number }) => (g.h > 0 ? 1 - erf(Math.abs(g.m / (g.h / 1.96)) / Math.SQRT2) : 1)
const tests: Array<{ g: { m: number; h: number }; p: number; holm: number }> = []
for (const r of rows as Array<any>) for (const g of [...(OBJECTIVE.has(r.cat) ? [r.g21] : []), r.g32, r.g43]) tests.push({ g, p: pValue(g), holm: 1 })
const byP = [...tests].sort((a, b) => a.p - b.p)
let running = 0
byP.forEach((t, i) => {
  running = Math.max(running, Math.min(1, (byP.length - i) * t.p))
  t.holm = running
})
const holmOf = (g: unknown) => tests.find((t) => t.g === g)!
const est = (g: { m: number; h: number }) => `${g.m >= 0 ? "+" : ""}${f3(g.m)} ± ${f3(g.h)}${holmOf(g).holm < 0.05 ? " †" : ""}`
console.log(`Held-out log-likelihood gain per vote of each added term (95% interval; † = significant after Holm–Bonferroni over the ${tests.length} tests):\n`)
console.log("category".padEnd(11), "votes".padStart(5), "  +correctness       +shared quality     +personal taste     S     experts disagree  shared length effect  correct answer preferred")
for (const r of rows as Array<any>) {
  console.log(
    r.cat.padEnd(11), String(r.votes).padStart(5), "  ",
    (OBJECTIVE.has(r.cat) ? est(r.g21) : "(no reference)").padEnd(19),
    est(r.g32).padEnd(19), est(r.g43).padEnd(19),
    r.S.toFixed(2).padStart(4), `   ${pct(r.disagree)} (${r.pairs} pairs)`.padEnd(20),
    `   ${r.lengthEffect >= 0 ? "+" : ""}${r.lengthEffect.toFixed(2)}`.padEnd(22),
    r.decided ? `${pct(r.right / r.decided)} of ${r.decided}` : "",
  )
}
console.log(`\np-values (raw / Holm): ${(rows as Array<any>).map((r) => `${r.cat} ${[...(OBJECTIVE.has(r.cat) ? [["c", r.g21]] : []), ["s", r.g32], ["p", r.g43]].map(([n, g]) => `${n} ${holmOf(g).p.toFixed(3)}/${holmOf(g).holm.toFixed(3)}`).join(" ")}`).join("; ")}`)

// ── 4. Judges against people ──

console.log("\nAgreement with an expert's vote on the same comparison (3-way: A, B, tie; and on votes where both picked a side):")
const agreeRows: string[] = []
for (const scope of ["all", ...CATEGORIES]) {
  const items = [...byItem].filter(([item, vs]) => vs.length >= 2 && opusVerdicts[item] !== undefined && (scope === "all" || catOf(vs[0]!) === scope))
  let hh = 0, hhN = 0, hhD = 0, hhDN = 0
  let oh = 0, ohN = 0, ohD = 0, ohDN = 0
  let gh = 0, ghN = 0, ghD = 0, ghDN = 0
  const gpt = new Map(gpt4.map((v) => [itemOf(v), label(v)]))
  for (const [item, vs] of items) {
    const ls = vs.map(label)
    for (let i = 0; i < ls.length; i++) for (let j = i + 1; j < ls.length; j++) {
      hhN++
      if (ls[i] === ls[j]) hh++
      if (ls[i] !== 0.5 && ls[j] !== 0.5) {
        hhDN++
        if (ls[i] === ls[j]) hhD++
      }
    }
    for (const [verdict, tally] of [[opusVerdicts[item], "o"], [gpt.get(item), "g"]] as const) {
      if (verdict === undefined) continue
      for (const l of ls) {
        const same = l === verdict ? 1 : 0
        const decided = l !== 0.5 && verdict !== 0.5
        if (tally === "o") {
          ohN++, (oh += same)
          if (decided) ohDN++, (ohD += same)
        } else {
          ghN++, (gh += same)
          if (decided) ghDN++, (ghD += same)
        }
      }
    }
  }
  agreeRows.push(`${scope.padEnd(11)} ${String(items.length).padStart(4)} comparisons   expert–expert ${pct(hh / hhN)} / ${pct(hhD / hhDN)}   GPT-4–expert ${pct(gh / ghN)} / ${pct(ghD / ghDN)}   Opus 5.5–expert ${pct(oh / ohN)} / ${pct(ohD / ohDN)}`)
}
console.log(agreeRows.join("\n"))
await Bun.write(`${OUT}results.json`, JSON.stringify({ rows, agreement: agreeRows }, null, 1))
