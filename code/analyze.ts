// Fits the unified quality equation to the judgments run.ts collected and runs its validation
// tests. Per task kind:
//
//   P(rater u prefers a over b) = σ( U_u(a) − U_u(b) ),   U_u(a) = q_a + d_{u,a}
//     M0  correctness only:  q_a = w · (claims of a the fact check found contradicted), d = 0
//     M1  + shared quality:  q_a free per answer (prior N(0, σ_q²)), d = 0
//     M2  + personal taste:  d_{u,a} ~ N(0, τ²) per rater and answer
//
// σ_q and τ are chosen by 5-fold cross-validation on held-out judgments; the subjectivity index is
// S = τ² / (σ_q² + τ²). The persona raters are the population; the impartial judge and DeepSeek are
// instruments, compared against it. Ties count as half a win.
//
// With --raters gemini the population is Gemini 3.8 Flash's portrayal of the same seven raters
// (run.ts --raters gemini), and the two portrayals are compared vote by vote.
//
//   bun run experiments/subjectivity/analyze.ts [--raters opus|gemini]

import { parseArgs } from "node:util"
import type { FactCheck, Judgment, Task, Variant } from "./run.ts"

const DATA = new URL("./data/", import.meta.url).pathname
const tasks = (await Bun.file(`${DATA}tasks.json`).json()) as Task[]
const facts = (await Bun.file(`${DATA}factchecks.json`).json()) as FactCheck[]
const { values } = parseArgs({ args: Bun.argv.slice(2), options: { raters: { type: "string", default: "opus" } } })
const FAMILY = values.raters === "gemini" ? "gemini" : "opus"
const read = async (f: string) => ((await Bun.file(f).exists()) ? (await Bun.file(f).text()).trim().split("\n").filter(Boolean).map((l) => JSON.parse(l) as Judgment) : [])
const all = [...(await read(`${DATA}judgments.jsonl`)), ...(FAMILY === "gemini" ? await read(`${DATA}judgments-gemini.jsonl`) : [])]
const KINDS = ["docs-qa", "support-reply", "launch-post"] as const
const PERSONAS = ["priya", "marco", "aisha", "tom", "lena", "kenji"]
const kindOf = new Map(tasks.map((t) => [t.id, t.kind]))
const sigmoid = (z: number) => 1 / (1 + Math.exp(-z))
const outcome = (j: Judgment) => (j.preferred === "first" ? 1 : j.preferred === "second" ? 0 : 0.5)
const contradicted = (task: string, v: Variant) => facts.find((f) => f.task === task && f.variant === v)?.claims.filter((c) => c.verdict === "contradicted").length ?? 0
const pct = (x: number) => `${(100 * x).toFixed(0)}%`
const ci = (k: number, n: number) => {
  // Wilson 95% interval for a proportion.
  if (!n) return [0, 1]
  const p = k / n
  const z = 1.96
  const c = (p + (z * z) / (2 * n)) / (1 + (z * z) / n)
  const h = (z * Math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n))) / (1 + (z * z) / n)
  return [c - h, c + h]
}

// ── The model ──

type Obs = { u: string; a: string; b: string; y: number }
const key = (task: string, v: Variant) => `${task}/${v}`

/** MAP fit by Adam: shared q per answer (prior sd σq), personal d per rater and answer (prior sd τ), or the M0 weight. */
function fit(obs: Obs[], answers: string[], model: "M0" | "M1" | "M2", sq: number, tau: number, errs: Map<string, number>) {
  const qi = new Map(answers.map((a, i) => [a, i]))
  const raters = [...new Set(obs.map((o) => o.u))]
  const di = new Map<string, number>()
  if (model === "M2") for (const u of raters) for (const a of answers) di.set(`${u}|${a}`, di.size)
  const P = model === "M0" ? 1 : answers.length + di.size
  const th = new Float64Array(P)
  const m = new Float64Array(P)
  const v = new Float64Array(P)
  const util = (u: string, a: string) =>
    model === "M0" ? th[0]! * (errs.get(a) ?? 0) : th[qi.get(a)!]! + (model === "M2" ? th[answers.length + di.get(`${u}|${a}`)!]! : 0)
  for (let it = 1; it <= 1500; it++) {
    const g = new Float64Array(P)
    for (const o of obs) {
      const r = o.y - sigmoid(util(o.u, o.a) - util(o.u, o.b))
      if (model === "M0") g[0]! += r * ((errs.get(o.a) ?? 0) - (errs.get(o.b) ?? 0))
      else {
        g[qi.get(o.a)!]! += r
        g[qi.get(o.b)!]! -= r
        if (model === "M2") {
          g[answers.length + di.get(`${o.u}|${o.a}`)!]! += r
          g[answers.length + di.get(`${o.u}|${o.b}`)!]! -= r
        }
      }
    }
    for (let p = 0; p < P; p++) {
      const prior = model === "M0" ? 0.01 : p < answers.length ? 1 / (sq * sq) : tau > 0 ? 1 / (tau * tau) : 1e6
      const gp = g[p]! - prior * th[p]!
      m[p] = 0.9 * m[p]! + 0.1 * gp
      v[p] = 0.999 * v[p]! + 0.001 * gp * gp
      th[p]! += (0.05 * (m[p]! / (1 - 0.9 ** it))) / (Math.sqrt(v[p]! / (1 - 0.999 ** it)) + 1e-8)
    }
  }
  return { util, q: (a: string) => (model === "M0" ? 0 : th[qi.get(a)!]!) }
}
const ll = (util: (u: string, a: string) => number, os: Obs[]) => os.map((o) => {
  const p = Math.min(1 - 1e-6, Math.max(1e-6, sigmoid(util(o.u, o.a) - util(o.u, o.b))))
  return o.y * Math.log(p) + (1 - o.y) * Math.log(1 - p)
})

/** 5-fold held-out log-likelihood per judgment; `fold` assigns each judgment to a fold (default: by index). */
function cv(obs: Obs[], answers: string[], model: "M0" | "M1" | "M2", sq: number, tau: number, errs: Map<string, number>, fold: (i: number) => number = (i) => i % 5): number[] {
  const out: number[] = new Array(obs.length).fill(0)
  for (let f = 0; f < 5; f++) {
    const test = obs.map((_, i) => fold(i) === f)
    const tr = obs.filter((_, i) => !test[i])
    const m = fit(tr, answers, model, sq, tau, errs)
    obs.forEach((o, i) => {
      if (test[i]) out[i] = ll(m.util, [o])[0]!
    })
  }
  return out
}

console.log(`judgments: ${all.length} (${all.filter((j) => j.repeat).length} repeats; ${all.filter((j) => j.judge === "deepseek").length} by DeepSeek)\n`)

/** The same, repeated over five seeded random fold assignments and averaged per judgment. */
function cvRepeated(obs: Obs[], answers: string[], model: "M0" | "M1" | "M2", sq: number, tau: number, errs: Map<string, number>): number[] {
  const out: number[] = new Array(obs.length).fill(0)
  for (let r = 0; r < 5; r++) {
    let seed = 20261025 + r
    const rand = () => ((seed = (seed * 1103515245 + 12345) % 2147483648), seed / 2147483648)
    const keys = obs.map(() => rand())
    const fold = new Array<number>(obs.length)
    obs.map((_, i) => i).sort((p, q) => keys[p]! - keys[q]!).forEach((i, k) => (fold[i] = k % 5))
    cv(obs, answers, model, sq, tau, errs, (i) => fold[i]!).forEach((x, i) => (out[i]! += x / 5))
  }
  return out
}
const erf = (x: number) => {
  const t = 1 / (1 + 0.3275911 * Math.abs(x))
  const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x)
  return x >= 0 ? y : -y
}
/** Two-sided p-value of a mean gain from its standard error (normal approximation). */
const pOf = (g: { m: number; se: number }) => (g.se > 0 ? 1 - erf(Math.abs(g.m / g.se) / Math.SQRT2) : 1)
const tests: Array<{ name: string; p: number }> = []

const meanSe = (xs: number[]) => {
  const m = xs.reduce((s, x) => s + x, 0) / xs.length
  const sd = Math.sqrt(xs.reduce((s, x) => s + (x - m) ** 2, 0) / (xs.length - 1))
  return { m, se: sd / Math.sqrt(xs.length) }
}

for (const kind of KINDS) {
  const ids = tasks.filter((t) => t.kind === kind).map((t) => t.id)
  const answers = ids.flatMap((t) => (["A", "B", "C", "D"] as Variant[]).map((v) => key(t, v)))
  const errs = new Map(answers.map((a) => [a, contradicted(a.split("/")[0]!, a.split("/")[1] as Variant)]))
  const persona = all.filter((j) => j.judge === FAMILY && PERSONAS.includes(j.rater) && kindOf.get(j.task) === kind)
  const obs: Obs[] = persona.filter((j) => !j.repeat).map((j) => ({ u: j.rater, a: key(j.task, j.first), b: key(j.task, j.second), y: outcome(j) }))

  // Hyperparameters by cross-validation.
  let best1 = { sq: 1, l: -Infinity }
  for (const sq of [0.5, 1, 2, 3, 5]) {
    const l = cv(obs, answers, "M1", sq, 0, errs).reduce((s, x) => s + x, 0)
    if (l > best1.l) best1 = { sq, l }
  }
  let best2 = { sq: best1.sq, tau: 0, l: -Infinity }
  for (const sq of [best1.sq / 2, best1.sq, best1.sq * 1.5]) for (const tau of [0.25, 0.5, 1, 1.5, 2, 3]) {
    const l = cv(obs, answers, "M2", sq, tau, errs).reduce((s, x) => s + x, 0)
    if (l > best2.l) best2 = { sq, tau, l }
  }
  // Prior widths chosen on the fixed split above; the reported gains average five random splits.
  const l0 = cvRepeated(obs, answers, "M0", 1, 0, errs)
  const l1 = cvRepeated(obs, answers, "M1", best1.sq, 0, errs)
  const l2 = cvRepeated(obs, answers, "M2", best2.sq, best2.tau, errs)
  const d10 = meanSe(l1.map((x, i) => x - l0[i]!))
  const d21 = meanSe(l2.map((x, i) => x - l1[i]!))
  tests.push({ name: `${kind} shared`, p: pOf(d10) }, { name: `${kind} personal`, p: pOf(d21) })
  const S = best2.tau ** 2 / (best2.sq ** 2 + best2.tau ** 2)

  // Non-parametric disagreement: between personas on the same comparison, and within a persona on repeats.
  const vote = (j: Judgment) => (j.preferred === "tie" ? 0.5 : (j.preferred === "first") === (j.first < j.second) ? 1 : 0) // 1 = the earlier variant letter wins
  const byItem = new Map<string, number[]>()
  for (const j of persona.filter((j) => !j.repeat)) {
    const k = `${j.task}|${[j.first, j.second].sort().join("")}`
    byItem.set(k, [...(byItem.get(k) ?? []), vote(j)])
  }
  let dis = 0
  let pairs = 0
  for (const vs of byItem.values()) for (let i = 0; i < vs.length; i++) for (let k = i + 1; k < vs.length; k++) {
    dis += Math.abs(vs[i]! - vs[k]!)
    pairs++
  }
  const between = dis / pairs
  let wdis = 0
  let wn = 0
  for (const r of persona.filter((j) => j.repeat)) {
    const orig = persona.find((j) => !j.repeat && j.rater === r.rater && j.task === r.task && [j.first, j.second].sort().join("") === [r.first, r.second].sort().join(""))
    if (!orig) continue
    wdis += Math.abs(vote(orig) - vote(r))
    wn++
  }
  const within = wn ? wdis / wn : NaN

  // Transitivity: cyclic triads per persona and task (strict choices only).
  let triads = 0
  let cyclic = 0
  for (const u of PERSONAS) for (const t of ids) {
    const beats = new Map<string, boolean>()
    for (const j of persona.filter((j) => !j.repeat && j.rater === u && j.task === t && j.preferred !== "tie")) {
      const [w, l] = j.preferred === "first" ? [j.first, j.second] : [j.second, j.first]
      beats.set(`${w}>${l}`, true)
    }
    const vs: Variant[] = ["A", "B", "C", "D"]
    for (let a = 0; a < 4; a++) for (let b = a + 1; b < 4; b++) for (let c = b + 1; c < 4; c++) {
      const [x, y, z] = [vs[a]!, vs[b]!, vs[c]!]
      const has = (p: string, q: string) => beats.has(`${p}>${q}`) || beats.has(`${q}>${p}`)
      if (!has(x, y) || !has(y, z) || !has(x, z)) continue
      triads++
      const cyc = (beats.has(`${x}>${y}`) && beats.has(`${y}>${z}`) && beats.has(`${z}>${x}`)) || (beats.has(`${y}>${x}`) && beats.has(`${z}>${y}`) && beats.has(`${x}>${z}`))
      if (cyc) cyclic++
    }
  }

  // Win rate of one variant over another, among a set of judgments.
  const winRate = (js: Judgment[], w: Variant, l: Variant) => {
    const rel = js.filter((j) => !j.repeat && [j.first, j.second].sort().join("") === [w, l].sort().join(""))
    const wins = rel.reduce((s, j) => s + (j.preferred === "tie" ? 0.5 : (j.preferred === "first" ? j.first : j.second) === w ? 1 : 0), 0)
    return { wins, n: rel.length, rate: rel.length ? wins / rel.length : NaN }
  }
  const judgeJs = all.filter((j) => j.judge === FAMILY && j.rater === "judge" && kindOf.get(j.task) === kind)
  const opusJs = all.filter((j) => j.judge === "opus" && j.rater === "judge" && kindOf.get(j.task) === kind)
  const dsJs = all.filter((j) => j.judge === "deepseek" && j.rater === "judge" && kindOf.get(j.task) === kind)
  const flagged = (v: Variant) => ids.filter((t) => contradicted(t, v) > 0).length

  // Outcome link (simulated): does the fitted utility predict "approve as is"?
  const m2 = fit(obs, answers, "M2", best2.sq, best2.tau, errs)
  const pts: Array<{ x: number; y: number }> = []
  for (const j of persona.filter((j) => !j.repeat)) {
    pts.push({ x: m2.util(j.rater, key(j.task, j.first)), y: j.useAsIs[0] ? 1 : 0 }, { x: m2.util(j.rater, key(j.task, j.second)), y: j.useAsIs[1] ? 1 : 0 })
  }
  let a0 = 0
  let b1 = 0
  for (let it = 0; it < 3000; it++) {
    let ga = 0
    let gb = 0
    for (const p of pts) {
      const r = p.y - sigmoid(a0 + b1 * p.x)
      ga += r
      gb += r * p.x
    }
    a0 += (0.5 * ga) / pts.length
    b1 += (0.5 * gb) / pts.length
  }
  const approve = (lo: boolean) => pts.filter((p) => (lo ? p.x < 0 : p.x >= 0))
  const rate = (ps: typeof pts) => ps.reduce((s, p) => s + p.y, 0) / Math.max(1, ps.length)

  const bA = winRate(persona, "B", "A")
  const [lo] = ci(bA.wins, bA.n)
  console.log(`══ ${kind} (${ids.length} tasks, ${obs.length} persona judgments) ══`)
  console.log(`  fit, held-out log-likelihood per judgment: M0 correctness only ${meanSe(l0).m.toFixed(3)}, M1 +shared ${meanSe(l1).m.toFixed(3)}, M2 +personal ${meanSe(l2).m.toFixed(3)}`)
  console.log(`    shared quality adds ${d10.m.toFixed(3)} ± ${(1.96 * d10.se).toFixed(3)}; personal taste adds ${d21.m.toFixed(3)} ± ${(1.96 * d21.se).toFixed(3)}   (± = 95% interval; a gain whose interval includes 0 is not established)`)
  console.log(`    σ_q ${best2.sq}, τ ${best2.tau}  →  subjectivity S = ${S.toFixed(2)}`)
  console.log(`  disagreement: two different personas ${pct(between)} of the time; the same persona with itself on a repeat ${Number.isNaN(within) ? "n/a" : pct(within)} (${wn} repeats)`)
  console.log(`  transitivity: ${cyclic}/${triads} triads cyclic (${triads ? pct(cyclic / triads) : "n/a"}; 25% if choices were random)`)
  const ac = winRate(persona, "A", "C")
  const acJ = winRate(judgeJs, "A", "C")
  console.log(`  error edit: clean A beats A-with-one-error ${pct(ac.rate)} of persona votes, ${pct(acJ.rate)} of the judge's; fact check flags the error in ${flagged("C")}/${ids.length}, flags a clean answer in A ${flagged("A")}, B ${flagged("B")}, D ${flagged("D")} of ${ids.length}`)
  const db = winRate(persona, "D", "B")
  const dbJ = winRate(judgeJs, "D", "B")
  const perPersona = PERSONAS.map((u) => `${u} ${pct(winRate(persona.filter((j) => j.rater === u), "D", "B").rate)}`).join(", ")
  console.log(`  padding: padded B beats plain B ${pct(db.rate)} of persona votes (${perPersona}); the judge ${pct(dbJ.rate)}`)
  const abO = winRate(opusJs, "A", "B")
  const abD = winRate(dsJs, "A", "B")
  const abG = FAMILY === "gemini" ? `, the Gemini judge (wrote neither) ${pct(winRate(judgeJs, "A", "B").rate)}` : ""
  console.log(`  self-preference: the Opus judge prefers the Opus answer over DeepSeek's ${pct(abO.rate)}, the DeepSeek judge ${pct(abD.rate)} (${abD.n} items)${abG}`)
  console.log(`  outcome link (simulated): P(approve as is) = σ(${a0.toFixed(2)} + ${b1.toFixed(2)}·U); approved ${pct(rate(approve(false)))} when U ≥ 0, ${pct(rate(approve(true)))} when U < 0`)
  console.log(`  decision: DeepSeek beats or ties Opus in ${pct(bA.rate)} of persona votes (95% lower bound ${pct(lo!)}); non-inferior at δ = 5 points ${lo! >= 0.45 ? "YES" : "no"}; answers with a contradicted claim: Opus ${flagged("A")}, DeepSeek ${flagged("B")} of ${ids.length}\n`)
}

// Holm–Bonferroni over the six gains (shared and personal, three task kinds).
{
  const byP = [...tests].sort((a, b) => a.p - b.p)
  let running = 0
  const holm = byP.map((t, i) => ((running = Math.max(running, Math.min(1, (byP.length - i) * t.p))), { ...t, holm: running }))
  console.log(`p-values (raw / Holm over ${tests.length}): ${tests.map((t) => `${t.name} ${t.p.toFixed(3)}/${holm.find((h) => h.name === t.name)!.holm.toFixed(3)}`).join("; ")}\n`)
}

if (FAMILY === "gemini") {
  // The same rater, task and pair judged by both portrayals: how often do they pick the same side?
  const side = (j: Judgment) => (j.preferred === "tie" ? "tie" : j.preferred === "first" ? j.first : j.second)
  const cell = (j: Judgment) => `${j.rater}|${j.task}|${[j.first, j.second].sort().join("")}`
  const opusVotes = new Map(all.filter((j) => j.judge === "opus" && !j.repeat && j.rater !== "judge").map((j) => [cell(j), side(j)]))
  console.log("Opus and Gemini portrayals of the same persona on the same comparison (same side picked; ties count as a different side):")
  for (const kind of KINDS) {
    const ds = all.filter((j) => j.judge === "gemini" && !j.repeat && PERSONAS.includes(j.rater) && kindOf.get(j.task) === kind && opusVotes.has(cell(j)))
    const same = ds.filter((j) => opusVotes.get(cell(j)) === side(j)).length
    const [lo, hi] = ci(same, ds.length)
    console.log(`  ${kind.padEnd(14)} ${pct(same / ds.length)} of ${ds.length} (95% ${pct(lo!)}–${pct(hi!)})`)
  }
}
