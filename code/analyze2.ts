import { parseArgs } from "node:util"
import type { FactCheck, Judgment, Task, Variant } from "./run.ts"
import type { Rating } from "./run2.ts"

// Analysis of the extended controlled study (run.ts + run2.ts), for one rater family.
//
//   Pairwise: the model of analyze.ts (M0 correctness, M1 + shared quality, M2 + personal taste),
//   fitted per task kind on all 30 tasks of that kind, with the prior widths chosen by 5-fold
//   cross-validation on a grid that extends past analyze.ts's (whose τ = 3 was the grid's largest
//   value and was selected in four of six fits). S = τ² / (σ_q² + τ²), with a 95% interval from a
//   bootstrap over tasks that reselects the widths in every replicate.
//
//   Ratings: the crossed design of run2.ts --stage ratings (six personas × 120 answers per kind, 10%
//   of cells repeated). Two-way random-effects ANOVA gives Var(α), Var(β), Var(γ + ε); the repeats
//   give Var(ε); hence Var(γ) and S = Var(γ) / (Var(α) + Var(γ)) directly from Definition 2, with a
//   95% bootstrap interval over tasks (raters fixed: the six personas are the population).
//
//   --check reruns analyze.ts's grid on the first 30 tasks and must reproduce its published numbers.
//
//   bun run experiments/analyze2.ts --raters opus|gemini [--kind docs-qa] [--boot 100] [--check]

const { values } = parseArgs({ args: Bun.argv.slice(2), options: { raters: { type: "string", default: "opus" }, kind: { type: "string" }, boot: { type: "string", default: "0" }, check: { type: "boolean", default: false }, part: { type: "string", default: "all" }, source: { type: "string" } } })
// profile-s.ts imports this module for its fit and sets ANALYZE2_PART=none so nothing else runs.
const PART = process.env.ANALYZE2_PART ?? values.part
const FAMILY = values.raters === "gemini" ? "gemini" : "opus"
const HERE = new URL("./", import.meta.url).pathname
const D1 = `${HERE}data/`
const D2 = `${HERE}data2/`
const readJson = async <T,>(f: string, dflt: T): Promise<T> => ((await Bun.file(f).exists()) ? ((await Bun.file(f).json()) as T) : dflt)
const readLines = async <T,>(f: string): Promise<T[]> => ((await Bun.file(f).exists()) ? (await Bun.file(f).text()).trim().split("\n").filter(Boolean).map((l) => JSON.parse(l) as T) : [])

type T2 = Task & { product: string; source: "100xprompt-1" | "100xprompt-2" | "uv" }
const tasks: T2[] = [
  ...(await readJson<Task[]>(`${D1}tasks.json`, [])).map((t) => ({ ...t, product: "100xprompt", source: "100xprompt-1" as const })),
  ...(await readJson<Array<Task & { product: string }>>(`${D2}tasks.json`, [])).map((t) => ({ ...t, source: t.product === "uv" ? ("uv" as const) : ("100xprompt-2" as const) })),
]
const taskOf = new Map(tasks.map((t) => [t.id, t]))
const factsO = [...(await readJson<FactCheck[]>(`${D1}factchecks.json`, [])), ...(await readJson<FactCheck[]>(`${D2}factchecks.json`, []))]
const factsG = [...(await readJson<FactCheck[]>(`${D1}factchecks-gemini.json`, [])), ...(await readJson<FactCheck[]>(`${D2}factchecks-gemini.json`, []))]
const nContra = (F: FactCheck[]) => new Map(F.map((f) => [`${f.task}/${f.variant}`, f.claims.filter((c) => c.verdict === "contradicted").length]))
const cO = nContra(factsO)
const cG = nContra(factsG)
const judgments = [
  ...(await readLines<Judgment>(`${D1}judgments.jsonl`)),
  ...(await readLines<Judgment>(`${D2}judgments.jsonl`)),
  ...(FAMILY === "gemini" ? [...(await readLines<Judgment>(`${D1}judgments-gemini.jsonl`)), ...(await readLines<Judgment>(`${D2}judgments-gemini.jsonl`))] : []),
]
// A model occasionally returns the same item twice in one reply; keep the first rating of each cell.
const ratings = (await readLines<Rating>(`${D2}${FAMILY === "opus" ? "ratings.jsonl" : "ratings-gemini.jsonl"}`)).filter(
  (r, i, xs) => xs.findIndex((y) => y.batch === r.batch && y.rater === r.rater && y.task === r.task && y.variant === r.variant && y.repeat === r.repeat) === i,
)

const KINDS = ["docs-qa", "support-reply", "launch-post"] as const
const PERSONAS = ["priya", "marco", "aisha", "tom", "lena", "kenji"]
const V: Variant[] = ["A", "B", "C", "D"]
const sigmoid = (z: number) => 1 / (1 + Math.exp(-z))
const outcome = (j: Judgment) => (j.preferred === "first" ? 1 : j.preferred === "second" ? 0 : 0.5)
const pct = (x: number) => `${(100 * x).toFixed(0)}%`
const f2 = (x: number) => x.toFixed(2)
const results: Record<string, unknown> = { family: FAMILY }

// ── Fact checks: two graders ──

if (PART === "all" || PART === "facts") {
  const keys = tasks.flatMap((t) => V.map((v) => `${t.id}/${v}`)).filter((k) => cO.has(k) && cG.has(k))
  const agree = keys.filter((k) => cO.get(k)! > 0 === cG.get(k)! > 0).length
  const bySrc: Record<string, unknown> = {}
  for (const src of ["100xprompt-1", "100xprompt-2", "uv"]) {
    const ts = tasks.filter((t) => t.source === src)
    const flagged = (m: Map<string, number>, v: Variant) => ts.filter((t) => (m.get(`${t.id}/${v}`) ?? 0) > 0).length
    bySrc[src] = Object.fromEntries(V.map((v) => [v, { opus: flagged(cO, v), gemini: flagged(cG, v), of: ts.length }]))
  }
  results.facts = { answers: keys.length, agreeOnFlag: agree, bySource: bySrc }
  console.log(`fact checks: Opus and Gemini agree on whether an answer has a contradicted claim for ${agree}/${keys.length} answers`)
  console.log(`  answers with ≥1 contradicted claim (Opus/Gemini): ${JSON.stringify(bySrc)}\n`)
}

// ── Pairwise model, integer-indexed (same objective and Adam settings as analyze.ts) ──

type Obs = { u: number; a: number; b: number; y: number }
type Data = { obs: Obs[]; nA: number; nU: number; err: Float64Array }
type Model = "M0" | "M1" | "M2"

function fit(d: Data, idx: number[], model: Model, sq: number, tau: number): Float64Array {
  const P = model === "M0" ? 1 : d.nA + (model === "M2" ? d.nU * d.nA : 0)
  const th = new Float64Array(P)
  const m = new Float64Array(P)
  const v = new Float64Array(P)
  const g = new Float64Array(P)
  const pq = 1 / (sq * sq)
  const pd = tau > 0 ? 1 / (tau * tau) : 1e6
  for (let it = 1; it <= 1500; it++) {
    g.fill(0)
    for (const i of idx) {
      const o = d.obs[i]!
      let z: number
      if (model === "M0") z = th[0]! * (d.err[o.a]! - d.err[o.b]!)
      else {
        z = th[o.a]! - th[o.b]!
        if (model === "M2") z += th[d.nA + o.u * d.nA + o.a]! - th[d.nA + o.u * d.nA + o.b]!
      }
      const r = o.y - sigmoid(z)
      if (model === "M0") g[0]! += r * (d.err[o.a]! - d.err[o.b]!)
      else {
        g[o.a]! += r
        g[o.b]! -= r
        if (model === "M2") {
          g[d.nA + o.u * d.nA + o.a]! += r
          g[d.nA + o.u * d.nA + o.b]! -= r
        }
      }
    }
    const c1 = 1 - 0.9 ** it
    const c2 = 1 - 0.999 ** it
    for (let p = 0; p < P; p++) {
      const prior = model === "M0" ? 0.01 : p < d.nA ? pq : pd
      const gp = g[p]! - prior * th[p]!
      m[p] = 0.9 * m[p]! + 0.1 * gp
      v[p] = 0.999 * v[p]! + 0.001 * gp * gp
      th[p]! += (0.05 * (m[p]! / c1)) / (Math.sqrt(v[p]! / c2) + 1e-8)
    }
  }
  return th
}
function llOne(d: Data, th: Float64Array, model: Model, o: Obs): number {
  let z: number
  if (model === "M0") z = th[0]! * (d.err[o.a]! - d.err[o.b]!)
  else {
    z = th[o.a]! - th[o.b]!
    if (model === "M2") z += th[d.nA + o.u * d.nA + o.a]! - th[d.nA + o.u * d.nA + o.b]!
  }
  const p = Math.min(1 - 1e-6, Math.max(1e-6, sigmoid(z)))
  return o.y * Math.log(p) + (1 - o.y) * Math.log(1 - p)
}
function cv(d: Data, model: Model, sq: number, tau: number, fold: (i: number) => number = (i) => i % 5): number[] {
  const out = new Array<number>(d.obs.length).fill(0)
  for (let f = 0; f < 5; f++) {
    const tr: number[] = []
    const te: number[] = []
    d.obs.forEach((_, i) => (fold(i) === f ? te : tr).push(i))
    const th = fit(d, tr, model, sq, tau)
    for (const i of te) out[i] = llOne(d, th, model, d.obs[i]!)
  }
  return out
}
function cvRepeated(d: Data, model: Model, sq: number, tau: number): number[] {
  const out = new Array<number>(d.obs.length).fill(0)
  for (let r = 0; r < 5; r++) {
    let seed = 20261025 + r
    const rand = () => ((seed = (seed * 1103515245 + 12345) % 2147483648), seed / 2147483648)
    const keys = d.obs.map(() => rand())
    const fold = new Array<number>(d.obs.length)
    d.obs.map((_, i) => i).sort((p, q) => keys[p]! - keys[q]!).forEach((i, k) => (fold[i] = k % 5))
    cv(d, model, sq, tau, (i) => fold[i]!).forEach((x, i) => (out[i]! += x / 5))
  }
  return out
}
const sum = (xs: number[]) => xs.reduce((s, x) => s + x, 0)
const meanSe = (xs: number[]) => {
  const m = sum(xs) / xs.length
  const sd = Math.sqrt(xs.reduce((s, x) => s + (x - m) ** 2, 0) / (xs.length - 1))
  return { m, se: sd / Math.sqrt(xs.length) }
}
const erf = (x: number) => {
  const t = 1 / (1 + 0.3275911 * Math.abs(x))
  const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x)
  return x >= 0 ? y : -y
}
const pOf = (g: { m: number; se: number }) => (g.se > 0 ? 1 - erf(Math.abs(g.m / g.se) / Math.SQRT2) : 1)

const GRID_ORIG = { sq: [0.5, 1, 2, 3, 5], mult: [0.5, 1, 1.5], tau: [0.25, 0.5, 1, 1.5, 2, 3] }
const GRID_WIDE = { sq: [0.5, 1, 2, 3, 5, 8], mult: [0.5, 1, 1.5, 2], tau: [0.25, 0.5, 1, 1.5, 2, 3, 4, 6, 9] }
type Grid = typeof GRID_ORIG

function select(d: Data, grid: Grid) {
  let b1 = { sq: 1, l: -Infinity }
  for (const sq of grid.sq) {
    const l = sum(cv(d, "M1", sq, 0))
    if (l > b1.l) b1 = { sq, l }
  }
  let b2 = { sq: b1.sq, tau: 0, l: -Infinity }
  for (const k of grid.mult) for (const tau of grid.tau) {
    const l = sum(cv(d, "M2", b1.sq * k, tau))
    if (l > b2.l) b2 = { sq: b1.sq * k, tau, l }
  }
  const atEdge = b2.tau === grid.tau[grid.tau.length - 1] || b2.sq === b1.sq * grid.mult[grid.mult.length - 1]! || b1.sq === grid.sq[grid.sq.length - 1]
  return { sq1: b1.sq, sq: b2.sq, tau: b2.tau, S: b2.tau ** 2 / (b2.sq ** 2 + b2.tau ** 2), atEdge }
}

/** Persona judgments of one kind on a set of tasks (a task may appear more than once in a bootstrap draw). */
function build(taskIds: string[], js: Judgment[]): Data {
  const aIndex = new Map<string, number>()
  const err: number[] = []
  const per = new Map<string, Judgment[]>()
  for (const j of js) per.set(j.task, [...(per.get(j.task) ?? []), j])
  const obs: Obs[] = []
  taskIds.forEach((t, copy) => {
    for (const v of V) {
      aIndex.set(`${copy}|${v}`, err.length)
      err.push(((cO.get(`${t}/${v}`) ?? 0) + (cG.get(`${t}/${v}`) ?? cO.get(`${t}/${v}`) ?? 0)) / 2)
    }
    for (const j of per.get(t) ?? []) obs.push({ u: PERSONAS.indexOf(j.rater), a: aIndex.get(`${copy}|${j.first}`)!, b: aIndex.get(`${copy}|${j.second}`)!, y: outcome(j) })
  })
  return { obs, nA: err.length, nU: PERSONAS.length, err: Float64Array.from(err) }
}
/** analyze.ts's observation order (judgments in file order), for --check. */
function buildOrig(kind: string, js: Judgment[]): Data {
  const ids = tasks.filter((t) => t.source === "100xprompt-1" && t.kind === kind).map((t) => t.id)
  const answers = ids.flatMap((t) => V.map((v) => `${t}/${v}`))
  const ai = new Map(answers.map((a, i) => [a, i]))
  const obs = js.filter((j) => ids.includes(j.task)).map((j) => ({ u: PERSONAS.indexOf(j.rater), a: ai.get(`${j.task}/${j.first}`)!, b: ai.get(`${j.task}/${j.second}`)!, y: outcome(j) }))
  // analyze.ts indexes personal terms by the raters present in training data; with all six present the order is immaterial.
  return { obs, nA: answers.length, nU: PERSONAS.length, err: Float64Array.from(answers.map((a) => cO.get(a) ?? 0)) }
}
const personaJs = (kind: string) => judgments.filter((j) => j.judge === FAMILY && PERSONAS.includes(j.rater) && !j.repeat && taskOf.get(j.task)?.kind === kind)

function gains(d: Data, sel: { sq1: number; sq: number; tau: number }) {
  const l0 = cvRepeated(d, "M0", 1, 0)
  const l1 = cvRepeated(d, "M1", sel.sq1, 0)
  const l2 = cvRepeated(d, "M2", sel.sq, sel.tau)
  const d10 = meanSe(l1.map((x, i) => x - l0[i]!))
  const d21 = meanSe(l2.map((x, i) => x - l1[i]!))
  return { ll: [meanSe(l0).m, meanSe(l1).m, meanSe(l2).m], shared: { ...d10, p: pOf(d10) }, personal: { ...d21, p: pOf(d21) } }
}

const kinds = values.kind ? [values.kind] : [...KINDS]

/** Cross-validated M2 log-likelihood per judgment on a grid of total width w and S = τ²/w² (profile-s.ts). */
export function profile(kind: string, source: string, widths: number[], Ss: number[]) {
  const ids = tasks.filter((t) => t.kind === kind && (source === "all" || t.source === source)).map((t) => t.id)
  const d = build(ids, personaJs(kind))
  return widths.flatMap((w) => Ss.map((S) => ({ w, S, ll: sum(cv(d, "M2", w * Math.sqrt(1 - S), w * Math.sqrt(S))) / d.obs.length })))
}

if (values.check) {
  console.log(`check: analyze.ts grid on the first 30 tasks (${FAMILY} raters)`)
  for (const kind of kinds) {
    const d = buildOrig(kind, personaJs(kind))
    const sel = select(d, GRID_ORIG)
    const g = gains(d, sel)
    console.log(`  ${kind}: ${d.obs.length} judgments; σ_q ${sel.sq}, τ ${sel.tau}, S ${f2(sel.S)}; shared +${g.shared.m.toFixed(3)} ± ${(1.96 * g.shared.se).toFixed(3)}, personal +${g.personal.m.toFixed(3)} ± ${(1.96 * g.personal.se).toFixed(3)}`)
  }
  process.exit(0)
}

// ── Pairwise results on all tasks ──

if (PART === "all" || PART === "pairs") {
  const pr: Record<string, unknown> = {}
  const tests: Array<{ name: string; p: number }> = []
  for (const kind of kinds) {
    const js = personaJs(kind)
    const ids = tasks.filter((t) => t.kind === kind).map((t) => t.id)
    const d = build(ids, js)
    const sel = select(d, GRID_WIDE)
    const g = gains(d, sel)
    tests.push({ name: `${kind} shared`, p: g.shared.p }, { name: `${kind} personal`, p: g.personal.p })
    // S by source, same grid.
    const bySource = Object.fromEntries(["100xprompt-1", "100xprompt-2", "uv"].map((src) => {
      const s = select(build(tasks.filter((t) => t.kind === kind && t.source === src).map((t) => t.id), js), GRID_WIDE)
      return [src, { S: s.S, sq: s.sq, tau: s.tau, atEdge: s.atEdge }]
    }))
    // Bootstrap over tasks, reselecting the widths each time.
    const B = Number(values.boot)
    const Sb: number[] = []
    let seed = 7
    const rand = () => ((seed = (seed * 1103515245 + 12345) % 2147483648), seed / 2147483648)
    for (let b = 0; b < B; b++) {
      const draw = ids.map(() => ids[Math.floor(rand() * ids.length)]!)
      Sb.push(select(build(draw, js), GRID_WIDE).S)
    }
    Sb.sort((x, y) => x - y)
    const ciS = B ? [Sb[Math.floor(0.025 * B)]!, Sb[Math.min(B - 1, Math.ceil(0.975 * B) - 1)]!] : null

    // Descriptives, as in analyze.ts.
    const all = judgments.filter((j) => j.judge === FAMILY && PERSONAS.includes(j.rater) && taskOf.get(j.task)?.kind === kind)
    const vote = (j: Judgment) => (j.preferred === "tie" ? 0.5 : (j.preferred === "first") === (j.first < j.second) ? 1 : 0)
    const byItem = new Map<string, number[]>()
    for (const j of all.filter((j) => !j.repeat)) {
      const k = `${j.task}|${[j.first, j.second].sort().join("")}`
      byItem.set(k, [...(byItem.get(k) ?? []), vote(j)])
    }
    let dis = 0
    let pairs = 0
    for (const vs of byItem.values()) for (let i = 0; i < vs.length; i++) for (let k = i + 1; k < vs.length; k++) (dis += Math.abs(vs[i]! - vs[k]!)), pairs++
    let wdis = 0
    let wn = 0
    for (const r of all.filter((j) => j.repeat)) {
      const o = all.find((j) => !j.repeat && j.rater === r.rater && j.task === r.task && [j.first, j.second].sort().join("") === [r.first, r.second].sort().join(""))
      if (o) (wdis += Math.abs(vote(o) - vote(r))), wn++
    }
    let triads = 0
    let cyclic = 0
    for (const u of PERSONAS) for (const t of ids) {
      const beats = new Set<string>()
      for (const j of all.filter((j) => !j.repeat && j.rater === u && j.task === t && j.preferred !== "tie")) beats.add(j.preferred === "first" ? `${j.first}>${j.second}` : `${j.second}>${j.first}`)
      for (let a = 0; a < 4; a++) for (let b = a + 1; b < 4; b++) for (let c = b + 1; c < 4; c++) {
        const [x, y, z] = [V[a]!, V[b]!, V[c]!]
        const has = (p: string, q: string) => beats.has(`${p}>${q}`) || beats.has(`${q}>${p}`)
        if (!has(x, y) || !has(y, z) || !has(x, z)) continue
        triads++
        if ((beats.has(`${x}>${y}`) && beats.has(`${y}>${z}`) && beats.has(`${z}>${x}`)) || (beats.has(`${y}>${x}`) && beats.has(`${z}>${y}`) && beats.has(`${x}>${z}`))) cyclic++
      }
    }
    const winRate = (js: Judgment[], w: Variant, l: Variant) => {
      const rel = js.filter((j) => !j.repeat && [j.first, j.second].sort().join("") === [w, l].sort().join(""))
      const wins = rel.reduce((s, j) => s + (j.preferred === "tie" ? 0.5 : (j.preferred === "first" ? j.first : j.second) === w ? 1 : 0), 0)
      return { wins, n: rel.length, rate: rel.length ? wins / rel.length : NaN }
    }
    const wilsonLo = (k: number, n: number) => {
      const p = k / n
      const z = 1.96
      return ((p + (z * z) / (2 * n)) - z * Math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n))) / (1 + (z * z) / n)
    }
    const judgeOf = (fam: string) => judgments.filter((j) => j.judge === fam && j.rater === "judge" && taskOf.get(j.task)?.kind === kind)
    const bA = winRate(all, "B", "A")
    const bySrcDecision = Object.fromEntries(["100xprompt-1", "100xprompt-2", "uv"].map((src) => [src, winRate(all.filter((j) => taskOf.get(j.task)!.source === src), "B", "A").rate]))
    const out = {
      tasks: ids.length,
      judgments: d.obs.length,
      sel,
      gains: g,
      S: sel.S,
      ciS,
      bySource,
      disagreeBetween: dis / pairs,
      disagreeWithin: wn ? wdis / wn : null,
      repeats: wn,
      triads,
      cyclic,
      cleanBeatsError: winRate(all, "A", "C").rate,
      paddedBeatsPlain: winRate(all, "D", "B").rate,
      paddedByPersona: Object.fromEntries(PERSONAS.map((u) => [u, winRate(all.filter((j) => j.rater === u), "D", "B").rate])),
      judgeOpusPrefersA: winRate(judgeOf("opus"), "A", "B").rate,
      judgeDeepseekPrefersA: winRate(judgeOf("deepseek"), "A", "B").rate,
      judgeGeminiPrefersA: winRate(judgeOf("gemini"), "A", "B").rate,
      cheaperPreferredOrTied: bA.rate,
      cheaperLowerBound: wilsonLo(bA.wins, bA.n),
      cheaperBySource: bySrcDecision,
    }
    pr[kind] = out
    console.log(`══ ${kind}: ${ids.length} tasks, ${d.obs.length} persona judgments (${FAMILY}) ══`)
    console.log(`  held-out ll: M0 ${g.ll[0]!.toFixed(3)}, M1 ${g.ll[1]!.toFixed(3)}, M2 ${g.ll[2]!.toFixed(3)}; shared +${g.shared.m.toFixed(3)} ± ${(1.96 * g.shared.se).toFixed(3)} (p ${g.shared.p.toFixed(4)}), personal +${g.personal.m.toFixed(3)} ± ${(1.96 * g.personal.se).toFixed(3)} (p ${g.personal.p.toFixed(4)})`)
    console.log(`  σ_q ${sel.sq}, τ ${sel.tau}${sel.atEdge ? " (AT GRID EDGE)" : ""} → S ${f2(sel.S)}${ciS ? ` (95% bootstrap over tasks ${f2(ciS[0]!)} to ${f2(ciS[1]!)}, B = ${B})` : ""}; by source ${Object.entries(bySource).map(([k, v]) => `${k} ${f2(v.S)}${v.atEdge ? "*" : ""}`).join(", ")}`)
    console.log(`  disagreement between personas ${pct(out.disagreeBetween)}, within on repeats ${out.disagreeWithin === null ? "n/a" : pct(out.disagreeWithin)} (${wn}); cyclic triads ${cyclic}/${triads}`)
    console.log(`  clean A beats one-error C ${pct(out.cleanBeatsError)}; padded D beats B ${pct(out.paddedBeatsPlain)} (${Object.entries(out.paddedByPersona).map(([u, r]) => `${u} ${pct(r)}`).join(", ")})`)
    console.log(`  judges prefer A over B: Opus ${pct(out.judgeOpusPrefersA)}, DeepSeek ${pct(out.judgeDeepseekPrefersA)}, Gemini ${Number.isNaN(out.judgeGeminiPrefersA) ? "n/a" : pct(out.judgeGeminiPrefersA)}`)
    console.log(`  cheaper model preferred or tied ${pct(bA.rate)} (95% lower bound ${pct(out.cheaperLowerBound)}); by source ${Object.entries(bySrcDecision).map(([k, r]) => `${k} ${pct(r)}`).join(", ")}\n`)
  }
  const byP = [...tests].sort((a, b) => a.p - b.p)
  let running = 0
  const holm = Object.fromEntries(byP.map((t, i) => [t.name, (running = Math.max(running, Math.min(1, (byP.length - i) * t.p)))]))
  console.log(`p (raw / Holm over ${tests.length}): ${tests.map((t) => `${t.name} ${t.p.toFixed(4)}/${holm[t.name]!.toFixed(4)}`).join("; ")}\n`)
  results.pairs = { ...pr, holm }
}

// ── Crossed ratings ──
//
// Raters are crossed with answers, and answers are nested in tasks (four per task), so the ratings
// follow a nested random-effects model,
//   Y_uta = μ + T_t + α_a(t) + β_u + (βT)_ut + γ_ua(t) + ε,
// with T the task, α_a(t) shared quality within a task, β leniency, (βT) a rater's liking for a task as
// a whole, γ personal taste for an answer within its task and ε noise (from the repeats). S within a
// request, Var(γ) / (Var(α) + Var(γ)), is the index of Definition 2 for a fixed request averaged over
// requests, and the quantity pairwise judgments see, since comparisons are always within a task. S over
// the task type adds the task-level terms: (Var(γ) + Var(βT)) / (Var(T) + Var(α) + Var(βT) + Var(γ)).
// The impartial judge is an instrument, not a member of the population, so only the six personas
// enter. Intervals resample tasks with the six personas fixed: they are conditional on this population.

if ((PART === "all" || PART === "ratings") && ratings.length) {
  const rr: Record<string, unknown> = {}
  for (const kind of kinds) {
    const ids = tasks.filter((t) => t.kind === kind).map((t) => t.id)
    const R = ratings.filter((r) => PERSONAS.includes(r.rater) && taskOf.get(r.task)?.kind === kind)
    const first = new Map<string, number>()
    const rep: Array<{ rater: string; task: string; variant: Variant; x: number; y: number }> = []
    for (const r of R.filter((r) => !r.repeat)) first.set(`${r.rater}|${r.task}/${r.variant}`, r.rating)
    for (const r of R.filter((r) => r.repeat)) {
      const f = first.get(`${r.rater}|${r.task}/${r.variant}`)
      if (f !== undefined) rep.push({ rater: r.rater, task: r.task, variant: r.variant, x: f, y: r.rating })
    }
    const complete = (t: string) => V.every((v) => PERSONAS.every((u) => first.has(`${u}|${t}/${v}`)))
    const okIds = ids.filter(complete)
    type Rep = (typeof rep)[number]
    function nested(taskIds: string[], vs: Variant[] = V, F: Map<string, number> = first, RP: Rep[] = rep) {
      const nU = PERSONAS.length
      const nT = taskIds.length
      const nA = vs.length
      const Y = PERSONAS.map((u) => taskIds.map((t) => vs.map((v) => F.get(`${u}|${t}/${v}`)!)))
      let grand = 0
      const mT = new Array<number>(nT).fill(0)
      const mU = new Array<number>(nU).fill(0)
      const mA = taskIds.map(() => new Array<number>(nA).fill(0))
      const mUT = PERSONAS.map(() => new Array<number>(nT).fill(0))
      for (let u = 0; u < nU; u++) for (let t = 0; t < nT; t++) for (let a = 0; a < nA; a++) {
        const y = Y[u]![t]![a]!
        grand += y / (nU * nT * nA)
        mT[t]! += y / (nU * nA)
        mU[u]! += y / (nT * nA)
        mA[t]![a]! += y / nU
        mUT[u]![t]! += y / nA
      }
      let ssT = 0, ssA = 0, ssU = 0, ssUT = 0, ssUA = 0
      for (let t = 0; t < nT; t++) {
        ssT += nU * nA * (mT[t]! - grand) ** 2
        for (let a = 0; a < nA; a++) ssA += nU * (mA[t]![a]! - mT[t]!) ** 2
      }
      for (let u = 0; u < nU; u++) {
        ssU += nT * nA * (mU[u]! - grand) ** 2
        for (let t = 0; t < nT; t++) {
          ssUT += nA * (mUT[u]![t]! - mU[u]! - mT[t]! + grand) ** 2
          for (let a = 0; a < nA; a++) ssUA += (Y[u]![t]![a]! - mUT[u]![t]! - mA[t]![a]! + mT[t]!) ** 2
        }
      }
      const msT = ssT / (nT - 1)
      const msA = ssA / (nT * (nA - 1))
      const msU = ssU / (nU - 1)
      const msUT = ssUT / ((nU - 1) * (nT - 1))
      const msUA = ssUA / ((nU - 1) * nT * (nA - 1))
      // Noise from the repeated ratings of the drawn tasks (each task weighted by how often it was drawn).
      const mult = new Map<string, number>()
      for (const t of taskIds) mult.set(t, (mult.get(t) ?? 0) + 1)
      let sE = 0
      let nE = 0
      for (const r of RP) {
        if (!vs.includes(r.variant)) continue
        const m = mult.get(r.task) ?? 0
        sE += m * (r.x - r.y) ** 2
        nE += m
      }
      const vE = nE ? sE / (2 * nE) : NaN
      const vG = Math.max(0, msUA - vE)
      const vA = Math.max(0, (msA - msUA) / nU)
      const vBT = Math.max(0, (msUT - msUA) / nA)
      const vB = Math.max(0, (msU - msUT) / (nT * nA))
      const vT = Math.max(0, (msT - msA - msUT + msUA) / (nU * nA))
      const S = vG / Math.max(1e-12, vA + vG)
      const Stype = (vG + vBT) / Math.max(1e-12, vT + vA + vBT + vG)
      // Untruncated moment estimates of the task-level terms, to show how far below zero they fall.
      const raw = { task: (msT - msA - msUT + msUA) / (nU * nA), raterTask: (msUT - msUA) / nA }
      return { vT, vA, vB, vBT, vG, vE, S, Stype, mA, mT, F: msUA / vE, raw }
    }
    // Scale use. S is invariant to a rater's shift (leniency) but not to a rater's stretch: a persona who
    // uses the scale more extremely while ranking answers alike adds to Var(γ). Two corrections, each
    // recomputed on the drawn tasks: "z" standardises every rater's ratings, "slope" divides them by the
    // rater's slope on the answer means (the stretch along shared quality).
    function rescale(taskIds: string[], mode: "z" | "slope") {
      const F = new Map<string, number>()
      const RP: Rep[] = []
      const mean = new Map<string, number>()
      if (mode === "slope") for (const t of new Set(taskIds)) for (const v of V) mean.set(`${t}/${v}`, sum(PERSONAS.map((u) => first.get(`${u}|${t}/${v}`)!)) / PERSONAS.length)
      const factor = new Map<string, { c: number; k: number }>()
      for (const u of PERSONAS) {
        const ys = taskIds.flatMap((t) => V.map((v) => first.get(`${u}|${t}/${v}`)!))
        const my = sum(ys) / ys.length
        let k: number
        if (mode === "z") k = Math.sqrt(sum(ys.map((y) => (y - my) ** 2)) / (ys.length - 1))
        else {
          const ms = taskIds.flatMap((t) => V.map((v) => mean.get(`${t}/${v}`)!))
          const mm = sum(ms) / ms.length
          k = sum(ys.map((y, i) => (y - my) * (ms[i]! - mm))) / Math.max(1e-12, sum(ms.map((x) => (x - mm) ** 2)))
        }
        factor.set(u, { c: my, k })
        for (const t of new Set(taskIds)) for (const v of V) F.set(`${u}|${t}/${v}`, (first.get(`${u}|${t}/${v}`)! - my) / k)
      }
      for (const r of rep) {
        const f = factor.get(r.rater)!
        RP.push({ ...r, x: (r.x - f.c) / f.k, y: (r.y - f.c) / f.k })
      }
      return { F, RP, factor }
    }
    const est = nested(okIds)
    const nat = nested(okIds, ["A", "B"])
    const zFull = rescale(okIds, "z")
    const sFull = rescale(okIds, "slope")
    const estZ = nested(okIds, V, zFull.F, zFull.RP)
    const estSlope = nested(okIds, V, sFull.F, sFull.RP)
    // Share of within-task shared quality explained by correctness (contradicted claim, either grader).
    let sxy = 0, sxx = 0, syy = 0
    okIds.forEach((t, ti) => {
      const c = V.map((v) => (((cO.get(`${t}/${v}`) ?? 0) + (cG.get(`${t}/${v}`) ?? 0)) / 2 > 0 ? 1 : 0))
      const cm = sum(c) / 4
      V.forEach((_, a) => {
        const x = c[a]! - cm
        const y = est.mA[ti]![a]! - est.mT[ti]!
        sxy += x * y
        sxx += x * x
        syy += y * y
      })
    })
    const r2c = sxx > 0 && syy > 0 ? (sxy * sxy) / (sxx * syy) : 0
    const B = 2000
    let seed = 11
    const rand = () => ((seed = (seed * 1103515245 + 12345) % 2147483648), seed / 2147483648)
    const boots = { S: [] as number[], Stype: [] as number[], Snat: [] as number[], vB: [] as number[], vG: [] as number[], vBT: [] as number[], Sz: [] as number[], Sslope: [] as number[] }
    for (let b = 0; b < B; b++) {
      const draw = okIds.map(() => okIds[Math.floor(rand() * okIds.length)]!)
      const e = nested(draw)
      boots.S.push(e.S)
      boots.Stype.push(e.Stype)
      boots.vB.push(e.vB)
      boots.vG.push(e.vG)
      boots.vBT.push(e.vBT)
      boots.Snat.push(nested(draw, ["A", "B"]).S)
      const z = rescale(draw, "z")
      boots.Sz.push(nested(draw, V, z.F, z.RP).S)
      const sl = rescale(draw, "slope")
      boots.Sslope.push(nested(draw, V, sl.F, sl.RP).S)
    }
    const q = (xs: number[]) => {
      const s = [...xs].sort((x, y) => x - y)
      return [s[Math.floor(0.025 * s.length)]!, s[Math.ceil(0.975 * s.length) - 1]!]
    }
    const exactRepeat = rep.filter((r) => r.x === r.y).length / Math.max(1, rep.length)
    const bySource = Object.fromEntries(["100xprompt-1", "100xprompt-2", "uv"].map((src) => [src, nested(okIds.filter((t) => taskOf.get(t)!.source === src)).S]))
    rr[kind] = {
      tasksComplete: okIds.length, tasks: ids.length, ratings: R.length, repeats: rep.length,
      var: { task: est.vT, alpha: est.vA, leniency: est.vB, raterTask: est.vBT, taste: est.vG, noise: est.vE },
      S: est.S, ciS: q(boots.S), Stype: est.Stype, ciStype: q(boots.Stype), Snatural: nat.S, ciSnatural: q(boots.Snat),
      ciLeniency: q(boots.vB), ciTaste: q(boots.vG), ciRaterTask: q(boots.vBT), Ftaste: est.F,
      exactRepeat, correctnessShareOfAlpha: r2c, bySource, raw: est.raw,
      Sz: estZ.S, ciSz: q(boots.Sz), Sslope: estSlope.S, ciSslope: q(boots.Sslope),
      raterSd: Object.fromEntries([...zFull.factor].map(([u, f]) => [u, f.k])), raterSlope: Object.fromEntries([...sFull.factor].map(([u, f]) => [u, f.k])),
    }
    console.log(`══ ratings, ${kind}: ${okIds.length}/${ids.length} tasks complete, ${R.length} ratings, ${rep.length} repeats (${FAMILY}) ══`)
    console.log(`  Var task ${f2(est.vT)}, Var α within task ${f2(est.vA)}, Var β leniency ${f2(est.vB)} (95% ${q(boots.vB).map(f2).join(" to ")}), Var rater×task ${f2(est.vBT)} (95% ${q(boots.vBT).map(f2).join(" to ")}), Var γ taste ${f2(est.vG)} (95% ${q(boots.vG).map(f2).join(" to ")}), Var ε noise ${f2(est.vE)}; F(taste vs noise) ${est.F.toFixed(2)}`)
    console.log(`  S within a request ${f2(est.S)} (95% ${q(boots.S).map(f2).join(" to ")}); natural answers A, B only ${f2(nat.S)} (95% ${q(boots.Snat).map(f2).join(" to ")}); over the task type ${f2(est.Stype)} (95% ${q(boots.Stype).map(f2).join(" to ")})`)
    console.log(`  S within a request by source ${Object.entries(bySource).map(([k, v]) => `${k} ${f2(v as number)}`).join(", ")}; repeats identical ${pct(exactRepeat)}; correctness explains ${pct(r2c)} of within-task Var(α)`)
    console.log(`  untruncated: Var task ${est.raw.task.toFixed(3)}, Var rater×task ${est.raw.raterTask.toFixed(3)}`)
    console.log(`  scale use: S standardised per rater ${f2(estZ.S)} (95% ${q(boots.Sz).map(f2).join(" to ")}); S after dividing by each rater's slope ${f2(estSlope.S)} (95% ${q(boots.Sslope).map(f2).join(" to ")}); rater SDs ${[...zFull.factor].map(([u, f]) => `${u} ${f.k.toFixed(2)}`).join(", ")}; slopes ${[...sFull.factor].map(([u, f]) => `${u} ${f.k.toFixed(2)}`).join(", ")}\n`)
  }
  results.ratings = rr
}

const OUT = `${HERE}outputs2/`
if (PART !== "none") await Bun.write(`${OUT}analyze2-${FAMILY}${values.kind ? `-${values.kind}` : ""}${PART !== "all" ? `-${PART}` : ""}.json`, JSON.stringify(results, null, 1))
