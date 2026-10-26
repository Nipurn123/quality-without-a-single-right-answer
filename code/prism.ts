import { invert } from "../../src/brain/math.ts"

// The personal-taste term on real people: PRISM (Kirk et al., NeurIPS 2024 Datasets and Benchmarks),
// 1,500 participants from 75 countries who each held about six conversations with 21 models, scoring
// every response 1-100; each turn shows the same person two to four responses to the same prompt.
//
// Within a turn the prompt and the person are fixed, so centring the scores on the turn's mean
// removes how hard the prompt was and how lenient the person is (the β_u of the decomposition).
// What is left is the person's preference among responses:
//
//   s − s̄ = (θ̄ + δ_u)ᵀ (φ − φ̄) + ε        φ: style features and the model's one-hot
//          shared quality   personal taste
//
// θ̄ is one least-squares fit over everyone; δ_u a ridge fit per person on what θ̄ leaves (its prior
// width τ chosen by cross-validation). Tested on conversations the fit never saw: each person's
// conversations are split three ways and each third predicted from the other two. If taste is
// real and stable, a person's own δ_u predicts their held-out conversation better than θ̄ alone,
// and better than another random person's δ (the control). Then, decided before looking: people
// who say safety matters to them should penalise refusals and disclaimers less.
//
// People write their own prompts, so the answers a person scores depend on the person: rater and
// answer are not independent, and part of a person's "taste" could be the kind of conversation they
// chose. A second fit gives each conversation type (unguided, values guided, controversy guided) its
// own shared weights, so a person's δ must predict beyond the shared taste of their conversation type.
//
//   bun run experiments/subjectivity/prism.ts

const IN = new URL("./prism-data/", import.meta.url).pathname
interface Row { user_id: string; conversation_id: string; interaction_id: string; conversation_type: string; model_name: string; model_response: string; score: number }
interface Survey { user_id: string; stated_prefs: Record<string, number | null> }
const rows = (await Bun.file(`${IN}utterances.jsonl`).text()).trim().split("\n").map((l) => JSON.parse(l) as Row)
const survey = new Map((await Bun.file(`${IN}survey.jsonl`).text()).trim().split("\n").map((l) => JSON.parse(l) as Survey).map((s) => [s.user_id, s]))

const MODELS = [...new Set(rows.map((r) => r.model_name))].sort()
const STYLE = ["length (log chars)", "markdown items", "has a list", "asks the user a question", "refusal or disclaimer"]
const REFUSAL = /\b(I'?m sorry|I am sorry|I (?:cannot|can't|can not)|as an AI|I'?m not able|I am not able|I don't have (?:personal|opinions|feelings))/i
function features(text: string): number[] {
  const t = text ?? ""
  const style = [
    Math.log(1 + t.length),
    (t.match(/^\s*(#{1,6} |[-*•] |\d+[.)] )/gm) ?? []).length,
    /^\s*([-*•] |\d+[.)] )/m.test(t) ? 1 : 0,
    /\?\s*$/.test(t.trim()) ? 1 : 0,
    REFUSAL.test(t) ? 1 : 0,
  ]
  return style
}
const F_STYLE = STYLE.length
const F = F_STYLE + MODELS.length
const mi = new Map(MODELS.map((m, i) => [m, i]))

// Turn groups of two or more scored responses, centred.
type Item = { user: string; conv: string; type: string; x: number[]; y: number; group: string }
const groups = new Map<string, Row[]>()
for (const r of rows) if (r.score !== null && r.score !== undefined) groups.set(r.interaction_id, [...(groups.get(r.interaction_id) ?? []), r])
const items: Item[] = []
const rawStyle = new Map<Row, number[]>()
for (const g of groups.values()) for (const r of g) rawStyle.set(r, features(r.model_response))
// z-score the style features over all responses so the weights are comparable.
const mean = Array.from({ length: F_STYLE }, (_, j) => [...rawStyle.values()].reduce((s, v) => s + v[j]!, 0) / rawStyle.size)
const sd = Array.from({ length: F_STYLE }, (_, j) => Math.sqrt([...rawStyle.values()].reduce((s, v) => s + (v[j]! - mean[j]!) ** 2, 0) / rawStyle.size) || 1)
for (const [gid, g] of groups) {
  if (g.length < 2) continue
  const xs = g.map((r) => [...rawStyle.get(r)!.map((v, j) => (v - mean[j]!) / sd[j]!), ...MODELS.map((m) => (r.model_name === m ? 1 : 0))])
  const xbar = Array.from({ length: F }, (_, j) => xs.reduce((s, x) => s + x[j]!, 0) / xs.length)
  const ybar = g.reduce((s, r) => s + r.score, 0) / g.length
  g.forEach((r, i) => items.push({ user: r.user_id, conv: r.conversation_id, type: r.conversation_type, x: xs[i]!.map((v, j) => v - xbar[j]!), y: (r.score - ybar) / 100, group: gid }))
}
const users = [...new Set(items.map((i) => i.user))]
console.log(`${items.length} scored responses in ${new Set(items.map((i) => i.group)).size} turns of 2+ responses, from ${users.length} people; ${MODELS.length} models\n`)

// ── Fitting ──

const dot = (a: number[], b: number[]) => a.reduce((s, v, i) => s + v * b[i]!, 0)
function ridge(data: Item[], lambda: number, target: (it: Item) => number): number[] {
  const A = Array.from({ length: F }, (_, i) => Array.from({ length: F }, (_, j) => (i === j ? lambda : 0)))
  const b = new Array<number>(F).fill(0)
  for (const it of data) {
    const t = target(it)
    for (let i = 0; i < F; i++) {
      if (it.x[i] === 0) continue
      b[i]! += it.x[i]! * t
      for (let j = 0; j < F; j++) A[i]![j]! += it.x[i]! * it.x[j]!
    }
  }
  const inv = invert(A)
  return inv.map((row) => dot(row, b))
}

// Each person's conversations split three ways; fold f holds out every third conversation.
const convIndex = new Map<string, number>()
for (const u of users) [...new Set(items.filter((i) => i.user === u).map((i) => i.conv))].forEach((c, k) => convIndex.set(c, k % 3))
const byUser = new Map<string, Item[]>()
for (const it of items) byUser.set(it.user, [...(byUser.get(it.user) ?? []), it])

const TYPES = ["unguided", "values guided", "controversy guided"]
function crossValidate(lambdaUser: number, control: boolean, byType = false, rng = mulberry(7)) {
  const pred = { shared: new Map<Item, number>(), personal: new Map<Item, number>() }
  for (let f = 0; f < 3; f++) {
    const train = items.filter((i) => convIndex.get(i.conv) !== f)
    const pooled = ridge(train, 1, (it) => it.y)
    const perType = new Map(TYPES.map((t) => [t, byType ? ridge(train.filter((i) => i.type === t), 1, (it) => it.y) : pooled]))
    const theta = (it: Item) => perType.get(it.type) ?? pooled
    const deltas = new Map<string, number[]>()
    for (const u of users) {
      const own = byUser.get(u)!.filter((i) => convIndex.get(i.conv) !== f)
      deltas.set(u, own.length ? ridge(own, lambdaUser, (it) => it.y - dot(theta(it), it.x)) : new Array(F).fill(0))
    }
    // Control: each person gets another random person's δ.
    const donor = new Map(users.map((u) => [u, users[Math.floor(rng() * users.length)]!]))
    for (const it of items.filter((i) => convIndex.get(i.conv) === f)) {
      const d = deltas.get(control ? donor.get(it.user)! : it.user)!
      pred.shared.set(it, dot(theta(it), it.x))
      pred.personal.set(it, dot(theta(it), it.x) + dot(d, it.x))
    }
  }
  return pred
}
function mulberry(seed: number) {
  return () => {
    seed |= 0
    seed = (seed + 0x6d2b79f5) | 0
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}
/** Explained share of the held-out within-turn variance, and pair-order accuracy, per person (for a bootstrap). */
function score(pred: Map<Item, number>, subset: Item[] = items) {
  const sse = new Map<string, [number, number]>()
  for (const it of subset) {
    const e = sse.get(it.user) ?? [0, 0]
    e[0] += (it.y - pred.get(it)!) ** 2
    e[1] += it.y ** 2
    sse.set(it.user, e)
  }
  let right = 0
  let pairs = 0
  const inGroup = new Map<string, Item[]>()
  for (const it of subset) inGroup.set(it.group, [...(inGroup.get(it.group) ?? []), it])
  for (const g of inGroup.values()) for (let i = 0; i < g.length; i++) for (let j = i + 1; j < g.length; j++) {
    if (g[i]!.y === g[j]!.y) continue
    pairs++
    if (pred.get(g[i]!)! - pred.get(g[j]!)! > 0 === g[i]!.y > g[j]!.y) right++
  }
  return { sse, accuracy: right / pairs, pairs }
}
const r2 = (sse: Map<string, [number, number]>) => {
  let a = 0
  let b = 0
  for (const [x, y] of sse.values()) (a += x), (b += y)
  return 1 - a / b
}

// τ by cross-validation (λ = noise / τ²).
let best = { lambda: 0, r2: -Infinity }
for (const lambda of [3, 10, 30, 100, 300]) {
  const r = r2(score(crossValidate(lambda, false).personal).sse)
  if (r > best.r2) best = { lambda, r2: r }
}
const real = crossValidate(best.lambda, false)
const ctrl = crossValidate(best.lambda, true)
const sShared = score(real.shared)
const sPersonal = score(real.personal)
const sControl = score(ctrl.personal)

// Bootstrap over people for the gain from personal taste.
const rng = mulberry(11)
const gains: number[] = []
const ctrlGains: number[] = []
for (let b = 0; b < 1000; b++) {
  const pick = Array.from({ length: users.length }, () => users[Math.floor(rng() * users.length)]!)
  const sum = (m: Map<string, [number, number]>) => pick.reduce(([a, c], u) => [a + (m.get(u)?.[0] ?? 0), c + (m.get(u)?.[1] ?? 0)], [0, 0])
  const [s0, t0] = sum(sShared.sse)
  const [s1] = sum(sPersonal.sse)
  const [s2] = sum(sControl.sse)
  gains.push((s0 - s1) / t0)
  ctrlGains.push((s0 - s2) / t0)
}
const q = (xs: number[], p: number) => [...xs].sort((a, b) => a - b)[Math.floor(p * (xs.length - 1))]!
const pct = (x: number) => `${(100 * x).toFixed(1)}%`
console.log(`Held-out conversations (never seen by the fit), ${sShared.pairs} within-turn pairs:`)
console.log(`  shared quality only (θ̄):                variance explained ${pct(r2(sShared.sse))}, pairs ordered right ${pct(sShared.accuracy)}`)
console.log(`  + the person's own taste (θ̄ + δ_u):      variance explained ${pct(r2(sPersonal.sse))}, pairs ordered right ${pct(sPersonal.accuracy)}   (λ = ${best.lambda})`)
console.log(`  control, another person's taste:          variance explained ${pct(r2(sControl.sse))}, pairs ordered right ${pct(sControl.accuracy)}`)
console.log(`  gain from own taste: ${pct(q(gains, 0.5))} of variance (95% ${pct(q(gains, 0.025))} to ${pct(q(gains, 0.975))}); from a random person's: ${pct(q(ctrlGains, 0.5))} (95% ${pct(q(ctrlGains, 0.025))} to ${pct(q(ctrlGains, 0.975))})`)

// The same with shared weights per conversation type.
{
  const realT = crossValidate(best.lambda, false, true)
  const ctrlT = crossValidate(best.lambda, true, true)
  const a = score(realT.shared)
  const b = score(realT.personal)
  const c = score(ctrlT.personal)
  const brng = mulberry(11)
  const g: number[] = []
  const cg: number[] = []
  for (let k = 0; k < 1000; k++) {
    const pick = Array.from({ length: users.length }, () => users[Math.floor(brng() * users.length)]!)
    const sum = (m: Map<string, [number, number]>) => pick.reduce(([x, y], u) => [x + (m.get(u)?.[0] ?? 0), y + (m.get(u)?.[1] ?? 0)], [0, 0])
    const [s0, t0] = sum(a.sse)
    g.push((s0 - sum(b.sse)[0]!) / t0)
    cg.push((s0 - sum(c.sse)[0]!) / t0)
  }
  console.log(`  with shared weights per conversation type: shared ${pct(r2(a.sse))} → + own taste ${pct(r2(b.sse))}; gain ${pct(q(g, 0.5))} (95% ${pct(q(g, 0.025))} to ${pct(q(g, 0.975))}); another person's: ${pct(q(cg, 0.5))} (95% ${pct(q(cg, 0.025))} to ${pct(q(cg, 0.975))})`)
}

// Subjectivity per conversation type: the personal part's share of the explained within-turn differences.
console.log(`\nBy conversation type (held out):`)
for (const type of ["unguided", "values guided", "controversy guided"]) {
  const sub = items.filter((i) => i.type === type)
  const a = score(real.shared, sub)
  const b = score(real.personal, sub)
  const varShared = sub.reduce((s, i) => s + real.shared.get(i)! ** 2, 0)
  const varPersonal = sub.reduce((s, i) => s + (real.personal.get(i)! - real.shared.get(i)!) ** 2, 0)
  console.log(`  ${type.padEnd(19)} shared ${pct(r2(a.sse))} → +own taste ${pct(r2(b.sse))}; pairs ${pct(a.accuracy)} → ${pct(b.accuracy)}; personal share of predicted differences S ≈ ${(varPersonal / (varShared + varPersonal)).toFixed(2)}`)
}

// Shared weights on all data, and each person's δ on all of their data.
const thetaAll = ridge(items, 1, (it) => it.y)
console.log(`\nShared weights (points of the 100-point score per sd of the feature): ${STYLE.map((s, j) => `${s} ${(100 * thetaAll[j]!).toFixed(1)}`).join(", ")}`)
const topModels = MODELS.map((m, i) => ({ m, w: thetaAll[F_STYLE + i]! })).sort((a, b) => b.w - a.w)
console.log(`Shared model effects (best, worst): ${topModels.slice(0, 3).map((x) => `${x.m.split("/").at(-1)} ${(100 * x.w).toFixed(1)}`).join(", ")} … ${topModels.slice(-3).map((x) => `${x.m.split("/").at(-1)} ${(100 * x.w).toFixed(1)}`).join(", ")}`)

// Stated vs measured: refusal tolerance against stated importance of safety (and, for contrast, of creativity).
const refusalIdx = STYLE.indexOf("refusal or disclaimer")
const personal = users
  .map((u) => ({ u, d: ridge(byUser.get(u)!, best.lambda, (it) => it.y - dot(thetaAll, it.x)), prefs: survey.get(u)?.stated_prefs }))
  .filter((p) => p.prefs && typeof p.prefs.safety === "number" && byUser.get(p.u)!.some((i) => i.x[refusalIdx] !== 0))
const corr = (xs: number[], ys: number[]) => {
  const mx = xs.reduce((s, x) => s + x, 0) / xs.length
  const my = ys.reduce((s, y) => s + y, 0) / ys.length
  const c = xs.reduce((s, x, i) => s + (x - mx) * (ys[i]! - my), 0)
  return c / Math.sqrt(xs.reduce((s, x) => s + (x - mx) ** 2, 0) * ys.reduce((s, y) => s + (y - my) ** 2, 0))
}
const ciR = (r: number, n: number) => {
  const zr = Math.atanh(r)
  const h = 1.96 / Math.sqrt(n - 3)
  return [Math.tanh(zr - h), Math.tanh(zr + h)]
}
for (const pref of ["safety", "creativity"]) {
  const ps = personal.filter((p) => typeof p.prefs![pref] === "number")
  const r = corr(ps.map((p) => p.d[refusalIdx]!), ps.map((p) => p.prefs![pref] as number))
  const [lo, hi] = ciR(r, ps.length)
  console.log(`Measured refusal tolerance (δ on refusals) vs stated importance of ${pref}: r = ${r.toFixed(3)} (95% ${lo!.toFixed(3)} to ${hi!.toFixed(3)}, ${ps.length} people)`)
}
