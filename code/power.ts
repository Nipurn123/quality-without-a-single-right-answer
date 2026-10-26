import { mulberry32 } from "./llm.ts"

// How large a human panel must be. Takes the variance components the crossed rating arm measured
// (outputs2/analyze2-<family>-ratings.json) as scenarios, simulates a panel of R raters rating T tasks
// with four answers each once (a share of cells twice) from the nested model, with new raters and tasks
// in every simulated panel, estimates the components by the same nested ANOVA as analyze2.ts, and
// reports per design: the 95% range of the estimated S within a request, and the power of two 5%-level
// F tests: taste (Var(γ) > 0; rater × answer mean square over the noise from repeats) and leniency
// (Var(β) > 0; rater mean square over rater × task mean square), with critical values simulated under
// each null.
//
//   bun run experiments/power.ts

const HERE = new URL("./", import.meta.url).pathname
// Each scenario and design draws from its own seed, so adding scenarios does not move the others.
let rng = mulberry32(20261101)
const seedOf = (key: string) => [...key].reduce((h, ch) => (Math.imul(h, 31) + ch.charCodeAt(0)) | 0, 20261101)
const normal = () => {
  const u = 1 - rng()
  const v = rng()
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v)
}
type Comp = { task: number; alpha: number; leniency: number; raterTask: number; taste: number; noise: number }

/**
 * One simulated panel: R raters rate T tasks × 4 answers once each (a share `repeatShare` of cells twice),
 * from the nested model of analyze2.ts with fresh raters and tasks every time. Returns S within a request
 * from the same nested ANOVA, the F statistic for taste (rater × answer mean square over the noise
 * from repeats) and for leniency (rater mean square over rater × task mean square).
 */
function simulate(c: Comp, R: number, T: number, repeatShare: number) {
  const nA = 4
  const sd = (v: number) => Math.sqrt(Math.max(0, v))
  const tEff = Array.from({ length: T }, () => sd(c.task) * normal())
  const aEff = Array.from({ length: T }, () => Array.from({ length: nA }, () => sd(c.alpha) * normal()))
  const bEff = Array.from({ length: R }, () => sd(c.leniency) * normal())
  const btEff = Array.from({ length: R }, () => Array.from({ length: T }, () => sd(c.raterTask) * normal()))
  const gEff = Array.from({ length: R }, () => Array.from({ length: T }, () => Array.from({ length: nA }, () => sd(c.taste) * normal())))
  const mean = (u: number, t: number, a: number) => tEff[t]! + aEff[t]![a]! + bEff[u]! + btEff[u]![t]! + gEff[u]![t]![a]!
  const Y = Array.from({ length: R }, (_, u) => Array.from({ length: T }, (_, t) => Array.from({ length: nA }, (_, a) => mean(u, t, a) + sd(c.noise) * normal())))
  // Repeated cells, drawn without replacement.
  const N = R * T * nA
  const cells = Array.from({ length: N }, (_, i) => i)
  const nRep = Math.min(N, Math.round(repeatShare * N))
  for (let i = 0; i < nRep; i++) {
    const j = i + Math.floor(rng() * (N - i))
    ;[cells[i], cells[j]] = [cells[j]!, cells[i]!]
  }
  let ssE = 0
  for (let i = 0; i < nRep; i++) {
    const u = Math.floor(cells[i]! / (T * nA))
    const t = Math.floor((cells[i]! % (T * nA)) / nA)
    const a = cells[i]! % nA
    ssE += (Y[u]![t]![a]! - (mean(u, t, a) + sd(c.noise) * normal())) ** 2
  }
  const vE = ssE / (2 * nRep)
  let grand = 0
  const mT = new Array<number>(T).fill(0)
  const mU = new Array<number>(R).fill(0)
  const mA = Array.from({ length: T }, () => new Array<number>(nA).fill(0))
  const mUT = Array.from({ length: R }, () => new Array<number>(T).fill(0))
  for (let u = 0; u < R; u++) for (let t = 0; t < T; t++) for (let a = 0; a < nA; a++) {
    const y = Y[u]![t]![a]!
    grand += y / N
    mT[t]! += y / (R * nA)
    mU[u]! += y / (T * nA)
    mA[t]![a]! += y / R
    mUT[u]![t]! += y / nA
  }
  let ssA = 0, ssU = 0, ssUT = 0, ssUA = 0
  for (let t = 0; t < T; t++) for (let a = 0; a < nA; a++) ssA += R * (mA[t]![a]! - mT[t]!) ** 2
  for (let u = 0; u < R; u++) {
    ssU += T * nA * (mU[u]! - grand) ** 2
    for (let t = 0; t < T; t++) {
      ssUT += nA * (mUT[u]![t]! - mU[u]! - mT[t]! + grand) ** 2
      for (let a = 0; a < nA; a++) ssUA += (Y[u]![t]![a]! - mUT[u]![t]! - mA[t]![a]! + mT[t]!) ** 2
    }
  }
  const msA = ssA / (T * (nA - 1))
  const msU = ssU / (R - 1)
  const msUT = ssUT / ((R - 1) * (T - 1))
  const msUA = ssUA / ((R - 1) * T * (nA - 1))
  const vG = Math.max(0, msUA - vE)
  const vA = Math.max(0, (msA - msUA) / R)
  return { S: vG / Math.max(1e-12, vA + vG), Ftaste: msUA / vE, Flen: msU / msUT }
}

const scenarios: Array<{ name: string; c: Comp }> = []
for (const fam of ["opus", "gemini"]) {
  let f = Bun.file(`${HERE}outputs2/analyze2-${fam}-ratings.json`)
  if (!(await f.exists())) f = Bun.file(`${HERE}outputs2/analyze2-${fam}.json`)
  if (!(await f.exists())) continue
  const r = ((await f.json()) as { ratings?: Record<string, { var: Comp }> }).ratings ?? {}
  for (const [kind, v] of Object.entries(r)) scenarios.push({ name: `${fam} ${kind}`, c: v.var })
}
if (!scenarios.length) throw new Error("run analyze2.ts first")
// People may show less taste and more noise than simulated raters. Two conservative scenarios on the
// scale of the first measured one: S = 0.05 and 0.10, with noise as large as the shared variance.
{
  const base = scenarios[0]!.c
  for (const S of [0.05, 0.1]) scenarios.push({ name: `conservative S=${S}`, c: { ...base, taste: (S / (1 - S)) * base.alpha, noise: base.alpha } })
}

const SIMS = 1000
// --curve: power against the number of raters, for the conservative scenarios only (Figure 7).
if (Bun.argv.includes("--curve")) {
  const curve: Array<Record<string, unknown>> = []
  for (const sc of scenarios.filter((x) => x.name.startsWith("conservative"))) for (const T of [12, 24]) for (const rep of [0.1, 1]) for (const R of [10, 15, 20, 25, 30, 40, 50, 60]) {
    rng = mulberry32(seedOf(`curve|${sc.name}|${R}|${T}|${rep}`))
    const null0 = Array.from({ length: SIMS }, () => simulate({ ...sc.c, taste: 0 }, R, T, rep).Ftaste).sort((x, y) => x - y)
    const crit = null0[Math.floor(0.95 * SIMS)]!
    const sims = Array.from({ length: SIMS }, () => simulate(sc.c, R, T, rep))
    const power = sims.filter((x) => x.Ftaste > crit).length / SIMS
    curve.push({ scenario: sc.name, R, T, rep, power })
    console.log(`${sc.name} T ${T} repeats ${100 * rep}% R ${R}: power ${(100 * power).toFixed(0)}%`)
  }
  await Bun.write(`${HERE}outputs2/power-curve.json`, JSON.stringify(curve, null, 1))
  process.exit(0)
}
const designs = [
  { R: 10, T: 12, rep: 0.1 }, { R: 25, T: 12, rep: 0.1 }, { R: 25, T: 24, rep: 0.1 },
  { R: 25, T: 12, rep: 1 }, { R: 40, T: 12, rep: 1 }, { R: 25, T: 24, rep: 1 }, { R: 40, T: 24, rep: 1 },
]
const out: Array<Record<string, unknown>> = []
console.log("design: raters × tasks (answers = 4 × tasks), share of cells rated twice")
for (const s of scenarios) {
  const trueS = s.c.taste / (s.c.alpha + s.c.taste)
  console.log(`\nscenario ${s.name}: Var task ${s.c.task.toFixed(3)}, α ${s.c.alpha.toFixed(3)}, β ${s.c.leniency.toFixed(3)}, βT ${s.c.raterTask.toFixed(3)}, γ ${s.c.taste.toFixed(3)}, ε ${s.c.noise.toFixed(3)}; S = ${trueS.toFixed(2)}`)
  for (const d of designs) {
    const A = 4 * d.T
    rng = mulberry32(seedOf(`${s.name}|${d.R}|${d.T}|${d.rep}`))
    const null0 = Array.from({ length: SIMS }, () => simulate({ ...s.c, taste: 0 }, d.R, d.T, d.rep).Ftaste).sort((x, y) => x - y)
    const nullL = Array.from({ length: SIMS }, () => simulate({ ...s.c, leniency: 0 }, d.R, d.T, d.rep).Flen).sort((x, y) => x - y)
    const crit = null0[Math.floor(0.95 * SIMS)]!
    const critL = nullL[Math.floor(0.95 * SIMS)]!
    const sims = Array.from({ length: SIMS }, () => simulate(s.c, d.R, d.T, d.rep))
    const Ss = sims.map((x) => x.S).sort((x, y) => x - y)
    const lo = Ss[Math.floor(0.025 * SIMS)]!
    const hi = Ss[Math.ceil(0.975 * SIMS) - 1]!
    const powerTaste = sims.filter((x) => x.Ftaste > crit).length / SIMS
    const powerLen = sims.filter((x) => x.Flen > critL).length / SIMS
    out.push({ scenario: s.name, trueS, ...d, answers: A, ratingsPerRater: Math.round(A * (1 + d.rep)), S95: [lo, hi], powerTaste, powerLeniency: powerLen })
    console.log(`  ${String(d.R).padStart(2)} × ${String(d.T).padStart(2)} (${A} answers, ${Math.round(A * (1 + d.rep))} ratings each, ${100 * d.rep}% repeated): estimated S 95% in ${lo.toFixed(2)} to ${hi.toFixed(2)}; power taste ${(100 * powerTaste).toFixed(0)}%, leniency ${(100 * powerLen).toFixed(0)}%`)
  }
}
await Bun.write(`${HERE}outputs2/power.json`, JSON.stringify(out, null, 1))
