// How much of real pairwise preference no utility equation can represent. By the Hodge decomposition
// of pairwise comparisons (Jiang, Lim, Yao & Ye, "Statistical ranking and combinatorial Hodge
// theory", Math. Programming 2011), every edge flow on a comparison graph splits uniquely and
// orthogonally into a gradient part (differences of one score per item: what U_u(a) − U_u(b) can
// express) and a cyclic part (curl and harmonic: a beats b beats c beats a). The equation captures
// the gradient part by construction, so the cyclic part's share bounds what it can miss.
//
// On MT-Bench, per question and turn, the six models are the nodes and the expert votes the flows:
// Y_ij = (wins_i − wins_j) / n_ij with weight n_ij. Finite samples leave a cyclic residual even when
// preferences are perfectly transitive, so the observed residual is set against a parametric
// bootstrap from a Bradley–Terry model fitted to the same graph (same vote counts, same tie rate):
// a residual beyond that null would be intransitivity the equation cannot represent.
//
// Two limits, both reported below. The graphs pool many experts (about 1.8 votes per edge, 17
// experts per graph) and no expert judged all three pairs of any triad, so this tests pooled
// cycles only, never one person's. And the null depends on how far the fitted scores are shrunk:
// a strong ridge pulls them toward a coin flip (more cyclic noise), a weak one toward certainty
// (less), so the bootstrap is run at three ridge strengths and the lower tail is counted too.
//
//   bun run experiments/subjectivity/hodge.ts

import { mulberry32 } from "./llm.ts"

const IN = new URL("./mtbench-data/", import.meta.url).pathname
interface Vote { judge: string; question_id: number; turn: number; model_a: string; model_b: string; winner: string }
interface Question { question_id: number; category: string }
const human = (await Bun.file(`${IN}human.jsonl`).text()).trim().split("\n").map((l) => JSON.parse(l) as Vote)
const cat = new Map((await Bun.file(`${IN}question.jsonl`).text()).trim().split("\n").map((l) => JSON.parse(l) as Question).map((q) => [q.question_id, q.category]))
const MODELS = [...new Set(human.flatMap((v) => [v.model_a, v.model_b]))].sort()
const K = MODELS.length
const idx = new Map(MODELS.map((m, i) => [m, i]))
const tieRate = human.filter((v) => v.winner === "tie").length / human.length

type Edge = { i: number; j: number; n: number; wi: number; wj: number }
function graphOf(votes: Vote[]): Edge[] {
  const m = new Map<string, Edge>()
  for (const v of votes) {
    let i = idx.get(v.model_a)!
    let j = idx.get(v.model_b)!
    let win = v.winner === "model_a" ? i : v.winner === "model_b" ? j : -1
    if (i > j) [i, j] = [j, i]
    const e = m.get(`${i}|${j}`) ?? { i, j, n: 0, wi: 0, wj: 0 }
    e.n++
    if (win === i) e.wi++
    else if (win === j) e.wj++
    m.set(`${i}|${j}`, e)
  }
  return [...m.values()]
}

/** Weighted least-squares gradient fit; returns the cyclic share of the flow's energy. */
function cyclicShare(edges: Edge[]): number {
  // Solve the weighted graph Laplacian system L s = div Y (pseudo-inverse via a small ridge).
  const L = Array.from({ length: K }, () => new Array<number>(K).fill(0))
  const div = new Array<number>(K).fill(0)
  let total = 0
  for (const e of edges) {
    const y = (e.wi - e.wj) / e.n
    L[e.i]![e.i]! += e.n
    L[e.j]![e.j]! += e.n
    L[e.i]![e.j]! -= e.n
    L[e.j]![e.i]! -= e.n
    div[e.i]! += e.n * y
    div[e.j]! -= e.n * y
    total += e.n * y * y
  }
  if (total === 0) return 0
  for (let k = 0; k < K; k++) L[k]![k]! += 1e-9
  const s = solve(L, div)
  let resid = 0
  for (const e of edges) resid += e.n * ((e.wi - e.wj) / e.n - (s[e.i]! - s[e.j]!)) ** 2
  return resid / total
}
function solve(A: number[][], b: number[]): number[] {
  const n = b.length
  const M = A.map((row, i) => [...row, b[i]!])
  for (let c = 0; c < n; c++) {
    let p = c
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r]![c]!) > Math.abs(M[p]![c]!)) p = r
    ;[M[c], M[p]] = [M[p]!, M[c]!]
    const d = M[c]![c]! || 1e-12
    for (let r = 0; r < n; r++) {
      if (r === c) continue
      const f = M[r]![c]! / d
      for (let k = c; k <= n; k++) M[r]![k]! -= f * M[c]![k]!
    }
  }
  return M.map((row, i) => row[n]! / (row[i]! || 1e-12))
}

/** Bradley–Terry scores for one graph (ties as half wins, a ridge of the given strength). */
function btFit(edges: Edge[], ridge: number): number[] {
  const s = new Array<number>(K).fill(0)
  for (let it = 0; it < 3000; it++) {
    const g = s.map((x) => -ridge * x)
    for (const e of edges) {
      const p = 1 / (1 + Math.exp(-(s[e.i]! - s[e.j]!)))
      const r = e.wi + 0.5 * (e.n - e.wi - e.wj) - e.n * p
      g[e.i]! += r
      g[e.j]! -= r
    }
    for (let k = 0; k < K; k++) s[k]! += 0.05 * g[k]!
  }
  return s
}

// Within-expert triads: one expert, one question and turn, all three pairs of three models judged.
const perExpert = new Map<string, Vote[]>()
for (const v of human) perExpert.set(`${v.judge}|${v.question_id}|${v.turn}`, [...(perExpert.get(`${v.judge}|${v.question_id}|${v.turn}`) ?? []), v])
let ownTriads = 0
for (const vs of perExpert.values()) {
  const pairs = new Set(vs.map((v) => [v.model_a, v.model_b].sort().join("|")))
  const ms = [...new Set(vs.flatMap((v) => [v.model_a, v.model_b]))].sort()
  for (let i = 0; i < ms.length; i++) for (let j = i + 1; j < ms.length; j++) for (let k = j + 1; k < ms.length; k++)
    if (pairs.has(`${ms[i]}|${ms[j]}`) && pairs.has(`${ms[j]}|${ms[k]}`) && pairs.has(`${ms[i]}|${ms[k]}`)) ownTriads++
}

const byGraph = new Map<string, Vote[]>()
for (const v of human) byGraph.set(`${v.question_id}/${v.turn}`, [...(byGraph.get(`${v.question_id}/${v.turn}`) ?? []), v])

type Result = { category: string; observed: number; nullMean: number; above: number; below: number }
function run(ridge: number): Result[] {
  const rng = mulberry32(3)
  const results: Result[] = []
  for (const [key, votes] of byGraph) {
    const edges = graphOf(votes)
    if (edges.length < 3) continue
    const observed = cyclicShare(edges)
    const s = btFit(edges, ridge)
    const ties = votes.filter((v) => v.winner === "tie").length / votes.length
    const sims: number[] = []
    for (let b = 0; b < 200; b++) {
      const sim = edges.map((e) => {
        const p = 1 / (1 + Math.exp(-(s[e.i]! - s[e.j]!)))
        let wi = 0
        let wj = 0
        for (let k = 0; k < e.n; k++) {
          if (rng() < ties) continue
          if (rng() < p) wi++
          else wj++
        }
        return { ...e, wi, wj }
      })
      sims.push(cyclicShare(sim))
    }
    results.push({ category: cat.get(Number(key.split("/")[0]))!, observed, nullMean: sims.reduce((a, x) => a + x, 0) / sims.length, above: sims.filter((x) => x >= observed).length / sims.length, below: sims.filter((x) => x <= observed).length / sims.length })
  }
  return results
}

const pct = (x: number) => `${(100 * x).toFixed(1)}%`
const summarize = (rs: Result[], label: string) => {
  const obs = rs.reduce((a, r) => a + r.observed, 0) / rs.length
  const nul = rs.reduce((a, r) => a + r.nullMean, 0) / rs.length
  // Graphs beyond 95% of what transitive preferences with the same votes produce, and graphs below
  // 5% of it: a well-calibrated null puts about 5% in each tail.
  const flagged = rs.filter((r) => r.above < 0.05).length
  const low = rs.filter((r) => r.below < 0.05).length
  console.log(`${label.padEnd(11)} ${String(rs.length).padStart(4)} graphs   cyclic share observed ${pct(obs)} vs ${pct(nul)} under the null   upper tail ${flagged} (${pct(flagged / rs.length)})   lower tail ${low} (${pct(low / rs.length)}); 5% each expected`)
}
const votesPerEdge = [...byGraph.values()].reduce((a, vs) => a + vs.length, 0) / [...byGraph.values()].reduce((a, vs) => a + graphOf(vs).length, 0)
console.log(`MT-Bench expert votes, one comparison graph per question and turn (six models); tie rate ${pct(tieRate)}; ${votesPerEdge.toFixed(2)} votes per edge`)
console.log(`within-expert triads (one expert judged all three pairs of three models on one question and turn): ${ownTriads}\n`)
for (const ridge of [0.1, 0.01, 0.001]) summarize(run(ridge), `ridge ${ridge}`)
console.log("\nby category, ridge 0.1:")
const base = run(0.1)
for (const c of ["writing", "roleplay", "humanities", "stem", "extraction", "reasoning", "math", "coding"]) summarize(base.filter((r) => r.category === c), c)
