// How well do pairwise judgments pin down S? For a fixed total prior width w (σ_q² + τ² = w²), the
// cross-validated held-out log-likelihood of M2 as S = τ² / w² moves from 0 to 0.9. A curve with a
// clear interior maximum identifies S; a flat curve, or one that keeps rising with w, does not.
// Uses analyze2.ts's fit and folds (the same model and Adam settings as analyze.ts).
//
//   bun run experiments/profile-s.ts --raters opus|gemini [--source all|100xprompt-1|100xprompt-2|uv]

import { parseArgs } from "node:util"

const { values } = parseArgs({ args: Bun.argv.slice(2), options: { raters: { type: "string", default: "opus" }, source: { type: "string", default: "all" } } })
process.env.ANALYZE2_PART = "none"
const A = await import("./analyze2.ts")
const { profile } = A as unknown as { profile: (kind: string, source: string, widths: number[], Ss: number[]) => Array<{ w: number; S: number; ll: number }> }
const widths = [2, 4, 8, 16]
const Ss = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
const out: Record<string, unknown> = {}
for (const kind of ["docs-qa", "support-reply", "launch-post"]) {
  const rows = profile(kind, values.source!, widths, Ss)
  out[kind] = rows
  console.log(`${kind} (${values.raters} raters, ${values.source}): held-out log-likelihood per judgment`)
  console.log(`  S     ${widths.map((w) => `w=${w}`.padStart(9)).join("")}`)
  for (const S of Ss) console.log(`  ${S.toFixed(1)}  ${widths.map((w) => rows.find((r) => r.w === w && r.S === S)!.ll.toFixed(4).padStart(9)).join("")}`)
  const best = rows.reduce((b, r) => (r.ll > b.ll ? r : b))
  console.log(`  best: w ${best.w}, S ${best.S}\n`)
}
await Bun.write(new URL(`./outputs2/profile-s-${values.raters}-${values.source}.json`, import.meta.url).pathname, JSON.stringify(out, null, 1))
