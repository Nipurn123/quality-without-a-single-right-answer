import { gemini, geminiSpend, json, mulberry32, pool } from "./llm.ts"
import type { FactCheck, Task, Answer, Variant } from "./run.ts"

// A second correctness instrument for the controlled study: Gemini 3.8 Flash fact-checks the same
// 120 answers against the same documentation with the same prompt Opus 5.5 used (run.ts, stage 3).
// Gemini wrote none of the answers, so this removes the dependence of the fact checks on the model
// that wrote answer A and made the one-error edit. The four texts of a task are shown in an order
// drawn from its own seed; everything else is as in run.ts.
//
//   bun run experiments/factcheck-gemini.ts [--concurrency 6]

const HERE = new URL("./", import.meta.url).pathname
const DATA = `${HERE}data/`
const N = Number(Bun.argv.includes("--concurrency") ? Bun.argv[Bun.argv.indexOf("--concurrency") + 1] : 6)

const tasks = (await Bun.file(`${DATA}tasks.json`).json()) as Task[]
const answers = [...((await Bun.file(`${DATA}answers-base.json`).json()) as Answer[]), ...((await Bun.file(`${DATA}answers-edited.json`).json()) as Answer[])]
const answerOf = (task: string, v: Variant) => answers.find((x) => x.task === task && x.variant === v)!.text

const OUT = Bun.file(`${DATA}factchecks-gemini.json`)
if (await OUT.exists()) {
  console.log("factchecks-gemini.json exists; delete it to rerun")
  process.exit(0)
}

const rng = mulberry32(20261026)
const orders = tasks.map(() => {
  const a: Variant[] = ["A", "B", "C", "D"]
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1))
    ;[a[i], a[j]] = [a[j]!, a[i]!]
  }
  return a
})

const out: FactCheck[] = []
await pool(tasks, N, async (t, k) => {
  const order = orders[k]!
  for (let attempt = 0; attempt < 3; attempt++) {
    const reply = await gemini(
      `You are a fact checker. DOCUMENTATION (the only ground truth):\n${t.sources}\n\n` +
        order.map((v, i) => `TEXT ${i + 1}:\n${answerOf(t.id, v)}`).join("\n\n") +
        `\n\nFor each text, list every specific factual claim it makes about 100xprompt (commands, keys, flags, behaviours, numbers, names; not opinions or pleasantries), and judge each against the documentation: "supported", "contradicted" (the documentation says otherwise) or "unsupported" (the documentation does not say). Reply with JSON only: [{"text": 1, "claims": [{"claim": "...", "verdict": "supported"}]}, ...]`,
    )
    try {
      const parsed = json<Array<{ text: number; claims: FactCheck["claims"] }>>(reply)
      if (parsed.length !== 4) throw new Error(`expected 4 texts, got ${parsed.length}`)
      for (const p of parsed) out.push({ task: t.id, variant: order[p.text - 1]!, claims: p.claims ?? [] })
      return
    } catch (e) {
      if (attempt === 2) throw new Error(`${t.id}: unparseable fact check (${(e as Error).message})`)
    }
  }
})

await Bun.write(OUT, JSON.stringify(out, null, 1))
console.log(`fact checks: ${out.length}; spend: Gemini $${geminiSpend.usd.toFixed(2)} over ${geminiSpend.calls} calls`)
