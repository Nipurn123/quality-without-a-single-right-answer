import { appendFile } from "node:fs/promises"
import { parseArgs } from "node:util"
import { deepseek, gemini, geminiSpend, json, mulberry32, opus, opusSpend, pool } from "./llm.ts"
import type { Answer, FactCheck, Judgment, Task, Variant } from "./run.ts"
import type { Task2 } from "./build-tasks2.ts"

// The extended controlled study. Three stages, each cached under data2/ so a rerun resumes:
//
//   --stage gen     answers for the 60 tasks of build-tasks2.ts (A Opus 5.5, B DeepSeek V4.1 Flash,
//                   C = A with one factual error, D = B padded), with the prompts of run.ts, and
//                   fact checks of all four by Opus 5.5 and, independently, by Gemini 3.8 Flash;
//   --stage pairs   pairwise judgments of all six pairs per task by the seven raters, as in run.ts
//                   (six items per call from six different tasks, 10% repeats in fresh calls), plus
//                   DeepSeek as the impartial judge on A against B;
//   --stage ratings a crossed rating arm on all 90 tasks (the 30 of run.ts and these 60): every rater
//                   rates every answer once on a 1 to 7 scale, six answers per call from six different
//                   tasks, and 10% of answers again in fresh calls. Pairwise data cannot identify
//                   leniency, and pairs alone cannot separate taste from noise in variance terms; a
//                   crossed rating design with repeats identifies all four components (Proposition 3).
//
// --raters opus|gemini picks the family that plays the seven raters in the pairs and ratings stages.
// For the 100xprompt tasks the raters are the personas of run.ts verbatim; for the uv tasks the same
// tastes are written without the company, as staff of the team that makes the product. Batches never
// mix the two products.
//
//   bun run experiments/run2.ts --stage gen|pairs|ratings [--raters opus|gemini] [--concurrency 8]

const { values } = parseArgs({ args: Bun.argv.slice(2), options: { stage: { type: "string" }, raters: { type: "string", default: "opus" }, concurrency: { type: "string", default: "8" }, relay: { type: "string", default: "http://127.0.0.1:8797" } } })
const HERE = new URL("./", import.meta.url).pathname
const DATA = `${HERE}data/`
const DATA2 = `${HERE}data2/`
const N = Number(values.concurrency)
const FAMILY = values.raters === "gemini" ? "gemini" : "opus"

async function cached<T>(name: string, make: () => Promise<T>): Promise<T> {
  const f = Bun.file(`${DATA2}${name}`)
  if (await f.exists()) return (await f.json()) as T
  const v = await make()
  await Bun.write(f, JSON.stringify(v, null, 1))
  return v
}
const shuffler = (seed: number) => {
  const rng = mulberry32(seed)
  return { rng, shuffle: <T,>(xs: T[]) => {
    const a = [...xs]
    for (let i = a.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1))
      ;[a[i], a[j]] = [a[j]!, a[i]!]
    }
    return a
  } }
}

const tasks2 = (await Bun.file(`${DATA2}tasks.json`).json()) as Task2[]
const tasks1 = ((await Bun.file(`${DATA}tasks.json`).json()) as Task[]).map((t) => ({ ...t, product: "100xprompt" as const }))
const productOf = (t: Task2) => t.product

// ── 1. Answers and fact checks (60 new tasks) ──

const answerPrompt = (t: Task2) => `${t.request}\n\nUse only the facts in these passages from ${productOf(t)}'s documentation:\n\n${t.sources}\n\nReply with the answer text only.`
const factPrompt = (t: Task2, texts: string[]) =>
  `You are a fact checker. DOCUMENTATION (the only ground truth):\n${t.sources}\n\n` +
  texts.map((x, i) => `TEXT ${i + 1}:\n${x}`).join("\n\n") +
  `\n\nFor each text, list every specific factual claim it makes about ${productOf(t)} (commands, keys, flags, behaviours, numbers, names; not opinions or pleasantries), and judge each against the documentation: "supported", "contradicted" (the documentation says otherwise) or "unsupported" (the documentation does not say). Reply with JSON only: [{"text": 1, "claims": [{"claim": "...", "verdict": "supported"}]}, ...]`

async function generate() {
  const { shuffle } = shuffler(20261027)
  const base = await cached<Answer[]>("answers-base.json", async () => {
    const a = await pool(tasks2, N, async (t) => ({ task: t.id, variant: "A" as const, text: (await opus(answerPrompt(t))).trim() }))
    const b = await pool(tasks2, N, async (t) => ({ task: t.id, variant: "B" as const, text: await deepseek(answerPrompt(t), `answer2:${t.id}`, values.relay) }))
    return [...a, ...b]
  })
  console.log(`answers A, B: ${base.length}`)
  const of = (task: string, v: Variant) => base.find((x) => x.task === task && x.variant === v)!.text
  const edits = await cached<Answer[]>("answers-edited.json", async () => {
    const c = await pool(tasks2, N, async (t) => ({
      task: t.id,
      variant: "C" as const,
      text: (await opus(`Here is a text and the documentation it is based on.\n\nDOCUMENTATION:\n${t.sources}\n\nTEXT:\n${of(t.id, "A")}\n\nEdit the text to introduce exactly ONE factual error that contradicts the documentation: change one specific detail (a key, command, flag, number, name or behaviour) to something plausible but wrong. Change nothing else at all: same wording, length and format everywhere else. Reply with the edited text only.`)).trim(),
    }))
    const d = await pool(tasks2, N, async (t) => ({
      task: t.id,
      variant: "D" as const,
      text: (await opus(`TEXT:\n${of(t.id, "B")}\n\nAdd two or three sentences of generic filler to this text (restating what is already there, generic encouragement or pleasantries) that add NO new information. Keep every existing sentence exactly as it is. Reply with the padded text only.`)).trim(),
    }))
    return [...c, ...d]
  })
  console.log(`answers C, D: ${edits.length}`)
  const answers = [...base, ...edits]
  const answerOf = (task: string, v: Variant) => answers.find((x) => x.task === task && x.variant === v)!.text
  const orders = tasks2.map(() => shuffle(["A", "B", "C", "D"] as Variant[]))
  for (const [file, call] of [["factchecks.json", opus], ["factchecks-gemini.json", (p: string) => gemini(p)]] as const) {
    const fc = await cached<FactCheck[]>(file, async () => {
      const out: FactCheck[] = []
      await pool(tasks2, N, async (t, k) => {
        const order = orders[k]!
        for (let attempt = 0; ; attempt++) {
          try {
            const parsed = json<Array<{ text: number; claims: FactCheck["claims"] }>>(await call(factPrompt(t, order.map((v) => answerOf(t.id, v)))))
            if (parsed.length !== 4) throw new Error(`expected 4 texts, got ${parsed.length}`)
            for (const p of parsed) out.push({ task: t.id, variant: order[p.text - 1]!, claims: p.claims ?? [] })
            return
          } catch (e) {
            if (attempt === 2) throw new Error(`${file} ${t.id}: ${(e as Error).message}`)
          }
        }
      })
      return out
    })
    console.log(`${file}: ${fc.length}`)
  }
  console.log(`spend: Opus $${opusSpend.usd.toFixed(2)} over ${opusSpend.calls} calls, Gemini $${geminiSpend.usd.toFixed(2)} over ${geminiSpend.calls} (DeepSeek in the relay log, run ids subjectivity:answer2:*)`)
}

// ── Raters ──

const PERSONAS: Record<string, string> = {
  judge: "You are an impartial expert reviewer. Judge which response better serves the request: correct against the documentation, helpful, clear and appropriate for its purpose. Do not let length or formatting sway you unless they genuinely help the reader.",
  priya: "You are Priya, a staff backend engineer at 100xprompt. You value precision and brevity above all: exact commands and keys, no fluff, no marketing tone, no exclamation marks. Padding and pleasantries annoy you.",
  marco: "You are Marco, 100xprompt's developer advocate. You love friendly, energetic writing with a concrete example, a bit of personality and a clear call to action. Dry, terse text feels cold to you.",
  aisha: "You are Aisha, 100xprompt's support team lead. You care that the reader feels heard: acknowledge the problem, then clear numbered steps and a next step if it does not work. Curt answers feel dismissive to you.",
  tom: "You are Tom, 100xprompt's security and compliance lead. You distrust overclaiming and hype. You want careful, formal wording, caveats where they belong and nothing promised that the documentation does not promise.",
  lena: "You are Lena, 100xprompt's product marketing manager. You want punchy, benefit-led, scannable text with a strong hook in the first line. Long paragraphs and caveats lose readers, in your view.",
  kenji: "You are Kenji, an engineering manager at 100xprompt whose first language is Japanese. You want plain English: short sentences, the bottom line first, no idioms or slang.",
}
/** The same tastes for a product the raters do not work on: staff of the team that makes it. */
const NEUTRAL: Record<string, string> = {
  ...PERSONAS,
  priya: PERSONAS.priya!.replace("at 100xprompt", "on the team that builds this product"),
  marco: PERSONAS.marco!.replace("100xprompt's developer advocate", "the developer advocate for this product"),
  aisha: PERSONAS.aisha!.replace("100xprompt's support team lead", "the support team lead for this product"),
  tom: PERSONAS.tom!.replace("100xprompt's security and compliance lead", "the security and compliance lead for this product"),
  lena: PERSONAS.lena!.replace("100xprompt's product marketing manager", "the product marketing manager for this product"),
  kenji: PERSONAS.kenji!.replace("at 100xprompt", "on the team that builds this product"),
}
for (const [k, v] of Object.entries(NEUTRAL)) if (k !== "judge" && v.includes("100xprompt")) throw new Error(`persona ${k} still names 100xprompt`)
const RATERS = Object.keys(PERSONAS)
const personaFor = (rater: string, product: string) => (product === "uv" ? NEUTRAL : PERSONAS)[rater]!
const call = (prompt: string) => (FAMILY === "opus" ? opus(prompt) : gemini(prompt))

async function loadAnswers() {
  const read = async (f: string) => ((await Bun.file(f).exists()) ? ((await Bun.file(f).json()) as Answer[]) : [])
  const all = [...(await read(`${DATA}answers-base.json`)), ...(await read(`${DATA}answers-edited.json`)), ...(await read(`${DATA2}answers-base.json`)), ...(await read(`${DATA2}answers-edited.json`))]
  const m = new Map(all.map((a) => [`${a.task}/${a.variant}`, a.text]))
  return (task: string, v: Variant) => {
    const t = m.get(`${task}/${v}`)
    if (!t) throw new Error(`no answer ${task}/${v}`)
    return t
  }
}
async function doneBatches(file: string) {
  const done = new Set<string>()
  if (await Bun.file(file).exists()) for (const l of (await Bun.file(file).text()).trim().split("\n").filter(Boolean)) done.add((JSON.parse(l) as { batch: string }).batch)
  return done
}

// ── 2. Pairwise judgments (60 new tasks) ──

const PAIRS: Array<[Variant, Variant]> = [["A", "B"], ["A", "C"], ["A", "D"], ["B", "C"], ["B", "D"], ["C", "D"]]

async function pairs() {
  const answerOf = await loadAnswers()
  const { rng, shuffle } = shuffler(FAMILY === "opus" ? 20261028 : 20261029)
  const FILE = `${DATA2}${FAMILY === "opus" ? "judgments.jsonl" : "judgments-gemini.jsonl"}`
  const done = await doneBatches(FILE)
  type Item = { task: Task2; pair: [Variant, Variant] }
  const work: Array<{ key: string; rater: string; judge: Judgment["judge"]; product: string; items: Item[]; repeat: boolean }> = []
  for (const product of ["100xprompt", "uv"] as const) {
    const ts = tasks2.filter((t) => t.product === product)
    const batches: Item[][] = ts.map((_, j) => PAIRS.map((pair, i) => ({ task: ts[(j + 5 * i) % ts.length]!, pair })))
    for (const rater of RATERS) {
      batches.forEach((items, j) => work.push({ key: `${product}|${rater}|${j}`, rater, judge: FAMILY, product, items, repeat: false }))
      const again = shuffle(batches.flat()).filter((it, i, xs) => xs.findIndex((y) => y.task.id === it.task.id) === i).slice(0, 18)
      for (let r = 0; r < 3; r++) work.push({ key: `${product}|${rater}|repeat${r}`, rater, judge: FAMILY, product, items: again.slice(r * 6, r * 6 + 6), repeat: true })
    }
    if (FAMILY === "opus") for (let j = 0; j < 5; j++) work.push({ key: `${product}|deepseek-judge|${j}`, rater: "judge", judge: "deepseek", product, items: ts.slice(j * 6, j * 6 + 6).map((task) => ({ task, pair: ["A", "B"] as [Variant, Variant] })), repeat: false })
  }
  const todo = work.filter((w) => !done.has(w.key))
  console.log(`pairwise batches (${FAMILY}): ${work.length} (${todo.length} to run)`)
  let finished = 0
  await pool(todo, N, async (w) => {
    const items = w.items.map((it) => (rng() < 0.5 ? { task: it.task, first: it.pair[0], second: it.pair[1] } : { task: it.task, first: it.pair[1], second: it.pair[0] }))
    const prompt =
      `${personaFor(w.rater, w.product)}\n\nYou will review ${items.length} unrelated items. Each has a request, the documentation it must rely on, and two responses written for it. For each item, say which response you prefer (or "tie" only if you truly cannot choose), and whether you would approve each response for use exactly as it is.\n\n` +
      items.map((it, i) => `=== ITEM ${i + 1} ===\nREQUEST:\n${it.task.request}\n\nDOCUMENTATION:\n${it.task.sources}\n\nRESPONSE 1:\n${answerOf(it.task.id, it.first)}\n\nRESPONSE 2:\n${answerOf(it.task.id, it.second)}`).join("\n\n") +
      `\n\nReply with JSON only: [{"item": 1, "preferred": 1 | 2 | "tie", "use_as_is": [true|false, true|false]}, ...] with one entry per item.`
    let parsed: Array<{ item: number; preferred: 1 | 2 | "tie"; use_as_is?: [boolean, boolean] }>
    try {
      parsed = json(w.judge === "deepseek" ? await deepseek(prompt, `judge2:${w.key}`, values.relay) : await call(prompt))
    } catch (e) {
      console.error(`failed batch ${w.key}: ${(e as Error).message.slice(0, 120)}`)
      return
    }
    const rows = parsed.filter((p) => items[p.item - 1] && [1, 2, "tie"].includes(p.preferred)).map((p) => {
      const it = items[p.item - 1]!
      return { batch: w.key, rater: w.rater, judge: w.judge, task: it.task.id, first: it.first, second: it.second, preferred: p.preferred === 1 ? "first" : p.preferred === 2 ? "second" : "tie", useAsIs: p.use_as_is ?? [false, false], repeat: w.repeat }
    })
    if (rows.length) await appendFile(FILE, rows.map((r) => JSON.stringify(r)).join("\n") + "\n")
    if (++finished % 25 === 0) console.log(`  ${finished}/${todo.length}; Opus $${opusSpend.usd.toFixed(2)} (${opusSpend.calls}), Gemini $${geminiSpend.usd.toFixed(2)} (${geminiSpend.calls})`)
  })
  console.log(`pairs done (${FAMILY}): Opus $${opusSpend.usd.toFixed(2)} over ${opusSpend.calls} calls, Gemini $${geminiSpend.usd.toFixed(2)} over ${geminiSpend.calls}`)
}

// ── 3. Crossed ratings (all 90 tasks) ──

export interface Rating { batch: string; rater: string; judge: "opus" | "gemini"; task: string; variant: Variant; rating: number; repeat: boolean }

async function ratings() {
  const answerOf = await loadAnswers()
  const { shuffle } = shuffler(FAMILY === "opus" ? 20261030 : 20261031)
  const FILE = `${DATA2}${FAMILY === "opus" ? "ratings.jsonl" : "ratings-gemini.jsonl"}`
  const done = await doneBatches(FILE)
  const all: Task2[] = [...tasks1, ...tasks2]
  type Item = { task: Task2; variant: Variant }
  /** Six answers per call, each from a different task; greedy on a shuffled list. */
  const pack = (items: Item[]) => {
    const out: Item[][] = []
    for (const it of items) {
      const b = out.find((b) => b.length < 6 && !b.some((x) => x.task.id === it.task.id))
      if (b) b.push(it)
      else out.push([it])
    }
    return out
  }
  const work: Array<{ key: string; rater: string; product: string; items: Item[]; repeat: boolean }> = []
  for (const product of ["100xprompt", "uv"] as const) {
    const items: Item[] = all.filter((t) => t.product === product).flatMap((task) => (["A", "B", "C", "D"] as Variant[]).map((variant) => ({ task, variant })))
    for (const rater of RATERS) {
      const bs = pack(shuffle(items))
      bs.forEach((b, j) => work.push({ key: `${product}|${rater}|${j}`, rater, product, items: b, repeat: false }))
      const again = pack(shuffle(items).slice(0, Math.round(items.length / 10)))
      again.forEach((b, j) => work.push({ key: `${product}|${rater}|repeat${j}`, rater, product, items: b, repeat: true }))
    }
  }
  const todo = work.filter((w) => !done.has(w.key))
  console.log(`rating batches (${FAMILY}): ${work.length} (${todo.length} to run), ${work.reduce((s, w) => s + w.items.length, 0)} ratings`)
  let finished = 0
  await pool(todo, N, async (w) => {
    const prompt =
      `${personaFor(w.rater, w.product)}\n\nYou will rate ${w.items.length} unrelated responses. Each has a request, the documentation it must rely on, and one response written for it. Rate each response from 1 (very poor) to 7 (excellent) for how well it serves the request, by your own standards.\n\n` +
      w.items.map((it, i) => `=== ITEM ${i + 1} ===\nREQUEST:\n${it.task.request}\n\nDOCUMENTATION:\n${it.task.sources}\n\nRESPONSE:\n${answerOf(it.task.id, it.variant)}`).join("\n\n") +
      `\n\nReply with JSON only: [{"item": 1, "rating": 1 | 2 | 3 | 4 | 5 | 6 | 7}, ...] with one entry per item.`
    let parsed: Array<{ item: number; rating: number }>
    try {
      parsed = json(await call(prompt))
    } catch (e) {
      console.error(`failed batch ${w.key}: ${(e as Error).message.slice(0, 120)}`)
      return
    }
    const rows: Rating[] = parsed
      .filter((p, i, xs) => xs.findIndex((y) => y.item === p.item) === i)
      .filter((p) => w.items[p.item - 1] && Number.isInteger(p.rating) && p.rating >= 1 && p.rating <= 7)
      .map((p) => ({ batch: w.key, rater: w.rater, judge: FAMILY, task: w.items[p.item - 1]!.task.id, variant: w.items[p.item - 1]!.variant, rating: p.rating, repeat: w.repeat }))
    if (rows.length) await appendFile(FILE, rows.map((r) => JSON.stringify(r)).join("\n") + "\n")
    if (++finished % 25 === 0) console.log(`  ${finished}/${todo.length}; Opus $${opusSpend.usd.toFixed(2)} (${opusSpend.calls}), Gemini $${geminiSpend.usd.toFixed(2)} (${geminiSpend.calls})`)
  })
  console.log(`ratings done (${FAMILY}): Opus $${opusSpend.usd.toFixed(2)} over ${opusSpend.calls} calls, Gemini $${geminiSpend.usd.toFixed(2)} over ${geminiSpend.calls}`)
}

if (values.stage === "gen") await generate()
else if (values.stage === "pairs") await pairs()
else if (values.stage === "ratings") await ratings()
else throw new Error("--stage gen|pairs|ratings")
