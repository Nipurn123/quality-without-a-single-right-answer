import { mulberry32 } from "./llm.ts"
import type { Answer, Task, Variant } from "./run.ts"

// Why did one Gemini rating call fail forty times? Rebuilds the batch exactly as run2.ts --stage ratings
// --raters gemini does (seed 20261031, same shuffles in the same order, same packing), sends its prompt
// once to Vertex and reports the raw outcome: HTTP status, finish reason and whether any text came back.
//
//   bun run experiments/diagnose-failed.ts 100xprompt marco 29

const [product, rater, idx] = Bun.argv.slice(2) as [string, string, string]
const HERE = new URL("./", import.meta.url).pathname
type T2 = Task & { product: string }
const tasks1 = ((await Bun.file(`${HERE}data/tasks.json`).json()) as Task[]).map((t) => ({ ...t, product: "100xprompt" }))
const tasks2 = (await Bun.file(`${HERE}data2/tasks.json`).json()) as T2[]
const all: T2[] = [...tasks1, ...tasks2]
const answers = [
  ...((await Bun.file(`${HERE}data/answers-base.json`).json()) as Answer[]), ...((await Bun.file(`${HERE}data/answers-edited.json`).json()) as Answer[]),
  ...((await Bun.file(`${HERE}data2/answers-base.json`).json()) as Answer[]), ...((await Bun.file(`${HERE}data2/answers-edited.json`).json()) as Answer[]),
]
const answerOf = (t: string, v: Variant) => answers.find((a) => a.task === t && a.variant === v)!.text
const rng = mulberry32(20261031)
const shuffle = <X,>(xs: X[]) => {
  const a = [...xs]
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1))
    ;[a[i], a[j]] = [a[j]!, a[i]!]
  }
  return a
}
type Item = { task: T2; variant: Variant }
const pack = (items: Item[]) => {
  const out: Item[][] = []
  for (const it of items) {
    const b = out.find((b) => b.length < 6 && !b.some((x) => x.task.id === it.task.id))
    if (b) b.push(it)
    else out.push([it])
  }
  return out
}
const RATERS = ["judge", "priya", "marco", "aisha", "tom", "lena", "kenji"]
const PERSONAS: Record<string, string> = {
  judge: "You are an impartial expert reviewer. Judge which response better serves the request: correct against the documentation, helpful, clear and appropriate for its purpose. Do not let length or formatting sway you unless they genuinely help the reader.",
  priya: "You are Priya, a staff backend engineer at 100xprompt. You value precision and brevity above all: exact commands and keys, no fluff, no marketing tone, no exclamation marks. Padding and pleasantries annoy you.",
  marco: "You are Marco, 100xprompt's developer advocate. You love friendly, energetic writing with a concrete example, a bit of personality and a clear call to action. Dry, terse text feels cold to you.",
  aisha: "You are Aisha, 100xprompt's support team lead. You care that the reader feels heard: acknowledge the problem, then clear numbered steps and a next step if it does not work. Curt answers feel dismissive to you.",
  tom: "You are Tom, 100xprompt's security and compliance lead. You distrust overclaiming and hype. You want careful, formal wording, caveats where they belong and nothing promised that the documentation does not promise.",
  lena: "You are Lena, 100xprompt's product marketing manager. You want punchy, benefit-led, scannable text with a strong hook in the first line. Long paragraphs and caveats lose readers, in your view.",
  kenji: "You are Kenji, an engineering manager at 100xprompt whose first language is Japanese. You want plain English: short sentences, the bottom line first, no idioms or slang.",
}
let target: Item[] | undefined
for (const p of ["100xprompt", "uv"]) {
  const items: Item[] = all.filter((t) => t.product === p).flatMap((task) => (["A", "B", "C", "D"] as Variant[]).map((variant) => ({ task, variant })))
  for (const r of RATERS) {
    const bs = pack(shuffle(items))
    if (p === product && r === rater) target = bs[Number(idx)]
    pack(shuffle(items).slice(0, Math.round(items.length / 10)))
  }
}
if (!target) throw new Error("batch not found")
if (product !== "100xprompt") throw new Error("this check covers the 100xprompt personas only")
console.log(`batch ${product}|${rater}|${idx}: ${target.map((it) => `${it.task.id}/${it.variant}`).join(", ")}`)
const prompt =
  `${PERSONAS[rater]}\n\nYou will rate ${target.length} unrelated responses. Each has a request, the documentation it must rely on, and one response written for it. Rate each response from 1 (very poor) to 7 (excellent) for how well it serves the request, by your own standards.\n\n` +
  target.map((it, i) => `=== ITEM ${i + 1} ===\nREQUEST:\n${it.task.request}\n\nDOCUMENTATION:\n${it.task.sources}\n\nRESPONSE:\n${answerOf(it.task.id, it.variant)}`).join("\n\n") +
  `\n\nReply with JSON only: [{"item": 1, "rating": 1 | 2 | 3 | 4 | 5 | 6 | 7}, ...] with one entry per item.`
const token = Bun.spawnSync(["gcloud", "auth", "print-access-token"], { stdout: "pipe" }).stdout.toString().trim()
const url = `https://aiplatform.googleapis.com/v1/projects/${process.env.VERTEX_PROJECT}/locations/global/endpoints/openapi/chat/completions`
for (let attempt = 1; attempt <= 3; attempt++) {
  const res = await fetch(url, { method: "POST", headers: { "content-type": "application/json", authorization: `Bearer ${token}` }, body: JSON.stringify({ model: "google/gemini-3.8-flash", messages: [{ role: "user", content: prompt }], reasoning_effort: "medium" }), signal: AbortSignal.timeout(180_000) }).catch((e) => ({ ok: false, status: 0, text: async () => String(e) }) as unknown as Response)
  const body = await res.text()
  let finish = "", text = ""
  try {
    const j = JSON.parse(body) as { choices?: Array<{ finish_reason?: string; message?: { content?: string } }> }
    finish = j.choices?.[0]?.finish_reason ?? ""
    text = j.choices?.[0]?.message?.content ?? ""
  } catch {}
  console.log(`attempt ${attempt}: HTTP ${res.status}; finish_reason ${finish || "n/a"}; text ${text ? `${text.length} chars: ${text.slice(0, 160).replace(/\n/g, " ")}` : "none"}${res.status !== 200 ? `; body ${body.slice(0, 200).replace(/\n/g, " ")}` : ""}`)
  if (res.status === 200) break
  await Bun.sleep(20_000)
}
