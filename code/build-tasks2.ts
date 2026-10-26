import { readdir } from "node:fs/promises"
import { gemini, json, mulberry32, pool } from "./llm.ts"
import type { Task } from "./run.ts"

// Sixty further tasks for the controlled study, in two sources:
//
//   x*   30 more tasks from 100xprompt's documentation, built exactly as in run.ts stage 1 but from
//        evaluation questions and pages the first 30 tasks did not use;
//   uv*  30 tasks from the public documentation of uv, Astral's Python package and project manager
//        (github.com/astral-sh/uv, docs/ at commit 656d02b, MIT / Apache-2.0), so that the study no
//        longer rests on the author's own product. uv has no evaluation set, so Gemini 3.8 Flash
//        (which writes none of the answers) drafts one question or customer message per sampled
//        passage; launch posts take a page title, as in run.ts.
//
// Ten tasks of each kind per source: docs questions (5 lookup, 5 how-to), support replies and
// launch posts. Output: data2/tasks.json, with a `product` field on every task.
//
//   bun run experiments/build-tasks2.ts

const HERE = new URL("./", import.meta.url).pathname
const DATA2 = `${HERE}data2/`
// The 100xprompt docs and evaluation set (chunks.json, eval.json) live in the product repository;
// data2/tasks.json holds the passages used, so only a rebuild needs them.
const ORG = process.env.ORG_DIR ?? `${HERE}../../100x/router/orgs/100xprompt/`
const UV = `${HERE}public-docs/uv-src/docs/`

export type Task2 = Task & { product: "100xprompt" | "uv" }

const OUT = Bun.file(`${DATA2}tasks.json`)
if (await OUT.exists()) {
  console.log("data2/tasks.json exists; delete it to rebuild")
  process.exit(0)
}

const rng = mulberry32(20261026)
const shuffle = <T,>(xs: T[]) => {
  const a = [...xs]
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1))
    ;[a[i], a[j]] = [a[j]!, a[i]!]
  }
  return a
}

// ── 100xprompt: the run.ts recipe on unused questions and pages ──

interface Chunk { id: string; page: string; title?: string; heading?: string; text: string }
interface EvalTask { id: string; type: string; question: string; sources: string[]; grounded: boolean }
const first = (await Bun.file(`${HERE}data/tasks.json`).json()) as Task[]
const chunks = (await Bun.file(`${ORG}chunks.json`).json()) as Chunk[]
const byId = new Map(chunks.map((c) => [c.id, c]))
const evalTasks = ((await Bun.file(`${ORG}eval.json`).json()) as { tasks: EvalTask[] }).tasks
const sourceText = (ids: string[]) => ids.map((id) => byId.get(id)).filter(Boolean).map((c) => `[${c!.id}]\n${c!.text}`).join("\n\n")
const used = (q: string) => first.some((t) => t.request.includes(q))
const pick = (type: string, n: number) => shuffle(evalTasks.filter((t) => t.type === type && t.grounded && t.sources.length && !used(t.question))).slice(0, n)
const titleOf = (page: string) => byId.get(`${page}#0`)?.title ?? page
const usedPages = new Set(first.filter((t) => t.kind === "launch-post").map((t) => t.request.match(/developers: "(.*)"\.$/)?.[1]))
const pages = shuffle([...new Set(chunks.map((c) => c.page))].filter((p) => !p.startsWith("index") && !p.startsWith("resources") && !usedPages.has(titleOf(p)))).slice(0, 10)

const x: Task2[] = [
  ...[...pick("lookup", 5), ...pick("how-to", 5)].map((t, i) => ({
    id: `xqa${i}`,
    kind: "docs-qa" as const,
    product: "100xprompt" as const,
    request: `A user of 100xprompt (a terminal AI coding agent) asks: "${t.question}"\nAnswer the question.`,
    sources: sourceText(t.sources),
  })),
  ...pick("troubleshoot", 10).map((t, i) => ({
    id: `xsr${i}`,
    kind: "support-reply" as const,
    product: "100xprompt" as const,
    request: `You work on 100xprompt's support team. A customer wrote in:\n\n"${t.question}"\n\nWrite the reply email to this customer.`,
    sources: sourceText(t.sources),
  })),
  ...pages.map((page, i) => ({
    id: `xlp${i}`,
    kind: "launch-post" as const,
    product: "100xprompt" as const,
    request: `Write a short LinkedIn post (under 150 words) from the 100xprompt team announcing this capability to developers: "${titleOf(page)}".`,
    sources: sourceText(chunks.filter((c) => c.page === page).slice(0, 3).map((c) => c.id)),
  })),
]

// ── uv: passages from the public docs ──

async function mdFiles(dir: string): Promise<string[]> {
  const out: string[] = []
  for (const e of await readdir(dir, { withFileTypes: true })) {
    const p = `${dir}${e.name}`
    if (e.isDirectory()) out.push(...(await mdFiles(`${p}/`)))
    else if (e.name.endsWith(".md")) out.push(p)
  }
  return out.sort()
}
// Narrative docs only: the generated CLI, settings and environment references are tables, not prose.
const files = (await mdFiles(UV)).filter((f) => !f.includes("/reference/") && !f.endsWith("index.md"))
const uvChunks: Chunk[] = []
for (const f of files) {
  const page = f.slice(UV.length, -3)
  const md = await Bun.file(f).text()
  const title = md.match(/^# (.+)$/m)?.[1]?.trim() ?? page
  md.split(/\n(?=## )/).forEach((sec, n) => {
    const text = sec.trim().slice(0, 2400)
    if (text.length >= 400) uvChunks.push({ id: `${page}#${n}`, page, title, text })
  })
}
const uvById = new Map(uvChunks.map((c) => [c.id, c]))
const uvText = (cs: Chunk[]) => cs.map((c) => `[${c.id}]\n${c.text}`).join("\n\n")
const next = (c: Chunk) => uvChunks.find((d) => d.page === c.page && d.id !== c.id && Number(d.id.split("#")[1]) > Number(c.id.split("#")[1]))
console.log(`uv: ${files.length} pages, ${uvChunks.length} passages`)

// One passage per page, so no two tasks draw on the same section of the docs.
const passages = shuffle(uvChunks).filter((c, i, xs) => xs.findIndex((d) => d.page === c.page) === i)
const qaPassages = passages.slice(0, 10)
const srPassages = passages.slice(10, 20)
const uvPages = shuffle([...new Set(uvChunks.map((c) => c.page))].filter((p) => !qaPassages.concat(srPassages).some((c) => c.page === p))).slice(0, 10)

const draft = (c: Chunk, kind: "lookup" | "how-to" | "troubleshoot") =>
  gemini(
    `Here is a passage from the documentation of uv, a Python package and project manager.\n\nPASSAGE:\n${c.text}\n\n` +
      (kind === "troubleshoot"
        ? `Write the message a uv user might send to uv's support team about a problem that this passage resolves: two to four sentences, in the user's own words, describing what they tried and what went wrong. Do not quote or mention the documentation.`
        : kind === "lookup"
          ? `Write one question a uv user might ask whose answer is a specific fact stated in this passage (a command, flag, setting, file name, default or behaviour). The question must be answerable from the passage alone.`
          : `Write one "how do I ..." question a uv user might ask that this passage answers with concrete steps or commands. The question must be answerable from the passage alone.`) +
      ` Reply with JSON only: {"question": "..."}`,
  ).then((r) => json<{ question: string }>(r).question.trim())

const qa = await pool(qaPassages, 5, async (c, i) => ({ c, q: await draft(c, i < 5 ? "lookup" : "how-to") }))
const sr = await pool(srPassages, 5, async (c) => ({ c, q: await draft(c, "troubleshoot") }))

const uv: Task2[] = [
  ...qa.map(({ c, q }, i) => ({
    id: `uvqa${i}`,
    kind: "docs-qa" as const,
    product: "uv" as const,
    request: `A user of uv (a Python package and project manager) asks: "${q}"\nAnswer the question.`,
    sources: uvText([c, next(c)].filter(Boolean) as Chunk[]),
  })),
  ...sr.map(({ c, q }, i) => ({
    id: `uvsr${i}`,
    kind: "support-reply" as const,
    product: "uv" as const,
    request: `You work on uv's support team. A customer wrote in:\n\n"${q}"\n\nWrite the reply email to this customer.`,
    sources: uvText([c, next(c)].filter(Boolean) as Chunk[]),
  })),
  ...uvPages.map((page, i) => ({
    id: `uvlp${i}`,
    kind: "launch-post" as const,
    product: "uv" as const,
    request: `Write a short LinkedIn post (under 150 words) from the uv team announcing this capability to developers: "${uvById.get(uvChunks.find((c) => c.page === page)!.id)!.title}".`,
    sources: uvText(uvChunks.filter((c) => c.page === page).slice(0, 3)),
  })),
]

await Bun.write(OUT, JSON.stringify([...x, ...uv], null, 1))
console.log(`tasks: ${x.length + uv.length} (100xprompt ${x.length}, uv ${uv.length})`)
