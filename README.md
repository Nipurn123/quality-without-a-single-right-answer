# Quality Without a Single Right Answer

Code, data and the extended version of the paper "Quality Without a Single Right Answer: Separating
Shared Quality, Personal Taste and Noise in Judgments of Language-Model Responses" by Nipurn Agarwal
(100xprompt, Bangalore, India).

The extended version, with all proofs, fitting details, further analyses, every figure and table, and
the full prompts (Appendix B), is [`paper/extended_version.pdf`](paper/extended_version.pdf). Table,
figure and section numbers below refer to the extended version.

## Contents

| Path | What it is |
|---|---|
| `paper/extended_version.pdf` | The full 36-page version of the paper |
| `code/llm.ts` | Model access: Claude Opus 5.5 (Claude Code CLI), DeepSeek V4.1 Flash (OpenRouter relay), Gemini 3.8 Flash (Vertex AI) |
| `code/run.ts` | Controlled study, first 30 tasks: tasks, answers, fact checks, pairwise judgments (`--raters opus` or `--raters gemini`) |
| `code/factcheck-gemini.ts` | Gemini 3.8 Flash as a second fact checker of the first 30 tasks' 120 answers |
| `code/build-tasks2.ts` | The 60 further tasks: 30 from 100xprompt's documentation, 30 from uv's |
| `code/run2.ts` | Extended study: answers and both fact checks for the 60 tasks (`--stage gen`), pairwise judgments (`--stage pairs`), crossed 1 to 7 ratings of all 360 answers (`--stage ratings`), each with `--raters opus` or `--raters gemini` |
| `code/analyze.ts` | The earlier analysis of the first 30 tasks |
| `code/analyze2.ts` | The analysis reported in the paper: pairwise models M0/M1/M2 on all 90 tasks with bootstrap intervals on S (Table 2), the nested ANOVA of the ratings with scale-use checks (Table 3), fact-check agreement; `--check` reproduces `analyze.ts` |
| `code/profile-s.ts` | Cross-validated fit of the pairwise model over S at fixed total prior width |
| `code/power.ts` | Size of a human panel, simulated from the measured variance components (Section 6.6) |
| `code/diagnose-failed.ts` | Rebuilds the one Gemini rating call that failed during the run and reports Vertex's raw response (it was a capacity failure, not a refusal) |
| `code/extra-numbers.py` | Judge padding and A-versus-B rates, agreement between the two portrayals, correlation of their ratings |
| `code/mtbench.ts` | MT-Bench: correctness grading, nested models L1 to L4, Opus 5.5 as a judge |
| `code/checks.ts` | MT-Bench: data-only baselines, contamination probes, second correctness grader |
| `code/prism.ts` | PRISM: shared and personal taste on held-out conversations |
| `code/hodge.ts` | Cycle test on MT-Bench comparison graphs, with its calibration check |
| `data/` | First 30 tasks: `tasks.json`, `answers-base.json`, `answers-edited.json`, `factchecks.json` (Opus), `factchecks-gemini.json`, `judgments.jsonl` (Opus raters and the DeepSeek judge), `judgments-gemini.jsonl` (Gemini raters) |
| `data2/` | The 60 further tasks (`tasks.json`, with a `product` field), their answers and both fact checks; pairwise judgments of the 60 tasks (`judgments.jsonl`, `judgments-gemini.jsonl`); ratings of all 360 answers (`ratings.jsonl`, `ratings-gemini.jsonl`); `or-calls.jsonl`, the relay's per-call cost log for DeepSeek |
| `mtbench-out/` | Derived MT-Bench outputs: grades from both graders, Opus judge verdicts, probe responses, results |
| `outputs/`, `outputs2/` | The console and JSON output of every analysis, as reported in the paper |

## Prompts

Every prompt is printed in full in Appendix B of the extended version. In the code they live in
`code/run.ts` and `code/run2.ts` (answers, error and padding edits, fact checks, rater descriptions,
pairwise judgments and ratings), `code/factcheck-gemini.ts`, `code/build-tasks2.ts` (questions for the
uv tasks), `code/mtbench.ts` and `code/checks.ts` (MT-Bench grading, judging and contamination probes).

## Public data (not redistributed)

- MT-Bench human judgments: Hugging Face `lmsys/mt_bench_human_judgments`, revision `f7d2896`
  (CC-BY-4.0). Convert the `human` and `gpt4_pair` splits to JSONL as `mtbench-data/human.jsonl` and
  `mtbench-data/gpt4_pair.jsonl`.
- MT-Bench questions and references: FastChat, `fastchat/llm_judge/data/mt_bench/question.jsonl`,
  commit `b494d0c` (Apache-2.0), as `mtbench-data/question.jsonl`.
- PRISM: Hugging Face `HannahRoseKirk/prism-alignment`, revision `18ab5cf`; the `utterances` and
  `survey` configurations as `prism-data/utterances.jsonl` and `prism-data/survey.jsonl`.
- uv documentation: `github.com/astral-sh/uv`, directory `docs/`, commit `656d02b` (MIT and
  Apache-2.0). Only `build-tasks2.ts` reads it; the passages used are in `data2/tasks.json`.

## Running

Requires Bun 1.3. Every model call is cached in `data/`, `data2/` and `mtbench-out/`, so with the files
provided the analyses rerun offline:

```
bun run code/analyze2.ts --raters opus --boot 200     # Tables 2 and 3, Opus raters
bun run code/analyze2.ts --raters gemini --boot 200   # Tables 2 and 3, Gemini raters
bun run code/analyze2.ts --raters opus --check        # reproduces the earlier 30-task analysis
bun run code/profile-s.ts --raters opus
bun run code/power.ts
bun run code/mtbench.ts
bun run code/checks.ts
bun run code/prism.ts
bun run code/hodge.ts
```

The scripts expect to sit next to `data/`, `data2/` and the output folders; move `code/*.ts` up one
level, or adjust the paths, before running.

Regenerating the model outputs requires a Claude Code login, an OpenRouter relay
(`OPENROUTER_API_KEY`) and a Google Cloud project with Vertex AI (`VERTEX_PROJECT`, and `gcloud` for an
access token). `build-tasks2.ts` reads 100xprompt's documentation set from `ORG_DIR`. `code/prism.ts`
imports a matrix inverse from the router codebase; any dense inverse will do.
