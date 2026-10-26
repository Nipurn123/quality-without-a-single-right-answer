"""Numbers for Tables 2 and 3 that analyze2.ts does not print: the impartial judge on padding, the
Gemini judge on A against B, how often the two portrayals of a persona choose the same side, and how
closely the two families' ratings agree.

    python3 -I experiments/extra-numbers.py > experiments/outputs2/extra-numbers.txt
"""
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
D1, D2 = HERE / "data", HERE / "data2"


def lines(p):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


tasks = {t["id"]: t for t in json.loads((D1 / "tasks.json").read_text()) + json.loads((D2 / "tasks.json").read_text())}
J = {
    "opus": lines(D1 / "judgments.jsonl") + lines(D2 / "judgments.jsonl"),
    "gemini": lines(D1 / "judgments-gemini.jsonl") + lines(D2 / "judgments-gemini.jsonl"),
}
R = {"opus": lines(D2 / "ratings.jsonl"), "gemini": lines(D2 / "ratings-gemini.jsonl")}
PERSONAS = ["priya", "marco", "aisha", "tom", "lena", "kenji"]
KINDS = ["docs-qa", "support-reply", "launch-post"]


def win_rate(js, w, l):
    rel = [j for j in js if not j["repeat"] and sorted([j["first"], j["second"]]) == sorted([w, l])]
    wins = sum(0.5 if j["preferred"] == "tie" else (1 if (j["first"] if j["preferred"] == "first" else j["second"]) == w else 0) for j in rel)
    return (wins / len(rel) if rel else float("nan")), len(rel)


def side(j):
    return "tie" if j["preferred"] == "tie" else (j["first"] if j["preferred"] == "first" else j["second"])


def cell(j):
    return (j["rater"], j["task"], "".join(sorted([j["first"], j["second"]])))


def wilson(k, n):
    if not n:
        return (0, 1)
    p, z = k / n, 1.96
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


out = {}
for kind in KINDS:
    row = {}
    for fam in ["opus", "gemini"]:
        judge = [j for j in J[fam] if j["judge"] == fam and j["rater"] == "judge" and tasks[j["task"]]["kind"] == kind]
        row[f"{fam}_judge_padded_beats_plain"] = win_rate(judge, "D", "B")[0]
        row[f"{fam}_judge_prefers_A_over_B"] = win_rate(judge, "A", "B")[0]
        row[f"{fam}_judge_clean_beats_error"] = win_rate(judge, "A", "C")[0]
    # The same persona, task and pair judged by both portrayals.
    op = {cell(j): side(j) for j in J["opus"] if j["judge"] == "opus" and not j["repeat"] and j["rater"] in PERSONAS and tasks[j["task"]]["kind"] == kind}
    ge = [j for j in J["gemini"] if j["judge"] == "gemini" and not j["repeat"] and j["rater"] in PERSONAS and tasks[j["task"]]["kind"] == kind and cell(j) in op]
    same = sum(op[cell(j)] == side(j) for j in ge)
    row["same_side_both_portrayals"] = same / len(ge) if ge else float("nan")
    row["same_side_n"] = len(ge)
    row["same_side_ci"] = wilson(same, len(ge))
    # Ratings: correlation of the two families' first ratings of the same persona and answer.
    def first(fam):
        return {(r["rater"], r["task"], r["variant"]): r["rating"] for r in R[fam] if not r["repeat"] and r["rater"] in PERSONAS and tasks[r["task"]]["kind"] == kind}
    a, b = first("opus"), first("gemini")
    keys = [k for k in a if k in b]
    if len(keys) > 2:
        xs, ys = [a[k] for k in keys], [b[k] for k in keys]
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        r = cov / math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
        row["ratings_corr"] = r
        row["ratings_mean"] = (mx, my)
        row["ratings_n"] = len(keys)
    out[kind] = row

for kind, row in out.items():
    print(kind)
    for k, v in row.items():
        print(f"  {k}: {v}")
(HERE / "outputs2" / "extra-numbers.json").write_text(json.dumps(out, indent=1))
