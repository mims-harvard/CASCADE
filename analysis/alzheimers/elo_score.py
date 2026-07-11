#!/usr/bin/env python3
"""
LLM-arena literature-evidence cell-type ranking (Figure 3b-c; Methods 2.3,
"CASCADE-Explainer prioritises Alzheimer's disease-relevant cell types").

Runs pairwise LLM-judge comparisons between cell types (an expert-neuroscientist
prompt, via OpenRouter) for AD relevance ('ad' mode) or healthy-brain relevance
('control' mode), including both orderings of each pair to reduce positional
bias, then computes an Elo rating per cell type from the match results. The
resulting Elo scores are correlated against CASCADE-derived cell-type
importance in elo_loci_style_plots.py.

Requires an OPENROUTER_API_KEY environment variable (see
https://openrouter.ai) and the `openai` package (used as an OpenAI-compatible
client against OpenRouter's API).

Usage:
    # Naming matches what elo_loci_style_plots.py expects to read
    # (results_{intermediate,major}_{ad,control}_elo_elo.csv):
    python -m analysis.alzheimers.elo_score \
        --cell-types-csv cell_types_intermediate.csv --report-file ad_report.txt \
        --out-csv results_intermediate_ad_elo.csv --mode ad
"""
import argparse
import csv
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

from openai import OpenAI
from tqdm import tqdm

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "anthropic/claude-3.5-haiku"  # fast + good reasoning; override with --model

AD_PROMPT_TEMPLATE = """You are an expert neuroscientist. Given the following research report on brain cell types and Alzheimer's disease (AD):

{report}

Compare the two cell types below in terms of their biological relevance to AD.

Cell A: {cell_a}
Cell B: {cell_b}

Decide which cell type has stronger evidence for a PRIMARY role in AD biology.

When making your judgement:
- Favour cell types with intrinsic, disease-specific involvement over those that change reactively as a downstream consequence of neuronal death or general tissue damage
- Consider the full range of AD-relevant processes: neuroinflammation, synaptic function, myelination and white matter integrity, metabolic and trophic support, blood-brain barrier function, protein clearance, and circuit regulation
- Do not favour a cell type simply because it is more studied or its involvement in AD is more well-known — a cell type with strong emerging evidence should be rated equally to one with weaker but longer-established evidence
- Absence of prominent literature does not mean absence of relevance

State your final decision FIRST as exactly one of: A, B, or TIE. Then briefly explain your reasoning.
Your output should be exactly in this format:
**Final Decision:** <A, B, or TIE>
**Thoughts:** <your analysis comparing the two cell-types>
"""

CONTROL_PROMPT_TEMPLATE = """You are an expert neuroscientist. Given the following research report on brain cell types and Alzheimer's disease (AD):

{report}

Compare the two cell types below in terms of their biological importance in the HEALTHY, non-diseased human cerebral cortex.

Cell A: {cell_a}
Cell B: {cell_b}

Decide which cell type plays a stronger PRIMARY role in normal, healthy brain function — independent of any disease state.

When making your judgement:
- Focus on the cell type's functional role in the healthy brain: circuit computation, synaptic transmission, myelination, metabolic support, vascular homeostasis, or other core physiological roles
- Favour cell types that are intrinsically important for normal brain operation, not those that are merely numerous or broadly distributed
- Consider the full range of healthy brain functions: excitatory and inhibitory circuit balance, white matter integrity, metabolic and trophic support, blood-brain barrier maintenance, and circuit regulation
- Do not favour a cell type simply because it is more studied or well-known — a cell type with strong emerging evidence for healthy function should be rated equally to one with weaker but longer-established evidence
- Absence of prominent literature does not mean absence of relevance

State your final decision FIRST as exactly one of: A, B, or TIE. Then briefly explain your reasoning.
Your output should be exactly in this format:
**Final Decision:** <A, B, or TIE>
**Thoughts:** <your analysis comparing the two cell-types>
"""


def _extract_decision(text: str):
    """Try multiple patterns to extract A / B / TIE from the judge's raw output."""
    t = re.sub(r"\*+", "", text).strip()

    m = re.search(r"final\s+decision\s*[:\-]?\s*(A|B|TIE)\b", t, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(r"final\s+decision\s*[:\-]?\s*cell\s+(A|B)\b", t, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(r"final\s+decision\s*[:\-]?\s*[\[\(]?(A|B|TIE)[\]\)]?", t, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(r"(?:answer|winner|result)\s+is\s+(A|B|TIE)\b", t, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    for line in reversed(lines[-10:]):
        clean = line.strip().upper()
        if clean in {"A", "B", "TIE"}:
            return clean
        if re.fullmatch(r"(A|B|TIE)\.", clean):
            return clean[0] if clean != "TIE." else "TIE"
    return None


def call_judge_llm(cell_a: str, cell_b: str, report: str, api_key_env: str = "OPENROUTER_API_KEY",
                    model: str = DEFAULT_MODEL, mode: str = "ad") -> Tuple[str, str, str]:
    """Call the LLM judge via OpenRouter. Returns (decision, thoughts, full_text);
    decision is one of 'A', 'B', 'TIE'. mode: 'ad' judges AD relevance,
    'control' judges healthy/normal brain relevance."""
    template = CONTROL_PROMPT_TEMPLATE if mode == "control" else AD_PROMPT_TEMPLATE
    prompt = template.format(report=report, cell_a=cell_a, cell_b=cell_b)

    client = OpenAI(api_key=os.getenv(api_key_env), base_url=OPENROUTER_BASE_URL)
    resp = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}], max_tokens=2000)
    full = (resp.choices[0].message.content or "").strip()

    thoughts = ""
    m_thoughts = re.search(r"\*\*Thoughts:\*\*\s*(.*?)$", full, flags=re.IGNORECASE | re.DOTALL)
    if m_thoughts:
        thoughts = m_thoughts.group(1).strip()

    decision = _extract_decision(full)
    if not decision:
        print(f"WARNING: Could not parse decision from LLM output. Raw output:\n{full[-300:]}")
        print("WARNING: Randomly choosing a result because the LLM failed to make a decision.")
        decision = random.choice(["A", "B", "TIE"])

    return decision, thoughts, full


def compute_elo(matches: List[Tuple[int, int, str]], k: int = 32) -> Dict[int, float]:
    """Compute Elo scores from match results.
    matches: list of (cell_a, cell_b, result) where result is 'A', 'B', or 'TIE'."""
    players = set()
    for a, b, _ in matches:
        players.add(a)
        players.add(b)
    elo = {p: 1500.0 for p in players}

    def expected(rating_a, rating_b):
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

    for a, b, res in matches:
        ra, rb = elo[a], elo[b]
        ea, eb = expected(ra, rb), expected(rb, ra)
        sa, sb = {'A': (1.0, 0.0), 'B': (0.0, 1.0)}.get(res, (0.5, 0.5))
        elo[a] = ra + k * (sa - ea)
        elo[b] = rb + k * (sb - eb)
    return elo


def run_pairwise(cell_descriptions: List[str], report: str, out_csv: str, names: List[str] = None,
                  ids: List[str] = None, seed: int = None, nworkers: int = 8, parallel: bool = True,
                  model: str = DEFAULT_MODEL, mode: str = "ad"):
    if seed is not None:
        random.seed(seed)

    # Create ordered pairs including flipped comparisons to reduce positional bias.
    pairs = []
    n = len(cell_descriptions)
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j))
            pairs.append((j, i))
    random.shuffle(pairs)
    print(f"Prepared {len(pairs)} comparisons ({len(pairs) // 2} unique pairs, including flipped).")

    names = names if names is not None else cell_descriptions[:]
    ids = ids if ids is not None else [""] * len(cell_descriptions)

    matches: List[Tuple[int, int, str]] = []
    llm_out_csv = out_csv.replace('.csv', '') + '_llm_outputs.csv'

    if parallel:
        print(f"Running in parallel with {nworkers} workers...")
        with ThreadPoolExecutor(max_workers=nworkers) as exe:
            futures = {
                exe.submit(call_judge_llm, cell_descriptions[i], cell_descriptions[j], report, model=model, mode=mode): (i, j)
                for i, j in pairs
            }
            with open(out_csv, 'w', newline='') as f, open(llm_out_csv, 'w', newline='', encoding='utf-8') as lf:
                writer, lwriter = csv.writer(f), csv.writer(lf)
                writer.writerow(['index_a', 'index_b', 'result', 'name_a', 'id_a', 'name_b', 'id_b', 'desc_a', 'desc_b'])
                lwriter.writerow(['index_a', 'index_b', 'name_a', 'id_a', 'name_b', 'id_b', 'decision', 'thoughts', 'full_text'])
                completed = 0
                for fut in as_completed(futures):
                    i, j = futures[fut]
                    decision, thoughts, full = fut.result()
                    writer.writerow([i, j, decision, names[i], ids[i], names[j], ids[j],
                                    cell_descriptions[i], cell_descriptions[j]])
                    lwriter.writerow([i, j, names[i], ids[i], names[j], ids[j], decision, thoughts, full])
                    f.flush()
                    lf.flush()
                    matches.append((i, j, decision))
                    completed += 1
                    if completed % 25 == 0:
                        print(f"Completed {completed}/{len(futures)} comparisons...")
    else:
        print("Running serially...")
        with open(out_csv, 'w', newline='') as f, open(llm_out_csv, 'w', newline='', encoding='utf-8') as lf:
            writer, lwriter = csv.writer(f), csv.writer(lf)
            writer.writerow(['index_a', 'index_b', 'result', 'name_a', 'id_a', 'name_b', 'id_b', 'desc_a', 'desc_b'])
            lwriter.writerow(['index_a', 'index_b', 'name_a', 'id_a', 'name_b', 'id_b', 'decision', 'thoughts', 'full_text'])
            for i, j in tqdm(pairs):
                decision, thoughts, full = None, "", ""
                for attempt in range(5):
                    try:
                        decision, thoughts, full = call_judge_llm(cell_descriptions[i], cell_descriptions[j], report, model=model, mode=mode)
                        break
                    except Exception:
                        time.sleep((2 ** attempt) + random.random())
                else:
                    decision = random.choice(['A', 'B', 'TIE'])
                writer.writerow([i, j, decision, names[i], ids[i], names[j], ids[j],
                                cell_descriptions[i], cell_descriptions[j]])
                lwriter.writerow([i, j, names[i], ids[i], names[j], ids[j], decision, thoughts, full])
                f.flush()
                lf.flush()
                matches.append((i, j, decision))
                if len(matches) % 25 == 0:
                    print(f"Completed {len(matches)} comparisons...")

    print(f"Computing Elo from {len(matches)} recorded matches...")
    elo = compute_elo(matches) if matches else {i: 1500.0 for i in range(len(cell_descriptions))}
    if not matches:
        print("Warning: no matches recorded — writing default Elo (1500) for all cells.")

    elo_csv = out_csv.replace('.csv', '') + '_elo.csv'
    with open(elo_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['index', 'name', 'cell_ontology_id', 'elo'])
        for idx, score in sorted(elo.items(), key=lambda kv: -kv[1]):
            writer.writerow([idx, names[idx], ids[idx], score])
    return elo


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--cell-types-csv', required=True, help='Path to CSV with columns CO, Cell-type, Name (one row per cell)')
    parser.add_argument('--report-file', required=True, help='Path to disease-context report text file')
    parser.add_argument('--out-csv', default='judgements.csv', help='Output CSV for pairwise judgements')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--nworkers', type=int, default=8, help='Number of parallel workers for pairwise judgements')
    parser.add_argument('--no-parallel', dest='parallel', action='store_false', help='Disable parallel execution')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL, help=f'OpenRouter model string (default: {DEFAULT_MODEL})')
    parser.add_argument('--mode', type=str, default='ad', choices=['ad', 'control'],
                        help='Judge mode: "ad" = AD relevance (default), "control" = healthy brain relevance')
    args = parser.parse_args()

    cells, names, ids = [], [], []
    with open(args.cell_types_csv, newline='') as f:
        for row in csv.DictReader(f):
            cid = row.get('CO') or row.get('co') or ''
            name = row.get('Name') or row.get('name') or row.get('Cell-type') or ''
            cells.append(name.strip())
            names.append(name.strip())
            ids.append(cid.strip())
    print(f"Loaded {len(cells)} cells from {args.cell_types_csv}")
    with open(args.report_file) as f:
        report = f.read()

    print(f"Using model: {args.model}, mode: {args.mode}")
    elo = run_pairwise(cells, report, args.out_csv, names=names, ids=ids, seed=args.seed,
                       nworkers=args.nworkers, parallel=args.parallel, model=args.model, mode=args.mode)
    for cell, score in sorted(elo.items(), key=lambda kv: -kv[1]):
        print(f"{score:.1f}\t{cell}")


if __name__ == '__main__':
    main()
