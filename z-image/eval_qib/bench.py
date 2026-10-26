"""Qwen-Image-Bench (EMNLP 2026, arXiv 2605.28091): the pinned release, its prompts, its leaderboard.

The benchmark is 1000 prompts, each with a Chinese and an English version, and a fine-tuned
Qwen3.6-27B judge ("Q-Judger", ``Qwen/Qwen-Image-Bench`` on the Hub, 54.7 GB bf16). Every prompt
lists the facets it tests (``dims_en`` in ``metadata/bench_metadata.json``); the judge is called
once per level-1 pillar the prompt touches (3-5 calls, 4105 in all) and returns, after its
thinking, a JSON of facet scores 0 (Fail) / 1 (Pass) / 2 (Excel) / N/A, mapped to 0 / 60 / 100.
Facets average to sub-capabilities (L2), those to pillars (L1), pillars to a per-image total;
the benchmark scores are the means over images (``judge.py: compute_bench_scores``).

    L1 pillar             L2 sub-capabilities
    Quality               Realism, Detail, Resolution
    Aesthetics            Composition, Color Harmony, Lighting, Anatomical Portraiture,
                          Emotional Expression, Style Control
    Alignment             Attributes, Actions, Layout, Relations, Scene
    Real-world Fidelity   Fairness, Safety & Compliance, World Knowledge
    Creative Generation   Imagination, Feature Matching, Logical Resolution, Text Rendering,
                          Design Applications, Visual Storytelling

The leaderboard (paper Table 2, reproduced exactly by the release's ``compute_scores.py`` on
its stored judge outputs) was judged with the CHINESE prompt for every image: the prompt quoted
in the released judge outputs is ``prompt_cn`` (checked on all 1000 IDs). So ``lang="zh"`` is the
comparable setting; ``"en"`` is offered for a language-robustness check only.
"""
from __future__ import annotations

import json
from pathlib import Path

BENCH_REPO = "Qwen/Qwen-Image-Bench"
BENCH_REVISION = "d2493deb153b020cf169c7e3f57d15e4dd697038"   # dataset: code, metadata, prompts
JUDGE_REVISION = "1b77ff83564ac4e4e8140769eacff7ff73f35f3c"   # model: Q-Judger weights
PROMPTS_FILE = "qwen_image_bench_hf_v0518.jsonl"
PILLARS = ["Quality", "Aesthetics", "Alignment", "Real-world Fidelity", "Creative Generation"]

#: Paper Table 2 (0-100): pillar scores in PILLARS order, then the total.
LEADERBOARD = {
    "GPT Image 2":        (58.65, 67.53, 65.85, 57.38, 75.23, 64.69),
    "Nano Banana 2.0":    (54.77, 61.08, 62.40, 54.28, 67.05, 59.82),
    "GPT Image 1.5":      (55.14, 60.88, 61.72, 53.95, 66.35, 59.65),
    "Nano Banana Pro":    (55.67, 60.26, 61.25, 54.07, 66.23, 59.45),
    "Qwen Image 2.0 Pro": (54.39, 58.67, 59.28, 51.83, 64.94, 57.84),
    "Seedream 5.0":       (52.55, 58.40, 58.90, 51.92, 65.29, 57.22),
    "Seedream 4.5":       (54.41, 58.72, 57.31, 51.69, 60.64, 56.78),
    "Seedream 4.0":       (54.01, 58.81, 56.64, 51.05, 58.15, 56.21),
    "FLUX 2 Max":         (53.64, 56.85, 57.35, 49.35, 56.50, 55.33),
    "FLUX 2 Pro":         (52.30, 56.94, 57.01, 47.29, 56.18, 54.57),
    "GPT Image 1":        (52.34, 55.09, 56.28, 48.14, 55.78, 54.07),
    "Qwen Image 2512":    (51.76, 54.74, 52.72, 47.00, 50.19, 52.06),
    "Imagen 4.0 Ultra":   (50.90, 54.25, 54.02, 45.59, 51.14, 51.99),
    "HunyuanImage 3.0":   (50.35, 53.57, 52.00, 44.31, 49.12, 50.81),
    "Imagen 4.0":         (50.16, 52.68, 51.64, 44.84, 47.94, 50.29),
    "Qwen Image":         (48.44, 52.25, 50.72, 43.16, 47.30, 49.23),
    "Kling Image 2.1":    (49.11, 50.15, 49.18, 44.74, 44.67, 48.26),
    "GLM Image":          (49.26, 50.64, 47.90, 44.69, 45.23, 48.19),
}


def fetch_bench(dest) -> Path:
    """The release's code, metadata and prompt file (not its 18 GB of sample images)."""
    from huggingface_hub import snapshot_download
    path = snapshot_download(BENCH_REPO, repo_type="dataset", revision=BENCH_REVISION,
                             allow_patterns=["*.py", "metadata/*", PROMPTS_FILE, "requirements.txt"],
                             local_dir=str(dest))
    return Path(path)


def bench_prompts(bench_dir, lang: str = "zh") -> list[tuple[int, str]]:
    """``[(ID, prompt)]`` for the 1000 prompts, in ID order."""
    if lang not in ("zh", "en"):
        raise ValueError(f"lang must be zh or en, got {lang!r}")
    key = "prompt_cn" if lang == "zh" else "prompt_en"
    rows = [json.loads(line) for line in open(Path(bench_dir) / PROMPTS_FILE, encoding="utf-8")
            if line.strip()]
    out = sorted((int(r["ID"]), r[key]) for r in rows)
    if len(out) != 1000 or len({i for i, _ in out}) != 1000:
        raise ValueError(f"{bench_dir}/{PROMPTS_FILE}: expected 1000 distinct IDs, got {len(out)}")
    return out
