"""Rebuild a training pack's Qwen3 prompt contexts: length-bucketed, one process per GPU.

    python -m zimage_distill.contexts build --pack <pack> --rank R --world W   # one per GPU
    python -m zimage_distill.contexts finalize --pack <pack>                   # merge the indices

The context of a prompt is ``ZImagePipeline.encode_prompt(prompt, max_sequence_length=512)``:
the Qwen3 chat template (``enable_thinking=True``), the second-to-last hidden state, kept for the
real tokens only. The pipeline pads every prompt to 512 tokens; here each batch holds prompts of
similar length and is padded only to its own longest. Padding is on the right and the attention
is causal, so a real token never sees a pad: the kept states are the same function of the prompt
(they differ from a 512-padded encode only by bf16 rounding, as any re-encode on another GPU
does). With ~100 real tokens per prompt that is ~5x less compute.

Layout (``ragged-flat-v1``), in ``<pack>/contexts/``:

    manifest.json        {"format", "dim", "num_prompts", "prompt_hash", "shards": [...]}
    shard_{r:02d}.bin    rank r's contexts, (tokens, 2560) bf16 stored as int16, back to back
    index.npy            (P, 3) int64: shard, first token, token count -- per prompt id

A rank writes its shard and ``index_{r:02d}.npy`` and only then ``done_{r:02d}.json``; a relaunch
skips finished ranks and redoes unfinished ones from scratch.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

FORMAT = "ragged-flat-v1"
DIM = 2560


def tokenize(tokenizer, prompts: list, max_sequence_length: int = 512) -> list:
    """Each prompt's token ids, exactly as the pipeline builds them (before padding)."""
    text = [tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                          add_generation_prompt=True, enable_thinking=True)
            for p in prompts]
    return tokenizer(text, truncation=True, max_length=max_sequence_length)["input_ids"]


def plan_batches(lengths, max_tokens: int, max_batch: int = 256) -> list:
    """Prompt indices sorted by length, cut into batches of ``len(batch) * longest <= max_tokens``."""
    order = np.argsort(np.asarray(lengths), kind="stable")
    batches, cur, longest = [], [], 0
    for i in order:
        n = int(lengths[i])
        if cur and (max(longest, n) * (len(cur) + 1) > max_tokens or len(cur) >= max_batch):
            batches.append(cur)
            cur, longest = [], 0
        cur.append(int(i))
        longest = max(longest, n)
    if cur:
        batches.append(cur)
    return batches


@torch.no_grad()
def encode_batch(model, ids: list, pad_id: int, device) -> list:
    """Right-padded batch -> each prompt's second-to-last hidden state on its real tokens."""
    longest = max(len(x) for x in ids)
    inp = torch.full((len(ids), longest), pad_id, dtype=torch.long)
    mask = torch.zeros((len(ids), longest), dtype=torch.bool)
    for j, x in enumerate(ids):
        inp[j, :len(x)] = torch.tensor(x)
        mask[j, :len(x)] = True
    hidden = model(input_ids=inp.to(device), attention_mask=mask.to(device),
                   output_hidden_states=True).hidden_states[-2]
    return [hidden[j, :len(x)] for j, x in enumerate(ids)]


def build(pack: Path, rank: int, world: int, repo: str | None, max_tokens: int, device="cuda",
          tokenizer=None, model=None) -> None:
    """Rank ``rank``'s share of the contexts (``tokenizer`` / ``model`` injected by the tests)."""
    from .store import read_prompts

    out = pack / "contexts"
    out.mkdir(exist_ok=True)
    done = out / f"done_{rank:02d}.json"
    if done.exists():
        print(f"[contexts {rank}/{world}] already done", flush=True)
        return
    prompts = read_prompts(pack / "prompts.jsonl")
    if tokenizer is None or model is None:
        from transformers import AutoModel, AutoTokenizer

        from .model import ZIMAGE_REPO, resolve_snapshot
        root = resolve_snapshot(repo or ZIMAGE_REPO)
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(root, "tokenizer"))
        model = AutoModel.from_pretrained(os.path.join(root, "text_encoder"),
                                          torch_dtype=torch.bfloat16).to(device).eval()
    t0 = time.time()
    ids = tokenize(tokenizer, prompts)
    lengths = np.array([len(x) for x in ids])
    # every rank plans the same global batches and takes every world-th: balanced, disjoint
    batches = plan_batches(lengths, max_tokens)[rank::world]
    mine = sum(len(b) for b in batches)
    print(f"[contexts {rank}/{world}] {len(prompts)} prompts tokenized in {time.time() - t0:.0f} s "
          f"(mean {lengths.mean():.0f}, max {lengths.max()} tokens); this rank: {mine} prompts in "
          f"{len(batches)} batches", flush=True)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    shard = out / f"shard_{rank:02d}.bin"
    index = np.zeros((mine, 4), dtype=np.int64)          # prompt, shard, first token, count
    row, offset, t0 = 0, 0, time.time()
    with open(shard, "wb") as f:
        for b, batch in enumerate(batches):
            for p, ctx in zip(batch, encode_batch(model, [ids[i] for i in batch], pad_id, device)):
                f.write(ctx.contiguous().view(torch.int16).cpu().numpy().tobytes())
                index[row] = (p, rank, offset, ctx.shape[0])
                offset += ctx.shape[0]
                row += 1
            if b % 200 == 0:
                print(f"[contexts {rank}/{world}] {row}/{mine} prompts, "
                      f"{row / max(time.time() - t0, 1e-9):.0f} prompts/s", flush=True)
    np.save(out / f"index_{rank:02d}.npy", index)
    done.write_text(json.dumps({"prompts": int(mine), "tokens": int(offset),
                                "seconds": round(time.time() - t0, 1)}))
    print(f"[contexts {rank}/{world}] done: {mine} prompts, {offset} tokens, "
          f"{offset * DIM * 2 / 1e9:.1f} GB in {time.time() - t0:.0f} s", flush=True)


def finalize(pack: Path, world: int) -> None:
    """Merge the ranks' indices into one ``(P, 3)`` table and write the manifest."""
    from .store import read_prompts

    out = pack / "contexts"
    num_prompts = len(read_prompts(pack / "prompts.jsonl"))
    missing = [r for r in range(world) if not (out / f"done_{r:02d}.json").exists()]
    if missing:
        raise SystemExit(f"ranks {missing} have not finished; relaunch build for them")
    parts = np.concatenate([np.load(out / f"index_{r:02d}.npy") for r in range(world)])
    if parts.shape[0] != num_prompts or np.unique(parts[:, 0]).size != num_prompts:
        raise SystemExit(f"the ranks cover {np.unique(parts[:, 0]).size} distinct prompts of "
                         f"{num_prompts}")
    index = np.zeros((num_prompts, 3), dtype=np.int64)
    index[parts[:, 0]] = parts[:, 1:]
    for r in range(world):
        tokens = int(index[index[:, 0] == r, 2].sum())
        size = (out / f"shard_{r:02d}.bin").stat().st_size
        if size != tokens * DIM * 2:
            raise SystemExit(f"shard_{r:02d}.bin is {size} bytes, its index says {tokens * DIM * 2}")
    np.save(out / "index.npy", index)
    prepare = json.loads((pack / "prepare.json").read_text()) if (pack / "prepare.json").exists() \
        else {}
    (out / "manifest.json").write_text(json.dumps({
        "format": FORMAT, "dim": DIM, "dtype": "bfloat16", "num_prompts": num_prompts,
        "prompt_hash": prepare.get("prompt_hash"), "max_sequence_length": 512,
        "shards": [f"shard_{r:02d}.bin" for r in range(world)]}, indent=2))
    print(f"[contexts] {num_prompts} prompts, {int(index[:, 2].sum())} tokens "
          f"(mean {index[:, 2].mean():.0f}) -> {out}", flush=True)


class FlatContexts:
    """Read-only access to a ``ragged-flat-v1`` context directory, by prompt id."""

    def __init__(self, directory):
        directory = Path(directory)
        self.manifest = json.loads((directory / "manifest.json").read_text())
        if self.manifest.get("format") != FORMAT:
            raise ValueError(f"{directory}: format {self.manifest.get('format')!r}, not {FORMAT}")
        self.dim = int(self.manifest["dim"])
        self.index = np.load(directory / "index.npy")
        self.shards = [np.memmap(directory / s, dtype=np.int16, mode="r").reshape(-1, self.dim)
                       for s in self.manifest["shards"]]

    def __len__(self) -> int:
        return self.index.shape[0]

    def get(self, prompt_id: int) -> torch.Tensor:
        shard, first, count = (int(v) for v in self.index[int(prompt_id)])
        rows = np.array(self.shards[shard][first:first + count])
        return torch.from_numpy(rows).view(torch.bfloat16)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("stage", choices=["build", "finalize"])
    ap.add_argument("--pack", type=Path, required=True, help="the pack root (holds prompts.jsonl)")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--repo", default=None, help="Z-Image-Turbo snapshot dir or repo id")
    ap.add_argument("--max-tokens", type=int, default=16384,
                    help="tokens per batch (padded); all 37 hidden states are kept transiently")
    a = ap.parse_args()
    if a.stage == "build":
        build(a.pack, a.rank, a.world, a.repo, a.max_tokens)
    else:
        finalize(a.pack, a.world)


if __name__ == "__main__":
    main()
