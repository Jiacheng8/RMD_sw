# Default preview cases: complex, measurable, fixed before inference

Latest user instruction2026-11-06: previous isolated portraits/handshake previews
were too simple and wasted time. Future previews default to genuinely complex
cases. A long descriptive portrait is not sufficient difficulty by itself.

Use `ops/h20-run-mmd2-complex-preview.sh` and
`ops/mmd2-complex-cases-v1.json` by default. The old simple MMD2 preview entry
requires explicit `--allow-legacy-simple-cases`; keep old outputs unchanged.

Eight authored diagnostic cases cover four-person hand/object interaction,
four full-body musicians, two-person mirror consistency, multi-line Chinese
and English text with price binding, exact counts/nesting/refraction, a
nine-person crossing with wheelchair/bicycles, veterinary contacts, and
two-person bicycle repair. Each has at least eight specific observable checks.
They are custom stress cases, not official benchmark IDs or scores. No seeds
or prompts are replaced after seeing failures. Keep both seed0/1 results.

Compare same new experiment0/50/100/150/200/250/300/350/400/450, same2-step
renoising sampler, batch4/native BF16 inference,1024 pixels without resampling.
GPU6 renders0/450/50/100/150;GPU7 renders200/250/300/350/400. Moving450 earlier
permits an early endpoint inspection; every stage still loads its own full
SHA-verified independent checkpoint. The full state is loaded strictly, so
inference stage ordering never carries weights between checkpoints.

Inference/CPU export/PNG transfer retain the existing full-SHA, exact stage,
160 raw images,240 pixel-identical pasted tiles, source binding and real-exit
gates. Ten-column panorama metadata must read10240x4416; case sheets5120x2240.
All experimental assets remain remote; Mac gets only requested final PNGs,
code/docs and small receipts. Training on4/5 continues unchanged.
