"""Run real FSDP updates for speed/memory validation; never save weights or start eval/W&B.

torchrun --nproc_per_node=2 ops/h20-benchmark.py --micro-batch 2 --backend _flash_3_varlen_hub
"""
import argparse
import json
import math
import time
from pathlib import Path
import torch
import torch.distributed as dist

from rdm.train.launch import load_config
from rdm.utils.distributed import setup_distributed
from rdm.utils.logging import setup_logging
from rdm.utils.seed import fix_random_seeds
from zimage_distill.launch import build_trainer
from h20_benchmark_acceptance import validate_step_record, validate_benchmark_record

ap = argparse.ArgumentParser()
ap.add_argument('--micro-batch', type=int, choices=[1, 2, 4, 8, 16], required=True)
ap.add_argument('--backend', default='native', choices=['native', '_flash_3_varlen_hub'])
ap.add_argument('--iterations', type=int, default=3)
ap.add_argument('--checkpoint-group-size', type=int, choices=[0, 1, 2], default=1,
                help='0 disables DiT checkpointing; VAE checkpointing remains on')
ap.add_argument('--out-dir', type=Path, default=Path('/data/rdm-workspace/rdm-sets/control'))
ap.add_argument('--probe-refresh', action='store_true')
ap.add_argument('--config', default='z-image/configs/zimage_2step_1024_dlr1e-4_qwen.yaml')
ap.add_argument('--reference-root', default='/data/rdm-workspace/rdm-sets/zimage/zimage-ref-v1/reference_store_qwen')
a = ap.parse_args()
rank, world, local_rank = setup_distributed()
assert world == 2
setup_logging(rank=rank)
fix_random_seeds(0)
cfg = load_config(a.config)
active_batch = int(cfg.active_global_batch)
assert active_batch % (world * a.micro_batch) == 0
spatial_enabled = bool(getattr(cfg, 'spatial_gan', {}).get('enabled', False))
configured_g_start = cfg.gan.get('g_start_step', 50)
if spatial_enabled and a.iterations < 3:
    raise ValueError('Spatial Page acceptance requires three complete D/G iterations')
cfg.reference_root = a.reference_root
cfg.memory['micro_batch'] = a.micro_batch
cfg.zimage['attention_backend'] = a.backend
cfg.grad_checkpointing = a.checkpoint_group_size != 0
cfg.zimage['checkpoint_group_size'] = max(1, a.checkpoint_group_size)
# Benchmark the steady-state generator adversarial path without burning 50 formal steps.
if cfg.gan.get('enabled', False):
    cfg.gan['g_start_step'] = 0
device = f'cuda:{local_rank}'
trainer = build_trainer(cfg, device=device)
gan_heads = list(trainer.step_critics[0].names) if spatial_enabled else None
if spatial_enabled:
    assert all(list(c.names)==gan_heads for c in trainer.step_critics)
trainer.bootstrap()
dist.barrier()
rows = []
for i in range(a.iterations):
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.monotonic()
    critic_updates_before = [critic.updates for critic in trainer.step_critics] if spatial_enabled else None
    logs = trainer.step()
    critic_updates_after = [critic.updates for critic in trainer.step_critics] if spatial_enabled else None
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    row = {'rank': rank, 'iteration': i + 1, 'seconds': elapsed,
           'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
           'peak_reserved_bytes': torch.cuda.max_memory_reserved(), 'logs': logs}
    if spatial_enabled:
        row.update(critic_updates_before=critic_updates_before, critic_updates_after=critic_updates_after)
    try:
        validate_step_record(row, spatial=spatial_enabled, generation_steps=trainer.num_steps,
                             spatial_heads=gan_heads or ())
    except ValueError as error:
        print(json.dumps({'event': 'iteration_gate_failed', 'rank': rank,
                          'error': str(error), 'row': row}, default=str), flush=True)
        raise
    gathered = [None] * world
    dist.all_gather_object(gathered, row)
    if rank == 0:
        rows.append(gathered)
        print(json.dumps({'micro_batch': a.micro_batch, 'grad_accum': active_batch // (world * a.micro_batch),
                          'backend': a.backend, 'checkpoint_group_size': a.checkpoint_group_size,
                          'ranks': gathered}), flush=True)
extra = None
probe_contract = None
if a.probe_refresh:
    from rdm.sw_lmmd.launch import cache_from_config
    from rdm.sw_lmmd.refresh import probe_drift, refresh_retained
    window = trainer.schedule.peek()
    policy = cache_from_config(cfg)
    granule = world * a.micro_batch
    probe_contract = {'requested_probe_rows': policy.refresh_probe_rows,
                      'retained_rows': int(window.retained_ids.size), 'world_size': world,
                      'micro_batch': a.micro_batch,
                      'expected_probe_rows': min(int(policy.refresh_probe_rows), int(window.retained_ids.size))
                                             // granule * granule,
                      'spatial_streams': list(getattr(trainer, 'spatial_stream_names', []))}
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.monotonic()
    stats = probe_drift(trainer, window, policy.refresh_probe_rows)
    stats.update(refresh_retained(trainer, window))
    assert all(math.isfinite(v) for v in stats.values() if isinstance(v, (int, float))), stats
    torch.cuda.synchronize()
    extra = {'seconds': time.monotonic() - started, 'stats': stats,
             'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
             'peak_reserved_bytes': torch.cuda.max_memory_reserved()}
    gathered = [None] * world
    dist.all_gather_object(gathered, extra)
    extra = gathered
if rank == 0:
    a.out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f'-group{a.checkpoint_group_size}' if a.out_dir.name != 'control' else ''
    out = a.out_dir / f'benchmark-{a.backend}-micro{a.micro_batch}{suffix}.json'
    record = {'micro_batch': a.micro_batch, 'backend': a.backend,
              'checkpoint_group_size': a.checkpoint_group_size,
              'global_batch': active_batch, 'generator_gan_active': bool(cfg.gan.get('enabled', False)), 'steps': rows,
              'probe_refresh': extra, 'probe_refresh_contract': probe_contract, 'spatial_patchgan': spatial_enabled,
              'generation_steps': trainer.num_steps, 'world_size': world,
              'gan_heads': gan_heads,
              'MMD_encoders': list(trainer.base_names),
              'spatial_feature_contract': getattr(trainer, 'spatial_contract', None),
              'source_commit': (Path(__file__).resolve().parents[1]/'SOURCE_COMMIT').read_text().strip(),
              'config': str(Path(a.config).resolve()),
              'G_base_lr': float(cfg.lr), 'D_base_lr': float(cfg.gan['lr']),
              'adaptive_D_lr': bool(getattr(cfg, 'adaptive_d_lr', {}).get('enabled', False)),
              'formal_training_started': False,
              'bounded_test_overrides': ({'gan.g_start_step': {'configured': configured_g_start, 'actual': 0}}
                                         if cfg.gan.get('enabled', False) else {})}
    try:
        validate_benchmark_record(record, expected_spatial=spatial_enabled, expected_iterations=a.iterations,
                                  require_probe_refresh=a.probe_refresh)
    except ValueError as error:
        print(json.dumps({'event': 'benchmark_gate_failed', 'rank': rank,
                          'error': str(error), 'record': record}, default=str), flush=True)
        raise
    out.write_text(json.dumps(record, indent=2))
dist.destroy_process_group()
