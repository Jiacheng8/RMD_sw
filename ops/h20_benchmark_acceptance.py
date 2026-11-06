"""Pure scalar/JSON gates for bounded benchmarks; no torch, GPU, or side effects."""
from __future__ import annotations

import math
import numbers
import re


SPATIAL_HEADS = ('dinov3_l', 'qwen3vl_4b', 'aimv2_huge')


def wait_for_idle_gpus(read_state, sleep_fn, clock_fn, *, timeout_seconds=5.0,
                       interval_seconds=0.25, memory_limit_mib=32):
    """Wait only for a low-memory NVML utilization sample to refresh.

    ``read_state`` returns rows ``[gpu, memory_used_mib, total_mib, utilization]``.
    Return the latest state unchanged; the caller still applies its idle gate.
    Any memory above the limit returns immediately, including if it appears on a
    later poll. Persistent nonzero utilization returns the last state at timeout.
    No device/process mutation or occupancy bypass is performed. Callbacks own
    their query timeouts; ``clock_fn`` must be a monotonic clock.
    """
    _require(isinstance(timeout_seconds, numbers.Real) and math.isfinite(timeout_seconds)
             and 0 <= timeout_seconds <= 5.0, 'Idle refresh timeout must be between0 and5 seconds')
    _require(isinstance(interval_seconds, numbers.Real) and math.isfinite(interval_seconds)
             and interval_seconds > 0, 'Idle refresh interval must be positive')
    _require(isinstance(memory_limit_mib, numbers.Real) and math.isfinite(memory_limit_mib)
             and memory_limit_mib >= 0, 'Idle memory limit must be finite and nonnegative')
    deadline = clock_fn() + timeout_seconds
    while True:
        state = read_state()
        _require(isinstance(state, (list, tuple)) and bool(state), 'GPU state must contain selected GPU rows')
        for row in state:
            _require(isinstance(row, (list, tuple)) and len(row) >= 4,
                     'GPU state rows must contain index,memory,total,utilization')
            _require(all(isinstance(row[index], numbers.Real) and math.isfinite(row[index])
                         and row[index] >= 0 for index in (1, 3)), 'GPU memory/utilization must be finite nonnegative numbers')
        if any(row[1] > memory_limit_mib for row in state) or all(row[3] == 0 for row in state):
            return state
        remaining = deadline - clock_fn()
        if remaining <= 0:
            return state
        sleep_fn(min(interval_seconds, remaining))


def is_expected_cuda_oom(log_text, *, timed_out=False):
    """An actual traceback exception, never a timeout or incidental OOM wording."""
    if timed_out:
        return False
    prefix = r'^(?:\[rank\d+\]:\s*)?'
    exception = r'(?:torch(?:\.cuda)?\.OutOfMemoryError|RuntimeError):\s*CUDA out of memory\b'
    return re.search(prefix + exception, log_text, flags=re.MULTILINE) is not None


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _finite_tree(value, path='record'):
    if isinstance(value, dict):
        for key, item in value.items():
            _finite_tree(item, f'{path}.{key}')
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _finite_tree(item, f'{path}[{index}]')
    elif isinstance(value, numbers.Real) and not math.isfinite(value):
        raise ValueError(f'Nonfinite benchmark value: {path}')


def _number(mapping, name, *, positive=False):
    value = mapping.get(name)
    _require(isinstance(value, numbers.Real) and not isinstance(value, bool) and math.isfinite(value),
             f'Missing or nonfinite numeric metric: {name}')
    _require(not positive or value > 0, f'Gradient norm must be positive: {name}')
    return value


def _counts(value, generation_steps, name):
    _require(isinstance(value, list) and len(value) == generation_steps,
             f'{name} must contain one update count per generation step')
    _require(all(isinstance(count, numbers.Integral) and not isinstance(count, bool) and count >= 0
                 for count in value), f'{name} must contain nonnegative integer counts')
    return value


def validate_step_record(row, *, spatial=False, generation_steps=2, spatial_heads=SPATIAL_HEADS):
    """Require finite completed G, and actual D/G activity for spatial Page acceptance."""
    _finite_tree(row)
    logs = row.get('logs', {})
    _require(logs.get('skipped') is False, 'Generator update was skipped or not recorded')
    if not spatial:
        return
    spatial_heads = tuple(spatial_heads)
    _require(spatial_heads in (SPATIAL_HEADS, SPATIAL_HEADS[:2]), 'Unexpected GAN head contract')
    if spatial_heads == SPATIAL_HEADS[:2]:
        _require(not any('/gan/aimv2_huge/' in key for key in logs), 'Two-head GAN must exclude AIM GAN metrics')
    before = _counts(row.get('critic_updates_before'), generation_steps, 'critic_updates_before')
    after = _counts(row.get('critic_updates_after'), generation_steps, 'critic_updates_after')
    _number(logs, 'grad_norm', positive=True)
    for k in range(1, generation_steps + 1):
        prefix = f's{k}/'
        _require(after[k-1] == before[k-1] + 1,
                 f'{prefix}critic update count must increase by exactly one')
        _require(logs.get(prefix + 'gan/d_skipped') is False,
                 f'{prefix}D update was skipped or not recorded')
        _require(_number(logs, prefix + 'gan/d_updates') == after[k-1],
                 f'{prefix}logged D update count disagrees with the runtime counter')
        _number(logs, prefix + 'gan/d_grad_norm', positive=True)
        if logs.get('pixel_gc/balance_mode') == 'fixed':
            _number(logs, prefix + 'pixel_adv_lambda', positive=True)
            _require(prefix + 'pixel_adv_grad_norm' not in logs, 'Fixed mode must omit balancing norms')
        else:
            _number(logs, prefix + 'pixel_adv_grad_norm', positive=True)
        for name in ('gan/d_loss', 'gan/g_loss'):
            _number(logs, prefix + name)
        for head in spatial_heads:
            _number(logs, prefix + f'gan/{head}/d_grad_norm', positive=True)
            for kind in ('d_loss', 'g_loss'):
                _number(logs, prefix + f'gan/{head}/{kind}')


def validate_benchmark_record(record, *, expected_spatial=None, expected_iterations=None,
                              require_probe_refresh=True, expected_probe_request=None,
                              expected_retained_rows=None, expected_heads=None):
    """Validate every rank/iteration and counter continuity, preserving old nonspatial files."""
    _finite_tree(record)
    spatial = record.get('spatial_patchgan', False)
    _require(isinstance(spatial, bool), 'spatial_patchgan must be a boolean')
    if expected_spatial is not None:
        _require(spatial == expected_spatial, 'Benchmark spatial mode does not match its configuration')
    steps = record.get('steps')
    _require(isinstance(steps, list) and steps, 'Benchmark has no completed iterations')
    if expected_iterations is not None:
        _require(len(steps) == expected_iterations, 'Benchmark iteration count differs from the request')
    generation_steps = record.get('generation_steps', 2)
    spatial_heads = tuple(record.get('gan_heads') or SPATIAL_HEADS)
    if expected_heads is not None:
        _require(spatial_heads == tuple(expected_heads), 'Executed GAN heads differ from expected heads')
    if spatial:
        _require(generation_steps == 2, 'Page acceptance requires two generation steps')
        _require(len(steps) >= 3, 'Spatial acceptance requires at least three complete D/G iterations')
        _require(record.get('world_size') == 2, 'Spatial acceptance requires two actual ranks')
        _require(record.get('formal_training_started') is False, 'Bounded benchmark must not be formal training')
        override = record.get('bounded_test_overrides', {}).get('gan.g_start_step', {})
        _require(override.get('actual') == 0 and isinstance(override.get('configured'), int),
                 'Record the actual bounded-only gan.g_start_step=0 override')
    previous = {}
    for index, ranks in enumerate(steps):
        _require(isinstance(ranks, list) and ranks, f'Iteration {index+1} has no rank records')
        if spatial:
            _require(len(ranks) == 2, f'Iteration {index+1} is missing a rank')
        for rank, row in enumerate(ranks):
            validate_step_record(row, spatial=spatial, generation_steps=generation_steps,
                                 spatial_heads=spatial_heads)
            if spatial:
                _require(row.get('rank') == rank and row.get('iteration') == index+1,
                         'Rank/iteration provenance is missing or out of order')
                before, after = row['critic_updates_before'], row['critic_updates_after']
                if rank in previous:
                    _require(before == previous[rank], 'Critic update counters are discontinuous across iterations')
                previous[rank] = after
        if spatial:
            _require(ranks[0]['critic_updates_before'] == ranks[1]['critic_updates_before'] and
                     ranks[0]['critic_updates_after'] == ranks[1]['critic_updates_after'],
                     'Replicated critic counters differ across ranks')
    if require_probe_refresh or spatial:
        _require(record.get('probe_refresh') is not None, 'Benchmark omitted probe/refresh')
    if spatial:
        contract = record.get('probe_refresh_contract', {})
        request = _number(contract, 'requested_probe_rows', positive=True)
        retained = _number(contract, 'retained_rows', positive=True)
        world = _number(contract, 'world_size', positive=True)
        micro = _number(contract, 'micro_batch', positive=True)
        _require(world == record['world_size'] and micro == record.get('micro_batch'),
                 'Probe contract world/micro differs from the executed benchmark')
        if expected_probe_request is not None:
            _require(request == expected_probe_request, 'Probe row request differs from its configuration')
        if expected_retained_rows is not None:
            _require(retained == expected_retained_rows, 'Retained window differs from its configuration')
        granule = world * micro
        expected_rows = int(min(request, retained) // granule * granule)
        _require(expected_rows > 0 and contract.get('expected_probe_rows') == expected_rows,
                 'Probe expected rows do not match requested/window/rank/micro rounding')
        streams = {f's{k}/patch/{head}' for k in range(1, generation_steps+1)
                   for head in ('dinov3_l', 'qwen3vl_4b')}
        _require(set(contract.get('spatial_streams', [])) == streams,
                 'Probe contract omitted a spatial generation-step stream')
        probes = record['probe_refresh']
        _require(isinstance(probes, list) and len(probes) == world,
                 'Probe/refresh is missing an actual rank')
        for probe in probes:
            stats = probe.get('stats', {})
            _require(_number(stats, 'drift_probe_rows') == expected_rows,
                     'Probe did not execute the expected row count')
            for key in ('refreshed_rows', 'spatial_refreshed_rows'):
                _require(_number(stats, key) == retained, f'{key} did not refresh the full retained window')
            for stream in streams:
                for suffix in ('drift_mean', 'drift_max'):
                    _number(stats, stream + '/' + suffix)
    return {'spatial_patchgan': spatial, 'complete_iterations': len(steps),
            'generation_steps': generation_steps, 'all_rank_DG_updates_verified': spatial}
