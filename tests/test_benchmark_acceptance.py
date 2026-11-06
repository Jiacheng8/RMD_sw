"""Regression gates reject finite G-only runs masquerading as complete spatial D/G updates."""
import copy
import importlib.util
from pathlib import Path

import pytest


_spec = importlib.util.spec_from_file_location(
    'h20_benchmark_acceptance', Path(__file__).resolve().parents[1] / 'ops/h20_benchmark_acceptance.py')
_helper = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_helper)


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def test_low_memory_residual_utilization_waits_for_fresh_zero_sample():
    timer = FakeClock()
    samples = [
        [[4, 0, 143771, 38], [5, 0, 143771, 0]],
        [[4, 0, 143771, 11], [5, 0, 143771, 0]],
        [[4, 0, 143771, 0], [5, 0, 143771, 0]],
    ]
    observed = []
    def read():
        sample = samples[len(observed)]
        observed.append(sample)
        return sample
    result = _helper.wait_for_idle_gpus(read, timer.sleep, timer.clock)
    assert result is samples[-1] and len(observed) == 3
    assert timer.sleeps == [0.25, 0.25] and timer.now == 0.5


def test_real_gpu_memory_occupancy_returns_immediately_without_sleep():
    timer = FakeClock()
    occupied = [[4, 33, 143771, 0], [5, 0, 143771, 0]]
    calls = []
    def read():
        calls.append(True)
        return occupied
    result = _helper.wait_for_idle_gpus(read, timer.sleep, timer.clock)
    assert result is occupied and len(calls) == 1
    assert timer.sleeps == [] and timer.now == 0


def test_persistent_low_memory_utilization_times_out_and_returns_last_nonidle_state():
    timer = FakeClock()
    observed = []
    def read():
        sample = [[4, 0, 143771, 38], [5, 0, 143771, 0]]
        observed.append(sample)
        return sample
    result = _helper.wait_for_idle_gpus(read, timer.sleep, timer.clock)
    assert result is observed[-1] and result[0][3] == 38
    assert timer.now == 5.0 and len(observed) == 21
    assert len(timer.sleeps) == 20 and all(seconds == 0.25 for seconds in timer.sleeps)


def test_memory_occupancy_appearing_during_refresh_is_not_waited_away():
    timer = FakeClock()
    samples = [[[4, 0, 143771, 20]], [[4, 500, 143771, 20]]]
    calls = []
    def read():
        sample = samples[len(calls)]
        calls.append(sample)
        return sample
    result = _helper.wait_for_idle_gpus(read, timer.sleep, timer.clock)
    assert result is samples[-1] and timer.sleeps == [0.25]


def test_already_idle_gpu_state_returns_without_sleep():
    timer = FakeClock()
    idle = [[4, 32, 143771, 0], [5, 0, 143771, 0]]
    assert _helper.wait_for_idle_gpus(lambda: idle, timer.sleep, timer.clock) is idle
    assert timer.sleeps == []


def record():
    iterations = []
    for iteration in range(1, 4):
        ranks = []
        for rank in range(2):
            logs = {'skipped': False, 'grad_norm': 0.4}
            for step in (1, 2):
                prefix = f's{step}/'
                logs.update({prefix+'gan/d_skipped': False, prefix+'gan/d_updates': iteration,
                             prefix+'gan/d_grad_norm': 0.3, prefix+'pixel_adv_grad_norm': 0.2,
                             prefix+'gan/d_loss': 1.1, prefix+'gan/g_loss': -0.1})
                for head in _helper.SPATIAL_HEADS:
                    logs.update({prefix+f'gan/{head}/d_loss': 1.1,
                                 prefix+f'gan/{head}/g_loss': -0.1,
                                 prefix+f'gan/{head}/d_grad_norm': 0.15})
            ranks.append({'rank': rank, 'iteration': iteration, 'seconds': 1.0, 'logs': logs,
                          'critic_updates_before': [iteration-1]*2, 'critic_updates_after': [iteration]*2})
        iterations.append(ranks)
    streams = [f's{k}/patch/{name}' for k in (1, 2) for name in ('dinov3_l', 'qwen3vl_4b')]
    stats = {'drift_probe_rows': 64, 'refreshed_rows': 96, 'spatial_refreshed_rows': 96}
    stats.update({name+'/'+suffix: 0.01 for name in streams for suffix in ('drift_mean', 'drift_max')})
    return {'spatial_patchgan': True, 'generation_steps': 2, 'world_size': 2, 'micro_batch': 2,
            'formal_training_started': False,
            'bounded_test_overrides': {'gan.g_start_step': {'configured': 50, 'actual': 0}},
            'steps': iterations, 'probe_refresh': [{'seconds': 0.1, 'stats': copy.deepcopy(stats)} for _ in range(2)],
            'probe_refresh_contract': {'requested_probe_rows': 64, 'expected_probe_rows': 64,
                                       'retained_rows': 96, 'world_size': 2, 'micro_batch': 2,
                                       'spatial_streams': streams}}


def check(value):
    return _helper.validate_benchmark_record(value, expected_spatial=True, expected_iterations=3)


def test_three_actual_two_rank_two_step_dg_updates_pass():
    result = check(record())
    assert result['all_rank_DG_updates_verified'] and result['complete_iterations'] == 3


def test_corrected_two_head_gate_requires_both_real_gan_branches_and_excludes_aim():
    value = record()
    value['gan_heads'] = list(_helper.SPATIAL_HEADS[:2])
    for ranks in value['steps']:
        for rank in ranks:
            for key in list(rank['logs']):
                if '/gan/aimv2_huge/' in key: del rank['logs'][key]
    _helper.validate_benchmark_record(value, expected_heads=_helper.SPATIAL_HEADS[:2])
    wrong = copy.deepcopy(value)
    wrong['steps'][0][0]['logs']['s1/gan/aimv2_huge/d_loss'] = 1.
    with pytest.raises(ValueError, match='exclude AIM'):
        _helper.validate_benchmark_record(wrong)
    del value['steps'][1][1]['logs']['s1/gan/qwen3vl_4b/g_loss']
    with pytest.raises(ValueError, match='Missing or nonfinite'):
        _helper.validate_benchmark_record(value)


def test_finite_generator_with_skipped_d_is_rejected():
    value = record()
    value['steps'][1][1]['logs']['s2/gan/d_skipped'] = True
    with pytest.raises(ValueError, match='D update was skipped'):
        check(value)


@pytest.mark.parametrize('metric', ['s1/gan/qwen3vl_4b/g_loss', 's2/gan/dinov3_l/d_loss',
                                   's1/gan/aimv2_huge/d_grad_norm', 's2/gan/d_grad_norm'])
def test_missing_branch_loss_or_gradient_metric_is_rejected(metric):
    value = record()
    del value['steps'][0][0]['logs'][metric]
    with pytest.raises(ValueError, match='Missing or nonfinite'):
        check(value)


@pytest.mark.parametrize('bad', [0.0, float('nan'), float('inf'), -0.1])
@pytest.mark.parametrize('metric', ['grad_norm', 's1/pixel_adv_grad_norm', 's2/gan/qwen3vl_4b/d_grad_norm'])
def test_g_and_d_gradient_norms_must_be_finite_positive(metric, bad):
    value = record()
    value['steps'][2][0]['logs'][metric] = bad
    with pytest.raises(ValueError):
        check(value)


def test_finite_losses_with_no_actual_d_optimizer_update_are_rejected():
    value = record()
    value['steps'][0][0]['critic_updates_after'][0] = 0
    with pytest.raises(ValueError, match='increase by exactly one'):
        check(value)


def test_logged_counter_does_not_substitute_for_observed_runtime_counter():
    value = record()
    value['steps'][0][0]['logs']['s1/gan/d_updates'] = 14
    with pytest.raises(ValueError, match='disagrees'):
        check(value)


def test_repeating_one_successful_iteration_three_times_does_not_pass():
    value = record()
    for iteration in (2, 3):
        value['steps'][iteration-1] = copy.deepcopy(value['steps'][0])
        for rank in value['steps'][iteration-1]:
            rank['iteration'] = iteration
    with pytest.raises(ValueError, match='discontinuous'):
        check(value)


def test_bounded_override_and_two_gpu_provenance_are_required():
    value = record()
    value['bounded_test_overrides']['gan.g_start_step']['actual'] = 50
    with pytest.raises(ValueError, match='bounded-only'):
        check(value)
    value = record()
    value['steps'][0].pop()
    with pytest.raises(ValueError, match='missing a rank'):
        check(value)


def test_legacy_nonspatial_capacity_record_still_accepts_original_schema():
    value = {'steps': [[{'iteration': i, 'logs': {'skipped': False, 'force': 0.2}}]
                       for i in (1, 2)], 'probe_refresh': [{'seconds': 0.1}]}
    result = _helper.validate_benchmark_record(value, expected_spatial=False, expected_iterations=2)
    assert result['complete_iterations'] == 2 and not result['spatial_patchgan']
    with pytest.raises(ValueError, match='spatial mode'):
        _helper.validate_benchmark_record(value, expected_spatial=True)
    value['probe_refresh'] = None
    _helper.validate_benchmark_record(value, expected_spatial=False, require_probe_refresh=False)
    with pytest.raises(ValueError, match='omitted probe'):
        _helper.validate_benchmark_record(value, expected_spatial=False)


@pytest.mark.parametrize('key', ['drift_probe_rows', 'refreshed_rows', 'spatial_refreshed_rows'])
def test_probe_and_refresh_must_execute_full_configured_counts(key):
    value = record()
    value['probe_refresh'][1]['stats'][key] -= 1
    with pytest.raises(ValueError):
        check(value)


def test_all_generation_step_spatial_drift_fields_are_required_and_finite():
    value = record()
    del value['probe_refresh'][0]['stats']['s2/patch/qwen3vl_4b/drift_max']
    with pytest.raises(ValueError, match='Missing or nonfinite'):
        check(value)
    value = record()
    value['probe_refresh'][1]['stats']['s1/patch/dinov3_l/drift_mean'] = float('nan')
    with pytest.raises(ValueError, match='Nonfinite'):
        check(value)


def test_probe_size_uses_actual_config_and_microbatch_rounding_not_hardcoded_64_96():
    value = record()
    value['probe_refresh_contract'].update(requested_probe_rows=13, expected_probe_rows=12, retained_rows=20)
    for probe in value['probe_refresh']:
        probe['stats'].update(drift_probe_rows=12, refreshed_rows=20, spatial_refreshed_rows=20)
    _helper.validate_benchmark_record(value, expected_probe_request=13, expected_retained_rows=20)
    with pytest.raises(ValueError, match='configuration'):
        _helper.validate_benchmark_record(value, expected_probe_request=64)


@pytest.mark.parametrize('line', ['torch.OutOfMemoryError: CUDA out of memory. Tried to allocate',
                                 '[rank1]: torch.cuda.OutOfMemoryError: CUDA out of memory',
                                 '[rank0]: RuntimeError: CUDA out of memory.'])
def test_only_actual_cuda_oom_exception_without_timeout_can_trigger_fallback(line):
    assert _helper.is_expected_cuda_oom('Traceback (most recent call last):\n'+line+'\n')
    assert not _helper.is_expected_cuda_oom(line, timed_out=True)


@pytest.mark.parametrize('text', ['RuntimeError: network query timed out; CUDA out of memory was mentioned',
                                 'note: torch.OutOfMemoryError: prior history',
                                 'torch.OutOfMemoryError: CPU allocator out of memory',
                                 'ValueError: CUDA out of memory phrase in unrelated error',
                                 'command --match torch.cuda.OutOfMemoryError: x'])
def test_unrelated_error_or_oom_wording_is_not_classified_as_oom(text):
    assert not _helper.is_expected_cuda_oom(text)


def test_fixed_weight_acceptance_requires_real_dg_without_norm_balancing():
    data = record()
    for iteration in data['steps']:
        for row in iteration:
            logs = row['logs']
            logs['pixel_gc/balance_mode'] = 'fixed'
            for k in [1, 2]:
                logs.pop(f's{k}/pixel_adv_grad_norm', None)
                logs[f's{k}/pixel_adv_lambda'] = 5e-5
    _helper.validate_benchmark_record(data, expected_spatial=True)
    data['steps'][0][0]['logs']['s1/gan/d_skipped'] = True
    with pytest.raises(ValueError):
        _helper.validate_benchmark_record(data, expected_spatial=True)
