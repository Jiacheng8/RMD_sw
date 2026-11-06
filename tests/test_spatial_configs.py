"""Preserve original settings except explicit spatial architecture/loss/D-LR changes."""
import copy
from pathlib import Path

import pytest
import yaml


CONFIGS = Path(__file__).resolve().parents[1] / 'z-image/configs'
BASE = 'zimage_2step_1024_dlr1e-4'
VARIANTS = [('', [1.0, 1.0]), ('_stepw0.25', [0.25, 1.0])]
QWEN_TEXT_REPO = 'jiachengcui888/zimage-ref-qwen-text'
GLOBAL_ENCODERS = ['dinov3_l', 'siglip2', 'aimv2_huge']


def _read(variant, spatial):
    suffix = 'spatial_patchgan_h20_fa3' if spatial else 'qwen_h20_fa3'
    with (CONFIGS / f'{BASE}{variant}_{suffix}.yaml').open() as handle:
        value = yaml.safe_load(handle)
    assert isinstance(value, dict)
    return value


def _unchanged_settings(config):
    value = copy.deepcopy(config)
    # Exactly these environment/experiment paths and newly authorized architecture
    # knobs may differ. In particular, do not discard all zimage/memory/gan blocks.
    for key in ('exp_name', 'output_dir', 'reference_root', 'spatial_gan'):
        value.pop(key, None)
    value['zimage'].pop('checkpoint_group_size', None)
    # Latest user authorization changes only the GAN objective and D learning rate.
    # Pin their exact new values below; retain every other GAN and MMD setting.
    value['gan'].pop('loss', None)
    value['gan'].pop('lr', None)
    return value


def _differences(expected, actual, path=''):
    """Show precise nested keys if a future edit accidentally changes training behavior."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        result = []
        for key in sorted(set(expected) | set(actual)):
            child = f'{path}.{key}' if path else key
            if key not in expected or key not in actual:
                result.append(child)
            else:
                result.extend(_differences(expected[key], actual[key], child))
        return result
    return [] if expected == actual else [path]


@pytest.mark.parametrize('variant,step_weights', VARIANTS)
def test_only_paths_checkpoint_group_spatial_block_and_user_gan_overrides_change(variant, step_weights):
    original, spatial = _read(variant, False), _read(variant, True)
    expected, actual = _unchanged_settings(original), _unchanged_settings(spatial)
    assert actual == expected, f'Unauthorized setting changes: {_differences(expected, actual)}'
    assert spatial['zimage']['checkpoint_group_size'] == 2
    assert spatial['spatial_gan']['enabled'] is True
    assert spatial['exp_name'] != original['exp_name'] and 'spatial-patchgan' in spatial['exp_name']
    for key in ('output_dir', 'reference_root'):
        assert isinstance(spatial[key], str) and Path(spatial[key]).is_absolute()
    assert Path(spatial['reference_root']).name == 'reference_store_qwen'
    assert spatial['qwen_text_repo'] == original['qwen_text_repo'] == QWEN_TEXT_REPO
    assert spatial['encoders'] == original['encoders'] == GLOBAL_ENCODERS
    assert spatial['student']['step_weights'] == original['student']['step_weights'] == step_weights


@pytest.mark.parametrize('variant,_weights', VARIANTS)
def test_lr_gan_window_batch_and_resume_contract_remain_pinned(variant, _weights):
    """Name the user-visible invariants, even if both compared files are edited together."""
    for spatial in (False, True):
        config = _read(variant, spatial)
        assert config['lr'] == pytest.approx(2.83e-6)
        assert config['lr_schedule'] == {'name': 'cosine', 'warmup_steps': 100,
                                         'min_lr_ratio': 0.1, 'total_steps': None}
        assert config['gan']['lr'] == pytest.approx(1.0e-5 if spatial else 1.0e-4)
        assert config['gan']['loss'] == ('ns' if spatial else 'hinge')
        assert config['gan']['weight'] == 1.5 and config['gan']['adaptive'] is True
        assert config['gan']['g_start_step'] == 50
        assert config['gan']['betas'] == [0.0, 0.99]
        assert config['active_global_batch'] == config['window']['stride'] == 32
        assert config['window']['size'] == 128
        assert config['memory']['micro_batch'] == 2
        assert config['memory']['shard'] == 'fsdp' and config['memory']['grad_reduce'] == 'none'
        assert config['memory']['optimizer'] == 'adamw8bit'
        assert config['zimage']['attention_backend'] == '_flash_3_varlen_hub'
        assert config['zimage']['feature_pool'] == 'auto' and config['zimage']['feature_quantize'] is True
        assert config['joint'] is True and config['img_size'] == 1024
        assert config['steps'] == config['save_freq'] == 500
        assert config['save_resume'] is True and config['resume_every'] == 50
        assert config['save_cache'] is False and config['save_optimizer'] is False


def test_spatial_a_b_still_differ_only_in_first_step_weight_and_experiment_name():
    a, b = _read('', True), _read('_stepw0.25', True)
    assert a['student']['step_weights'] == [1.0, 1.0]
    assert b['student']['step_weights'] == [0.25, 1.0]
    b['exp_name'] = a['exp_name']
    b['student']['step_weights'] = a['student']['step_weights']
    assert b == a


def test_adaptive_variant_preserves_every_existing_a_setting():
    base = _read('', True)
    adaptive = yaml.safe_load((CONFIGS/'zimage_2step_1024_spatial_patchgan_adaptive_h20_fa3.yaml').read_text())
    policy = adaptive.pop('adaptive_d_lr')
    assert adaptive.pop('adaptive_d_lr_allow_legacy_resume') is False
    assert adaptive == base, f'Adaptive policy changed unrelated settings: {_differences(base, adaptive)}'
    assert policy['enabled'] is True
    assert policy['window_size'] == 10 and policy['cooldown_steps'] == 20
    assert policy['min_lr'] == 1e-7 and policy['max_lr'] == 2e-5
    assert policy['decrease_factor'] == 0.5 and policy['increase_factor'] == 1.25
    assert policy['min_abs_change'] == 0.05 and policy['min_relative_change'] == 0.20


def test_two_head_variants_preserve_three_mmd_streams_and_every_other_setting():
    legacy = _read('', True)
    two = yaml.safe_load((CONFIGS/'zimage_2step_1024_twohead_patchgan_h20_fa3.yaml').read_text())
    adaptive = yaml.safe_load((CONFIGS/'zimage_2step_1024_twohead_patchgan_adaptive_h20_fa3.yaml').read_text())
    assert two['encoders'] == adaptive['encoders'] == GLOBAL_ENCODERS
    assert two['spatial_gan']['global_gan_encoders'] == []
    assert adaptive['spatial_gan']['global_gan_encoders'] == []
    compared = copy.deepcopy(two)
    compared['exp_name'] = legacy['exp_name']
    del compared['spatial_gan']['global_gan_encoders']
    assert compared == legacy, _differences(legacy, compared)
    assert adaptive['adaptive_d_lr'] == yaml.safe_load(
        (CONFIGS/'zimage_2step_1024_spatial_patchgan_adaptive_h20_fa3.yaml').read_text())['adaptive_d_lr']
    assert adaptive.pop('adaptive_d_lr_allow_legacy_resume') is False
    del adaptive['adaptive_d_lr']
    assert adaptive['exp_name'] != two['exp_name']
    adaptive['exp_name'] = two['exp_name']
    assert adaptive == two, _differences(two, adaptive)


def test_latest_generator_lr_tenfold_preserves_corrected_disc_and_all_mmd_settings():
    base = yaml.safe_load((CONFIGS/'zimage_2step_1024_twohead_patchgan_adaptive_h20_fa3.yaml').read_text())
    new = yaml.safe_load((CONFIGS/'zimage_2step_1024_twohead_glr10_h20_fa3.yaml').read_text())
    assert new['lr'] == pytest.approx(2.83e-5)
    assert new['gan']['lr'] == pytest.approx(1e-5)
    assert new['spatial_gan']['global_gan_encoders'] == []
    assert new['encoders'] == GLOBAL_ENCODERS
    assert new['adaptive_d_lr']['enabled'] is True
    assert new['grad_clip'] == 4.
    assert new['exp_name'] != base['exp_name']
    new['exp_name'],new['lr'] = base['exp_name'],base['lr']
    assert new == base, _differences(base,new)
