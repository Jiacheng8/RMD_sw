"""New-run contracts and exact scheduler endpoints; no pretrained assets required."""
import importlib.util
from pathlib import Path
import sys
import pytest
import yaml
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('fixed_config_under_test', ROOT/'rdm/sw_lmmd/config.py')
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
CONFIG = ROOT/'z-image/configs/zimage_2step_1024_dino_qwenvl_k512_b64_fixed_from0.yaml'


def test_batch_window_fixed_weight_and_exact_lr_endpoints():
    cfg = yaml.safe_load(CONFIG.read_text())
    assert cfg['window']['size'] == 512
    assert cfg['window']['stride'] == cfg['active_global_batch'] == 64
    assert cfg['active_global_batch'] // (2 * cfg['memory']['micro_batch']) == 16
    assert cfg['gan']['adaptive'] is False and cfg['gan']['weight'] == 5e-5
    assert cfg['adaptive_d_lr']['enabled'] is False
    assert cfg['encoders'] == ['dinov3_l', 'qwen3vl_4b']
    assert cfg['gan']['grad_clip'] == cfg['grad_clip'] == 4
    g = module.LRSchedule(**cfg['lr_schedule'])
    d = module.LRSchedule(**cfg['discriminator_lr_schedule'])
    for schedule in (g, d): schedule.validate()
    assert g.factor(99) * cfg['lr'] == pytest.approx(2.83e-5)
    assert g.factor(499) * cfg['lr'] == pytest.approx(1.415e-5)
    assert d.factor(0) * cfg['gan']['lr'] == 1e-5
    assert d.factor(499) * cfg['gan']['lr'] == 5e-6
    assert g.factor(700) == d.factor(700) == .5
    assert all(g.factor(i+1) <= g.factor(i) for i in range(100, 499))
    assert all(d.factor(i+1) <= d.factor(i) for i in range(499))


def test_mmd_only_ablation_changes_only_identity_and_disables_gan():
    together = yaml.safe_load(CONFIG.read_text())
    alone = yaml.safe_load((CONFIG.parent/'zimage_2step_1024_dino_qwenvl_k512_b64_mmd_only_from0.yaml').read_text())
    assert alone['exp_name'] != together['exp_name']
    assert alone['gan']['enabled'] is False and alone['spatial_gan']['enabled'] is False
    assert together['gan']['enabled'] is True and together['spatial_gan']['enabled'] is True
    assert not alone.get('load_from') and not alone.get('resume_from')
    alone['exp_name'] = together['exp_name']
    alone['gan']['enabled'] = True
    alone['spatial_gan']['enabled'] = True
    assert alone == together, 'Ablation must preserve all non-GAN training settings'
