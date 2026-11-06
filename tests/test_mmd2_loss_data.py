"""Contract tests use scalar fixtures only; no fake experimental plots or metrics saved."""
import importlib.util
from pathlib import Path
import pytest
spec=importlib.util.spec_from_file_location('mmd2_loss',Path(__file__).resolve().parents[1]/'ops/h20-mmd2-loss-data.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def row(step):
    r=dict(step=step,skipped=False,grad_norm=1.,lr=2.83e-5)
    for k in [1,2]:
        r.update({f's{k}/gan/d_skipped':False,f's{k}/gan/g_active':step>50,f's{k}/gan/d_loss':1.})
        for name in m.NAMES:
            r.update({f's{k}/{name}/mmd2':.01,f's{k}/gan/{name}/d_loss':1.})
            if step>50:r[f's{k}/gan/{name}/g_loss']=1.
    return r


def payload(rows):
    return dict(MMD_encoders=m.NAMES,GAN_encoders=m.NAMES,init='official pretrained0',restart_boundaries=[],GAN_G_first_iteration=51,rows=rows)


def test_accepts_only_recorded_exact_two_streams():
    assert len(m.validate_payload(payload([row(1),row(2)])))==2


def test_historical_siglip_aim_data_cannot_be_relabelled():
    r=row(1);r['s1/siglip2/mmd2']=.01;r['s1/aimv2_huge/mmd2']=.01
    with pytest.raises(AssertionError,match='Actual MMD'):m.validate_payload(payload([r]))


def test_old_logits_or_generator_proxy_cannot_be_shown_as_active_G():
    rows=[row(i) for i in range(1,52)];rows[49]['s1/gan/g_active']=True
    with pytest.raises(AssertionError):m.validate_rows(rows)
    rows[49]['s1/gan/g_active']=False;assert len(m.validate_rows(rows))==51


def test_skips_nonfinite_gaps_or_extra_GAN_head_are_rejected():
    with pytest.raises(AssertionError,match='contiguous'):m.validate_rows([row(1),row(3)])
    r=row(1);r['grad_norm']=float('nan')
    with pytest.raises(AssertionError,match='Nonfinite'):m.validate_rows([r])
    r=row(1);r['s2/gan/aimv2_huge/d_loss']=1.
    with pytest.raises(AssertionError,match='Actual GAN'):m.validate_rows([r])


def test_trailing_mean_is_causal_and_not_future_centered():
    assert m.trailing_mean([1.,3.,8.],2)==[1.,2.,5.5]


def test_zero_norm_and_zero_softplus_stay_visible_for_diagnosis():
    r=row(1);r['grad_norm']=0.;r['s1/gan/dinov3_l/d_loss']=0.
    assert m.validate_rows([r])==[r]
