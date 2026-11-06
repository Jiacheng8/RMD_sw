"""Validate already recorded MMD2 scalar data; stdlib only, no training instrumentation."""
import math

NAMES=['dinov3_l','qwen3vl_4b']
LABELS={'dinov3_l':'DINOv3','qwen3vl_4b':'Qwen-VL'}


def finite(value):
    if isinstance(value,dict):
        for v in value.values():finite(v)
    elif isinstance(value,list):
        for v in value:finite(v)
    elif isinstance(value,(int,float)):assert math.isfinite(value),'Nonfinite recorded metric'


def validate_rows(rows):
    assert rows and [r['step'] for r in rows]==list(range(1,len(rows)+1)), 'Require contiguous actual completed iterations from0'
    for row in rows:
        finite(row)
        assert row['skipped'] is False and row['grad_norm']>=0
        for step in [1,2]:
            assert row[f's{step}/gan/d_skipped'] is False
            assert bool(row[f's{step}/gan/g_active'])==(row['step']>50)
            keys={k.removeprefix(f's{step}/').removesuffix('/mmd2') for k in row
                if k.startswith(f's{step}/') and k.endswith('/mmd2')}
            assert keys==set(NAMES),'Actual MMD streams disagree with DINO/Qwen-only contract'
            heads={k.removeprefix(f's{step}/gan/').removesuffix('/d_loss') for k in row
                if k.startswith(f's{step}/gan/') and k.endswith('/d_loss') and k.count('/')==3}
            assert heads==set(NAMES),'Actual GAN heads disagree with DINO/Qwen-only contract'
            for name in NAMES:
                assert row[f's{step}/{name}/mmd2']>=-1e-6
                assert row[f's{step}/gan/{name}/d_loss']>=0
                if row['step']>50:assert row[f's{step}/gan/{name}/g_loss']>=0
    return rows


def validate_payload(data):
    assert data['MMD_encoders']==data['GAN_encoders']==NAMES
    assert data['init']=='official pretrained0' and not data['restart_boundaries']
    assert data['GAN_G_first_iteration']==51
    return validate_rows(data['rows'])


def keys():
    result=['step','skipped','grad_norm','seconds','lr','force','loss_total','raw_mmd2','gan/d_lr_used','gan/d_lr_next']
    for step in [1,2]:
        result += [f's{step}/gan/g_active',f's{step}/gan/d_skipped',f's{step}/gan/lambda']
        result += [f's{step}/{name}/mmd2' for name in NAMES]
        result += [f's{step}/gan/{name}/{kind}_loss' for name in NAMES for kind in ['d','g']]
    return result


def trailing_mean(values,width):
    assert width>0
    return [sum(values[max(0,i-width+1):i+1])/min(i+1,width) for i in range(len(values))]
