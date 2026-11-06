#!/usr/bin/env python3
"""Measure corrected two-head configurations on owned GPUs4/5; never train formally."""
import json
from pathlib import Path
import subprocess
import sys

from h20_benchmark_acceptance import validate_benchmark_record

ROOT=Path('/data/rdm-workspace/rdm-sets')
REPO=Path(__file__).resolve().parents[1]
CONFIG=REPO/'z-image/configs/zimage_2step_1024_twohead_glr10_h20_fa3.yaml'
HEADS=['dinov3_l','qwen3vl_4b']
source=(REPO/'SOURCE_COMMIT').read_text().strip()
candidates=[]
cases=[]
for micro in [2,4]:
    prefix=f'twohead-lr10-micro{micro}'
    command=[sys.executable,str(REPO/'ops/h20-checkpoint-sweep.py'),
             '--only-case','2',str(micro),'--config',str(CONFIG),'--state-prefix',prefix]
    subprocess.run(command,cwd=REPO,check=True)
    summary=json.loads((ROOT/f'control/{prefix}-latest.json').read_text())
    assert summary['source_commit']==source
    case=summary['cases'][-1]
    cases.append(case)
    if case['status']=='oom':break
    assert case['status']=='passed',case
    record=json.loads(Path(case['result']).read_text())
    validate_benchmark_record(record,expected_spatial=True,expected_iterations=3,
                              expected_heads=HEADS,expected_probe_request=64,expected_retained_rows=96)
    assert record['source_commit']==source and record['gan_heads']==HEADS
    assert record['G_base_lr']==2.83e-5 and record['D_base_lr']==1e-5
    assert record['adaptive_D_lr'] is True and record['world_size']==2
    assert all(row['logs']['pixel_gc/generator_replay_max_abs']==0.
               for ranks in record['steps'] for row in ranks)
    assert record['spatial_feature_contract']['representations']['input']=='1024px-PNG-uint8-device-fp32-div255'
    rows=subprocess.check_output(['nvidia-smi','-i','4,5',
        '--query-gpu=memory.total','--format=csv,noheader,nounits'],text=True)
    limit=min(map(int,rows.splitlines()))
    if max(case['nvml_peak_mib'].values())<=limit*.95:candidates.append(case)
    # Stop at the measured capacity boundary; do not relax precision/offload or checkpoint grouping.
    if max(case['nvml_peak_mib'].values())>limit*.95:break
assert candidates,'No finite measured configuration with memory headroom'
winner=min(candidates,key=lambda c:c['mean_seconds'])
receipt={'source_commit':source,'config':str(CONFIG),'CPU_and_GPU_validation_complete':True,
         'formal_training_started':False,'physical_gpus':[4,5],'global_batch':32,
         'gan_heads':HEADS,'G_base_lr':2.83e-5,'D_base_lr':1e-5,'adaptive_D_lr':True,
         'cases':cases,'winner':winner,'first_iteration_excluded_from_speed':True}
(ROOT/'control/twohead-lr10-selected.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'validation_complete':True,'winner_micro':winner['micro_batch'],
                  'grad_accum':winner['grad_accum'],'steady_seconds':winner['mean_seconds'],
                  'peak_MiB':winner['nvml_peak_mib']}),flush=True)
