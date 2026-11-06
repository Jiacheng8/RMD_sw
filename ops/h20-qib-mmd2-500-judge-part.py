#!/usr/bin/env python3
"""Run the pinned original judge with a bounded allocator on one authorized GPU."""
import argparse
import os
from pathlib import Path
import runpy
import sys

ROOT = Path('/data/rdm-workspace/rdm-sets')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--batch', type=int, required=True)
    parser.add_argument('--model', type=Path, required=True)
    args = parser.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES'] in ('4', '5', '6', '7')
    assert 1 <= args.batch <= 128
    assert args.model.resolve().parent == Path('/dev/shm')
    receipt = __import__('json').loads((ROOT/'control/qib-turbo-baselines-judge-ram-verified.json').read_text())
    assert receipt['passed'] and receipt['model'] == str(args.model) and receipt['final_named_readback']
    import torch
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.95)
    bench = ROOT/'qib/bench'
    os.chdir(bench)
    sys.path.insert(0, str(bench))
    sys.argv = [str(bench/'judge.py'), '--input', str(args.input),
        '--model', str(args.model), '--local-metadata', str(bench/'metadata/bench_metadata.json'),
        '--max-batch-size', str(args.batch)]
    runpy.run_path(bench/'judge.py', run_name='__main__')


if __name__ == '__main__':
    main()
