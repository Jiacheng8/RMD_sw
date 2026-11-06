#!/usr/bin/env python3
"""Repair only stale six-column dimension metadata; never modify generated PNG bytes."""
import datetime
import hashlib
import json
from pathlib import Path

ROOT=Path('/data/rdm-workspace/rdm-sets')
BASE=ROOT/'visualizations/zimage-2step-1024-dino-qwenvl-mmd2-from0/qib-portraits-v1/native-ten-stages'


def main():
    from PIL import Image
    path=BASE/'ten-stages-verified.json'
    original=path.read_bytes();proof=json.loads(original)
    assert proof['passed'] and proof['stages']==list(range(0,451,50))
    assert len(proof['files'])==4 and proof['pixel_identical_tiles']==160
    saved=ROOT/'control/mmd2-ten-stage-original-dimension-record.json'
    assert not saved.exists(),'Inspect existing repair before running again'
    saved.write_bytes(original)
    for item in proof['files']+proof['case_pages']:
        png=Path(item['path']);raw=png.read_bytes()
        assert len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['sha256']
        with Image.open(png) as image:
            expected=(10240,4416) if item in proof['files'] else (5120,2240)
            assert image.format=='PNG' and image.size==expected
        assert item['dimensions'] in [[6144,4416],list(expected)]
        item['dimensions']=list(expected)
    proof['dimension_record_repair']=dict(passed=True,PNG_bytes_unchanged=True,
        original_record=str(saved),source_commit=(Path(__file__).resolve().parents[1]/'SOURCE_COMMIT').read_text().strip(),
        corrected_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    temp=path.with_name(path.name+'.partial');temp.write_text(json.dumps(proof,indent=2)+'\n');temp.replace(path)
    print(json.dumps(proof['dimension_record_repair']))


if __name__=='__main__':
    main()
