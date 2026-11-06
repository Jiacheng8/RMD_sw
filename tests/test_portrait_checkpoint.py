import importlib.util
import json
from pathlib import Path
import pytest
import hashlib

spec = importlib.util.spec_from_file_location('portrait_checkpoint_under_test',
    Path(__file__).resolve().parents[1]/'ops/h20_portrait_checkpoint.py')
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


def proof(control, **changes):
    data = dict(passed=True, step=250, path=str(module.ARCHIVE/'step_0000250.pth'),
                bytes=24619908172, sha256='a'*64)
    data.update(changes)
    (control/'checkpoint-final-readback-250.json').write_text(json.dumps(data))


def test_waits_for_full_final_object_receipt(tmp_path):
    assert module.archived_checkpoint(250, tmp_path) is None


def test_rejects_receipt_for_another_step(tmp_path):
    proof(tmp_path, step=300)
    with pytest.raises(AssertionError, match='step'):
        module.archived_checkpoint(250, tmp_path)


def test_rejects_wrong_weight_even_with_same_size(tmp_path):
    proof(tmp_path, path=str(module.ARCHIVE/'step_0000200.pth'))
    with pytest.raises(AssertionError, match='Wrong'):
        module.archived_checkpoint(250, tmp_path)


def test_rejects_unverified_or_partial_receipt(tmp_path):
    proof(tmp_path, passed=False)
    with pytest.raises(AssertionError):module.archived_checkpoint(250, tmp_path)
    proof(tmp_path, sha256='not-a-complete-sha')
    with pytest.raises(AssertionError, match='complete-model'):
        module.archived_checkpoint(250, tmp_path)


def test_rejects_wrong_loaded_payload_step():
    with pytest.raises(AssertionError, match='different step'):
        module.validate_state({'step':300, 'schedule':{'step':300}}, 250, archived=True)


def test_ram_stream_matches_bytes_and_sha_without_disk_staging(tmp_path):
    payload=b'small-synthetic-test-data'
    source=tmp_path/'source.txt';source.write_bytes(payload)
    receipt=dict(bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest())
    result=module.read_verified_bytes(source,receipt)
    try:assert result.read()==payload
    finally:result.close()
    assert list(tmp_path.iterdir())==[source]
    with pytest.raises(AssertionError,match='differs'):
        module.read_verified_bytes(source,dict(receipt,sha256='0'*64))
