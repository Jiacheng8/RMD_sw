"""No GPU/weights: exercise the real evaluation-to-training dependency gates."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import pytest

spec=importlib.util.spec_from_file_location('mmd2_handoff',Path(__file__).resolve().parents[1]/'ops/h20-dino-qwenvl-mmd2-workflow.py')
workflow=importlib.util.module_from_spec(spec);spec.loader.exec_module(workflow)


def fixtures(control):
    payloads={
        'qib350-two-gpu-complete.json':dict(passed=True,images=1000,judge_parts=8,actual_raw_dimension_answers=4105),
        'dino-qwenvl-mmd2-reference-verified.json':dict(passed=True,rows=125808,MMD_encoders=['dinov3_l','qwen3vl_4b'],
            raw_image_features_finite=True,text_norm_verified=True,image_features_L2_normalized=False),
        'dino-qwenvl-mmd2-judge-cache-offload-verified.json':dict(passed=True,local_path_is_verified_symlink=True)}
    for name,payload in payloads.items():(control/name).write_text(json.dumps(payload))
    for name in ['qib350-two-gpu.exit','dino-qwenvl-mmd2-reference.exit']:(control/name).write_text('0\n')


def test_complete_data_does_not_replace_actual_benchmark_exit(tmp_path):
    fixtures(tmp_path);(tmp_path/'qib350-two-gpu.exit').unlink()
    with mock.patch.object(workflow,'CONTROL',tmp_path):
        passed,detail=workflow.ready()
    assert not passed and 'actual QIB' in detail['waiting']


def test_true_dependency_failure_is_not_endless_wait(tmp_path):
    (tmp_path/'qib350-two-gpu.exit').write_text('1\n')
    with mock.patch.object(workflow,'CONTROL',tmp_path),pytest.raises(RuntimeError,match='real exit1'):
        workflow.ready()


def test_reference_must_not_silently_retain_aim(tmp_path):
    fixtures(tmp_path);p=tmp_path/'dino-qwenvl-mmd2-reference-verified.json';value=json.loads(p.read_text())
    value['MMD_encoders'].append('aimv2_huge');p.write_text(json.dumps(value))
    with mock.patch.object(workflow,'CONTROL',tmp_path),pytest.raises(AssertionError):workflow.ready()


def test_capacity_covers_new_resume_plus_student(tmp_path):
    fixtures(tmp_path)
    with mock.patch.object(workflow,'CONTROL',tmp_path),mock.patch.object(workflow.shutil,'disk_usage',return_value=SimpleNamespace(free=54_000_000_000)):
        passed,detail=workflow.ready()
    assert not passed and '62GB' in detail['waiting']
    with mock.patch.object(workflow,'CONTROL',tmp_path),mock.patch.object(workflow.shutil,'disk_usage',return_value=SimpleNamespace(free=70_000_000_000)):
        passed,detail=workflow.ready()
    assert passed and detail['benchmark_images']==1000


def test_missing_4105_answers_cannot_start_training(tmp_path):
    fixtures(tmp_path);p=tmp_path/'qib350-two-gpu-complete.json';value=json.loads(p.read_text());value['actual_raw_dimension_answers']=4104;p.write_text(json.dumps(value))
    with mock.patch.object(workflow,'CONTROL',tmp_path),pytest.raises(AssertionError):workflow.ready()


def test_final_audit_cannot_use_live_or_failed_training(tmp_path):
    spec=importlib.util.spec_from_file_location('mmd2_final',Path(__file__).resolve().parents[1]/'ops/h20-audit-mmd2-final.py')
    audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
    (tmp_path/(audit.PREFIX+'.exit')).write_text('1\n')
    with mock.patch.object(audit,'CONTROL',tmp_path),pytest.raises(AssertionError,match='actual successful'):
        audit.audit(tmp_path,'not-a-real-source')


def test_benchmark_preservation_cannot_use_prewrite_completion(tmp_path):
    spec=importlib.util.spec_from_file_location('qib_preserve',Path(__file__).resolve().parents[1]/'ops/h20-preserve-qib350.py')
    preserve=importlib.util.module_from_spec(spec);spec.loader.exec_module(preserve)
    (tmp_path/'qib350-two-gpu.exit').write_text('1\n')
    with mock.patch.object(preserve,'CONTROL',tmp_path),pytest.raises(AssertionError):preserve.preserve()


def test_full_run_compression_hash_covers_actual_tar_bytes(tmp_path):
    import hashlib
    import io
    import tarfile
    spec=importlib.util.spec_from_file_location('qib_tar_stream',Path(__file__).resolve().parents[1]/'ops/h20-preserve-qib350.py')
    preserve=importlib.util.module_from_spec(spec);spec.loader.exec_module(preserve)
    out=io.BytesIO()
    with mock.patch.object(preserve,'CONTROL',tmp_path):
        hashed=preserve.HashWriter(out)
        with preserve.archive_stream(hashed) as tar:
            data=b'code-only audit fixture';info=tarfile.TarInfo('fixture.txt');info.size=len(data);tar.addfile(info,io.BytesIO(data))
    payload=out.getvalue()
    assert hashed.bytes==len(payload) and hashed.h.hexdigest()==hashlib.sha256(payload).hexdigest()
    with tarfile.open(fileobj=io.BytesIO(payload),mode='r:gz') as tar:
        assert tar.extractfile('fixture.txt').read()==data
