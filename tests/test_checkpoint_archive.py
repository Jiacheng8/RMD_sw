import hashlib
import json
from pathlib import Path
from unittest import mock
import pytest
import importlib.util
spec = importlib.util.spec_from_file_location('checkpoint_archive_under_test',
    Path(__file__).resolve().parents[1]/'rdm/sw_lmmd/checkpoint_archive.py')
archive_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(archive_module)
archive_checkpoint, wait_for_space = archive_module.archive_checkpoint, archive_module.wait_for_space


def validator(path, step):
    return dict(step=step, model_dtype='test-bytes')


def test_roundtrip_receipt_before_removing_source(tmp_path):
    source = tmp_path/'step_0000200.pth'; payload = bytes(range(256))*1024
    source.write_bytes(payload)
    proof = tmp_path/'control/proof.json'
    result = archive_checkpoint(source, tmp_path/'archive', proof, validator)
    assert not source.exists()
    assert Path(result['destination']).read_bytes() == payload
    assert json.loads(proof.read_text())['sha256'] == hashlib.sha256(payload).hexdigest()
    assert result['local_source_removed'] is True


def test_failed_readback_preserves_only_source_and_does_not_publish(tmp_path):
    source = tmp_path/'step_0000250.pth'; source.write_bytes(b'only-good-copy')
    with mock.patch.object(archive_module, 'sha256', return_value='corrupt'):
        with pytest.raises(AssertionError, match='readback'):
            archive_checkpoint(source, tmp_path/'archive', tmp_path/'proof.json', validator)
    assert source.read_bytes() == b'only-good-copy'
    assert not (tmp_path/'archive'/source.name).exists()
    assert not (tmp_path/'proof.json').exists()


def test_existing_different_checkpoint_is_never_overwritten(tmp_path):
    source = tmp_path/'step_0000300.pth'; source.write_bytes(b'new-unique')
    archive = tmp_path/'archive'; archive.mkdir(); target = archive/source.name; target.write_bytes(b'previous-unique')
    with pytest.raises(AssertionError, match='differs'):
        archive_checkpoint(source, archive, tmp_path/'proof.json', validator)
    assert source.read_bytes() == b'new-unique' and target.read_bytes() == b'previous-unique'


def test_resume_is_never_an_archive_deletion_candidate(tmp_path):
    source = tmp_path/'resume.pth'; source.write_bytes(b'state')
    with pytest.raises(AssertionError, match='Never'):
        archive_checkpoint(source, tmp_path/'archive', tmp_path/'proof.json', validator)
    assert source.exists()


def test_final_named_corruption_preserves_source(tmp_path):
    source = tmp_path/'step_0000050.pth'; source.write_bytes(b'new-milestone')
    real_sha = archive_module.sha256
    def readback(path):
        return 'corrupt-final-name' if Path(path).name == source.name else real_sha(path)
    with mock.patch.object(archive_module, 'sha256', side_effect=readback):
        with pytest.raises(AssertionError, match='Final-named'):
            archive_checkpoint(source, tmp_path/'archive', tmp_path/'proof.json', validator, verify_final=True)
    assert source.read_bytes() == b'new-milestone'
    assert not (tmp_path/'proof.json').exists()


@pytest.mark.parametrize('first,expected', [(50,list(range(50,501,50))), (200,list(range(200,501,50)))])
def test_watch_milestones_include_first50_and_keep_old_default(tmp_path, first, expected):
    source=tmp_path/'source'; source.mkdir(); control=tmp_path/'control'; control.mkdir()
    for step in expected:(source/f'step_{step:07d}.pth').write_bytes(b'test')
    real_archive=archive_module.archive_checkpoint
    def execute(src, dst, proof, **kwargs):
        return real_archive(src,dst,proof,validator,**kwargs)
    with mock.patch.object(archive_module,'archive_checkpoint',side_effect=execute):
        kwargs={} if first==200 else dict(first_step=first,verify_final=True)
        archive_module.watch(source,tmp_path/'archive',control,**kwargs)
    complete=json.loads((control/'checkpoint-archive-complete.json').read_text())
    assert [r['step'] for r in complete['weights']]==expected
    assert all(r['final_named_sha_readback']==(first==50) for r in complete['weights'])
    assert json.loads((control/'checkpoint-archive-progress.json').read_text())['pending_steps']==[]


def test_low_space_with_dead_archiver_stops_before_write(tmp_path):
    exitfile = tmp_path/'archive.exit'; exitfile.write_text('1\n')
    with mock.patch.object(archive_module.shutil, 'disk_usage', return_value=mock.Mock(free=10)):
        with pytest.raises(RuntimeError, match='Archive worker exited'):
            wait_for_space(tmp_path, 100, exitfile)


def test_failed_remote_flush_preserves_source(tmp_path):
    source = tmp_path/'step_0000350.pth'; source.write_bytes(b'only-unique-model')
    with mock.patch.object(archive_module.os, 'fsync', side_effect=OSError(28, 'No space left on device')):
        with pytest.raises(OSError):
            archive_checkpoint(source, tmp_path/'archive', tmp_path/'proof.json', validator)
    assert source.read_bytes() == b'only-unique-model'
    assert not (tmp_path/'archive'/source.name).exists()
    assert not (tmp_path/'proof.json').exists()


def test_object_metadata_refresh_requires_complete_content_identity():
    import types
    helper_spec = importlib.util.spec_from_file_location('archive_verifier_under_test',
        Path(__file__).resolve().parents[1]/'ops/h20-verify-archived-weights.py')
    helper = importlib.util.module_from_spec(helper_spec); helper_spec.loader.exec_module(helper)
    before = types.SimpleNamespace(st_size=100, st_mtime_ns=10)
    after = types.SimpleNamespace(st_size=100, st_mtime_ns=20)
    helper.verify_content_identity(before, after, 100, 100, 'correct-sha', 'correct-sha')
    with pytest.raises(AssertionError, match='SHA'):
        helper.verify_content_identity(before, after, 100, 100, 'bad-sha', 'correct-sha')
    with pytest.raises(AssertionError, match='size'):
        helper.verify_content_identity(before, after, 99, 100, 'correct-sha', 'correct-sha')
