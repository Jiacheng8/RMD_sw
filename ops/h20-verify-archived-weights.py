#!/usr/bin/env python3
"""Independently read full final-named milestone objects; CPU-only, no model hooks."""
import hashlib, json, os, pathlib, time, traceback

ROOT = pathlib.Path('/data/rdm-workspace/rdm-sets')
CONTROL = ROOT/'control/twohead-retain50-archive'


def verify_content_identity(before, after, read_bytes, expected_bytes, actual_sha, expected_sha):
    # ks3fs may refresh synthetic inode/mtime metadata while reading an unchanged
    # object. The complete expected SHA and byte count remain the identity check.
    assert before.st_size == after.st_size == read_bytes == expected_bytes, 'Archived size changed'
    assert actual_sha == expected_sha, 'Final archived object SHA differs'


def write(name, value):
    p = CONTROL/name; tmp = p.with_name(p.name+'.partial')
    tmp.write_text(json.dumps(value, indent=2)+'\n'); os.replace(tmp, p)


def main():
    deadline = time.monotonic()+36*3600
    results = []
    for step in range(200, 501, 50):
        proof = CONTROL/f'checkpoint-archive-{step}.json'
        while not proof.exists():
            assert time.monotonic() < deadline, 'Final-named SHA verification exceeded36h'
            status = CONTROL/'checkpoint-archive-worker.exit'
            assert not status.exists() or status.read_text().strip() == '0', 'Archiver failed; preserve existing objects'
            time.sleep(5)
        expected = json.loads(proof.read_text()); assert expected['passed']
        path = pathlib.Path(expected['destination'])
        assert str(path).startswith('/mnt/airs-business/shaoshitong/rdm-sets/checkpoints/')
        before = path.stat(); assert before.st_size == expected['bytes']
        h = hashlib.sha256(); read = 0; reported = -1
        with path.open('rb') as f:
            for block in iter(lambda:f.read(8*1024**2), b''):
                h.update(block); read += len(block)
                if read//1024**3 != reported:
                    reported = read//1024**3
                    write('checkpoint-final-readback-progress.json', dict(
                        step=step, read_bytes=read, total_bytes=before.st_size,
                        verified_steps=[r['step'] for r in results], updated_epoch=time.time()))
        assert read == expected['bytes'] and h.hexdigest() == expected['sha256'], 'Final archived object SHA differs'
        after = path.stat()
        verify_content_identity(before, after, read, expected['bytes'], h.hexdigest(), expected['sha256'])
        receipt = dict(passed=True, step=step, path=str(path), bytes=read, sha256=h.hexdigest(),
            verified_epoch=time.time(), scope='full final-named remote object SHA readback',
            mtime_before_ns=before.st_mtime_ns, mtime_after_ns=after.st_mtime_ns)
        write(f'checkpoint-final-readback-{step}.json', receipt); results.append(receipt)
        print(json.dumps(receipt), flush=True)
    write('checkpoint-final-readback-complete.json', dict(passed=True, weights=results))


if __name__ == '__main__':
    code = 0
    try:
        main()
    except BaseException:
        traceback.print_exc(); code = 1
    finally:
        (CONTROL/'checkpoint-final-readback.exit').write_text(str(code)+'\n')
    raise SystemExit(code)
