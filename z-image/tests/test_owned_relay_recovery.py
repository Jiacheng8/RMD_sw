"""Execute the shipped remote cleanup, entirely offline with injected OS probes."""

import ast
import base64
import copy
import json
from pathlib import Path
import signal
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "ops" / "rdm-owned-relay-mux.py"
TREE = ast.parse(SCRIPT.read_text())
ASSIGNMENTS = {
    node.targets[0].id: node.value
    for node in TREE.body
    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
}
REMOTE = ast.parse(ast.literal_eval(ASSIGNMENTS["REMOTE"]))
BOOT = "42c371d1-e5e3-4181-bc52-68c13c855aca"
CLAIM = {
    "protocol": 1,
    "goal_id": "01a19c31-f2fb-70c0-bf90-dbae201d2479",
    "session_uuid": "9c9458ca-bf7e-4fa2-aec1-d73c7253bbd0",
    "registration_method": "same_controlmaster_exec_ancestor",
    "host_alias": "jump_h20-1",
    "remote_port": 18791,
    "macbook_proxy_port": 7896,
    "local_master_pid": 12345,
    "local_supervisor_pid": 12344,
    "registered_utc": "2026-11-03T16:00:00+00:00",
    "remote_pid": 20001,
    "remote_start_ticks": 90210,
    "remote_boot_id": BOOT,
    "sshd_exe": "/usr/sbin/sshd",
    "sshd_command": "sshd: root@notty",
    "bound_endpoint": "127.0.0.1:18791",
}


@pytest.fixture
def remote():
    # Importing the remote prelude only defines probes; the final dispatcher is
    # executed after every filesystem/process/signal probe has been replaced.
    assert isinstance(REMOTE.body[-1], ast.Try)
    namespace = {}
    exec(compile(ast.Module(body=REMOTE.body[:-1], type_ignores=[]), str(SCRIPT), "exec"), namespace)
    replies, events = [], []
    boot_file = SimpleNamespace(read_text=Mock(return_value=BOOT))

    def path(name):
        assert name == "/proc/sys/kernel/random/boot_id"
        return boot_file

    def record_audit(*args, **kwargs):
        events.append("audit")

    def send_signal(*args):
        events.append("TERM")

    os_mock = SimpleNamespace(pidfd_open=Mock(return_value=73), close=Mock())
    signal_mock = SimpleNamespace(SIGTERM=signal.SIGTERM, pidfd_send_signal=Mock(side_effect=send_signal))
    namespace.update(
        Path=path,
        os=os_mock,
        signal=signal_mock,
        read_claim=Mock(return_value=copy.deepcopy(CLAIM)),
        owner=Mock(return_value=CLAIM["remote_pid"]),
        identity=Mock(return_value=copy.deepcopy(CLAIM)),
        write_json=Mock(side_effect=record_audit),
        reply=lambda status, reason, **data: replies.append(dict(status=status, reason=reason, **data)),
    )

    def run(**overrides):
        payload = {
            "claim": copy.deepcopy(CLAIM),
            "terminal_observed": True,
            "master_exit_code": 255,
            "master_wait_utc": "2026-11-03T17:29:58+00:00",
            **overrides,
        }
        encoded = base64.b64encode(json.dumps(payload).encode()).decode()
        namespace["sys"] = SimpleNamespace(argv=["remote", "cleanup", encoded])
        exec(compile(ast.Module(body=[REMOTE.body[-1]], type_ignores=[]), str(SCRIPT), "exec"), namespace)
        assert len(replies) == 1
        return replies[0]

    return SimpleNamespace(ns=namespace, run=run, events=events, boot_file=boot_file)


def assert_no_signal(remote):
    remote.ns["os"].pidfd_open.assert_not_called()
    remote.ns["signal"].pidfd_send_signal.assert_not_called()
    remote.ns["write_json"].assert_not_called()


@pytest.mark.parametrize("owner", [None, 99999])
def test_missing_claim_releases_only_if_original_pid_no_longer_owns_port(remote, owner):
    remote.ns["read_claim"].side_effect = FileNotFoundError("removed mount")
    remote.ns["owner"].return_value = owner
    assert remote.run() == dict(status="released", reason="claim_no_longer_owns_port")
    remote.ns["identity"].assert_not_called()
    assert_no_signal(remote)


def test_missing_claim_with_matching_live_owner_never_signals(remote):
    remote.ns["read_claim"].side_effect = FileNotFoundError("removed mount")
    assert remote.run() == dict(status="skip", reason="remote_claim_missing_owner_still_present")
    remote.ns["identity"].assert_not_called()
    assert_no_signal(remote)


def test_missing_claim_and_changed_boot_releases_without_pid_probe(remote):
    remote.ns["read_claim"].side_effect = FileNotFoundError("removed mount")
    remote.boot_file.read_text.return_value = "0c815c62-fd7a-419a-9fbc-9db351b880aa"
    assert remote.run() == dict(status="released", reason="old_boot_claim_no_longer_applicable")
    remote.ns["owner"].assert_not_called()
    assert_no_signal(remote)


@pytest.mark.parametrize("foreign", [None, {**CLAIM, "session_uuid": "foreign-session"}])
def test_existing_foreign_or_null_file_still_blocks_even_if_port_is_empty(remote, foreign):
    remote.ns["read_claim"].return_value = foreign
    remote.ns["owner"].return_value = None
    assert remote.run() == dict(status="skip", reason="remote_claim_file_mismatch")
    remote.boot_file.read_text.assert_not_called()
    remote.ns["owner"].assert_not_called()
    assert_no_signal(remote)


@pytest.mark.parametrize("overrides", [
    {"terminal_observed": False},
    {"terminal_observed": 1},
    {"master_exit_code": None},
    {"master_exit_code": True},
    {"claim": {**CLAIM, "goal_id": "foreign"}},
    {"claim": {**CLAIM, "registration_method": "guessed-pid"}},
    {"claim": {**CLAIM, "remote_pid": True}},
    {"claim": {**CLAIM, "remote_pid": 1}},
    {"claim": {**CLAIM, "remote_start_ticks": None}},
    {"claim": {**CLAIM, "remote_boot_id": ""}},
    {"claim": {**CLAIM, "bound_endpoint": "0.0.0.0:18791"}},
    {"claim": {**CLAIM, "sshd_exe": "/bin/other"}},
])
def test_invalid_terminal_witness_or_claim_cannot_release_or_signal(remote, overrides):
    remote.ns["read_claim"].side_effect = FileNotFoundError("removed mount")
    remote.ns["owner"].return_value = None
    assert remote.run(**overrides) == dict(
        status="skip", reason="remote_verification_failed", error_type="ValueError")
    remote.ns["read_claim"].assert_not_called()
    remote.ns["owner"].assert_not_called()
    assert_no_signal(remote)


def test_other_read_error_does_not_count_as_missing_and_redacts_detail(remote):
    remote.ns["read_claim"].side_effect = PermissionError("private sensitive diagnostic")
    assert remote.run() == dict(status="skip", reason="remote_verification_failed", error_type="PermissionError")
    remote.ns["owner"].assert_not_called()
    assert_no_signal(remote)


def test_owned_term_requires_two_matching_claim_and_identity_checks_and_audit(remote):
    result = remote.run()
    assert result["status"] == "term_sent"
    assert result["reason"] == "registered_owned_orphan_only"
    assert result["audit_path"].startswith("/data/rdm-workspace/rdm-sets/control/relay-owned-repair-")
    assert remote.ns["read_claim"].call_count == 2
    assert remote.ns["identity"].call_count == 2
    remote.ns["os"].pidfd_open.assert_called_once_with(CLAIM["remote_pid"])
    remote.ns["signal"].pidfd_send_signal.assert_called_once_with(73, signal.SIGTERM)
    remote.ns["os"].close.assert_called_once_with(73)
    assert remote.events == ["audit", "TERM"]
    assert remote.ns["write_json"].call_args.kwargs == {"exclusive": True}


def test_identity_mismatch_before_pidfd_prevents_signal(remote):
    remote.ns["identity"].return_value = {**CLAIM, "remote_start_ticks": 90211}
    assert remote.run() == dict(status="skip", reason="registered_process_identity_mismatch")
    assert_no_signal(remote)


@pytest.mark.parametrize("change", ["claim", "identity", "claim_removed"])
def test_race_after_pidfd_open_still_cannot_signal(remote, change):
    if change == "claim":
        remote.ns["read_claim"].side_effect = [copy.deepcopy(CLAIM), {**CLAIM, "session_uuid": "foreign"}]
    elif change == "identity":
        remote.ns["identity"].side_effect = [copy.deepcopy(CLAIM), {**CLAIM, "remote_start_ticks": 90211}]
    else:
        remote.ns["read_claim"].side_effect = [copy.deepcopy(CLAIM), FileNotFoundError("removed during cleanup")]
    result = remote.run()
    assert result["status"] == "skip"
    assert result["reason"] == ("remote_verification_failed" if change == "claim_removed" else "claim_changed_before_signal")
    remote.ns["os"].pidfd_open.assert_called_once_with(CLAIM["remote_pid"])
    remote.ns["os"].close.assert_called_once_with(73)
    remote.ns["signal"].pidfd_send_signal.assert_not_called()
    remote.ns["write_json"].assert_not_called()


def test_previous_term_never_escalates(remote):
    assert remote.run(term_sent_utc="2026-11-03T17:30:00+00:00") == dict(
        status="skip", reason="previous_TERM_pending_no_escalation")
    assert_no_signal(remote)


def test_unavailable_pidfd_never_falls_back_to_pid_signal(remote):
    del remote.ns["os"].pidfd_open
    assert remote.run() == dict(status="skip", reason="pidfd_unavailable")
    remote.ns["signal"].pidfd_send_signal.assert_not_called()
    remote.ns["write_json"].assert_not_called()


def test_state_lock_and_remote_control_keep_intended_paths():
    assert ast.literal_eval(ASSIGNMENTS["WORK"].args[0]) == "/Users/shaoshitong/Documents/Codex/2026-11-01/ben-d/work"
    for name, filename in (("STATE", "rdm-relay-state.json"), ("LOCK", "rdm-relay-mux.lock")):
        node = ASSIGNMENTS[name]
        assert isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div)
        assert isinstance(node.left, ast.Name) and node.left.id == "WORK"
        assert ast.literal_eval(node.right) == filename
    remote_assignments = {
        node.targets[0].id: node.value for node in REMOTE.body
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
    }
    assert ast.literal_eval(remote_assignments["CONTROL"].args[0]) == "/data/rdm-workspace/rdm-sets/control"
