#!/usr/bin/env python3
"""Task-scoped streaming relay with ownership bound to one ControlMaster.

Importing this file does nothing. It neither starts experiments nor changes SSH,
Clash, or system configuration. Unregistered listeners are never signalled.
"""

import base64
import datetime as dt
import fcntl
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import shlex
import signal
import subprocess
import threading
import time
import uuid


WORK = Path("/Users/shaoshitong/Documents/Codex/2026-11-01/ben-d/work")
STATE = WORK / "rdm-relay-state.json"
LOCK = WORK / "rdm-relay-mux.lock"
LOG = WORK / "rdm-macbook-relay-mux.log"
HOST = "jump_h20-1"
FORWARD = "127.0.0.1:18791:127.0.0.1:7896"
GOAL = "01a19c31-f2fb-70c0-bf90-dbae201d2479"
MODE = "controlmaster-owned-v1"
CALL_TIMEOUT = 40


REMOTE = r'''
import base64
import datetime as dt
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import uuid

GOAL = "01a19c31-f2fb-70c0-bf90-dbae201d2479"
CONTROL = Path("/share-evpfs/shaoshitong/rdm-sets/control")
OWNED = CONTROL / "relay-owned.json"
ENDPOINT = "127.0.0.1:18791"

def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()

def reply(status, reason, **data):
    print(json.dumps({"status": status, "reason": reason, **data}), flush=True)

def stat(pid):
    value = Path(f"/proc/{pid}/stat").read_text()
    fields = value[value.rfind(")") + 2:].split()
    return int(fields[1]), int(fields[19])

def ancestors():
    pid, result = os.getpid(), set()
    while pid > 0 and pid not in result:
        result.add(pid)
        pid, unused = stat(pid)
    return result

def listeners(port_only=False):
    args = ["ss", "-H", "-ltnp"]
    if port_only:
        args += ["( sport = :18791 )"]
    data = subprocess.run(args, check=True, capture_output=True,
                          text=True, timeout=3).stdout
    result = []
    for line in data.splitlines():
        fields = line.split()
        if len(fields) < 5:
            raise ValueError("unparseable_listener")
        result.append((fields[3], set(map(int, re.findall(r"pid=(\d+)", line)))))
    return result

def owner():
    rows = listeners(True)
    if not rows:
        return None
    if len(rows) != 1 or rows[0][0] != ENDPOINT or len(rows[0][1]) != 1:
        raise ValueError("listener_not_unique_loopback_owner")
    return next(iter(rows[0][1]))

def identity(pid, registration=False):
    if pid <= 1:
        raise ValueError("invalid_pid")
    chain = ancestors()
    if registration != (pid in chain):
        raise ValueError("registration_or_management_ancestry_mismatch")
    proc = Path(f"/proc/{pid}")
    unused, ticks = stat(pid)
    if os.readlink(proc / "exe") != "/usr/sbin/sshd":
        raise ValueError("not_sshd_executable")
    command = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode().strip()
    if command not in {"sshd: root", "sshd: root@notty"}:
        raise ValueError("not_root_forwarding_sshd")
    uid = next(line for line in (proc / "status").read_text().splitlines()
               if line.startswith("Uid:"))
    if any(int(value) != 0 for value in uid.split()[1:]):
        raise ValueError("not_root_uid")
    children = set()
    for p in (proc / "task").glob("*/children"):
        children.update(map(int, p.read_text().split()))
    if registration:
        if not children.issubset(chain - {pid}):
            raise ValueError("other_children_during_registration")
    elif children:
        raise ValueError("owner_has_children")
    owned = [address for address, owners in listeners() if pid in owners]
    if owned != [ENDPOINT] or owner() != pid or stat(pid)[1] != ticks:
        raise ValueError("listener_or_identity_changed")
    return {"remote_pid": pid, "remote_start_ticks": ticks,
            "remote_boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "sshd_exe": "/usr/sbin/sshd", "sshd_command": command,
            "bound_endpoint": ENDPOINT}

def read_claim():
    return json.loads(OWNED.read_text())

def write_json(path, data, exclusive=False):
    flags = os.O_WRONLY | os.O_CREAT | (os.O_EXCL if exclusive else os.O_TRUNC)
    fd = os.open(path, flags, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(data, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())

def matches_identity(current, expected):
    return all(current.get(k) == expected.get(k) for k in
               ["remote_pid", "remote_start_ticks", "remote_boot_id",
                "sshd_exe", "bound_endpoint"])

def register():
    session = str(uuid.UUID(sys.argv[2]))
    local_master, local_supervisor = int(sys.argv[3]), int(sys.argv[4])
    pid = owner()
    if pid is None:
        raise ValueError("forward_missing")
    current = identity(pid, registration=True)
    previous = read_claim() if OWNED.exists() else None
    if previous is not None and previous.get("goal_id") != GOAL:
        raise ValueError("foreign_goal_claim_exists")
    # A prior response may have been lost after the durable write. Re-acknowledge
    # only this exact session through the same proven ancestor connection.
    if previous is not None and previous.get("session_uuid") == session:
        if (previous.get("local_master_pid") != local_master or
                previous.get("local_supervisor_pid") != local_supervisor or
                previous.get("registration_method") != "same_controlmaster_exec_ancestor" or
                not matches_identity(identity(pid, registration=True), previous)):
            raise ValueError("same_session_claim_mismatch")
        reply("registered", "same_session_durable_claim_reacknowledged", claim=previous)
        return
    claim = {"protocol": 1, "goal_id": GOAL, "session_uuid": session,
             "registration_method": "same_controlmaster_exec_ancestor",
             "host_alias": "jump_h20-1", "remote_port": 18791,
             "macbook_proxy_port": 7896, "local_master_pid": local_master,
             "local_supervisor_pid": local_supervisor, "registered_utc": utc(),
             **current}
    # Check again immediately before publishing the durable ownership record.
    if not matches_identity(identity(pid, registration=True), claim):
        raise ValueError("owner_changed_before_registration")
    temp = CONTROL / (".relay-owned-" + session + ".tmp")
    write_json(temp, claim, exclusive=True)
    os.replace(temp, OWNED)
    reply("registered", "owner_is_this_exec_channel_ancestor", claim=claim)

def cleanup():
    payload = json.loads(base64.b64decode(sys.argv[2], validate=True))
    claim = payload["claim"]
    if (payload.get("terminal_observed") is not True or
            type(payload.get("master_exit_code")) is not int or
            claim.get("goal_id") != GOAL or
            claim.get("registration_method") != "same_controlmaster_exec_ancestor"):
        raise ValueError("invalid_terminal_witness_or_claim")
    if read_claim() != claim:
        reply("skip", "remote_claim_file_mismatch")
        return
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    if boot != claim["remote_boot_id"]:
        reply("released", "old_boot_claim_no_longer_applicable")
        return
    pid = owner()
    if pid is None or pid != claim["remote_pid"]:
        reply("released", "claim_no_longer_owns_port")
        return
    current = identity(pid)
    if not matches_identity(current, claim):
        reply("skip", "registered_process_identity_mismatch")
        return
    if payload.get("term_sent_utc"):
        reply("skip", "previous_TERM_pending_no_escalation")
        return
    if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
        reply("skip", "pidfd_unavailable")
        return
    descriptor = os.pidfd_open(pid)
    try:
        if read_claim() != claim or not matches_identity(identity(pid), claim):
            reply("skip", "claim_changed_before_signal")
            return
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        audit = CONTROL / ("relay-owned-repair-" + stamp + ".json")
        write_json(audit, {"utc": utc(), "event": "pre_TERM_owned_orphan_verified",
                          "claim": claim, "terminal_witness": payload,
                          "requested_signal": "SIGTERM"}, exclusive=True)
        signal.pidfd_send_signal(descriptor, signal.SIGTERM)
        reply("term_sent", "registered_owned_orphan_only", audit_path=str(audit))
    finally:
        os.close(descriptor)

try:
    if sys.argv[1] == "register":
        register()
    elif sys.argv[1] == "cleanup":
        cleanup()
    else:
        raise ValueError("unknown_operation")
except Exception as exc:
    # Never include environment, process arguments, full configuration or stderr.
    reply("skip", "remote_verification_failed", error_type=type(exc).__name__)
'''


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


class Relay:
    def __init__(self):
        self.stop = threading.Event()
        self.master = None
        self.socket = None
        self.attempt = 0
        self.session = None
        self.claim = None
        self.pending = []
        self.last_exit = None
        self.logger = logging.getLogger("rdm-owned-mux-relay")
        self.logger.setLevel(logging.INFO)
        handler = RotatingFileHandler(LOG, maxBytes=256 * 1024, backupCount=2)
        handler.setFormatter(logging.Formatter("%(message)s"))
        self.logger.addHandler(handler)
        self.load_pending()

    def log(self, event, **data):
        message = json.dumps({"utc": utc(), "event": event, **data}, sort_keys=True)
        self.logger.info(message)
        print(message, flush=True)

    def load_pending(self):
        if not STATE.exists():
            return
        previous = json.loads(STATE.read_text())
        if previous.get("relay_mode") != MODE:
            return  # The legacy supervisor has no ownership claim to adopt.
        for item in previous.get("pending_claims", []):
            if (item.get("terminal_observed") is not True or
                    type(item.get("master_exit_code")) is not int):
                raise RuntimeError("Unconfirmed prior master; manual review required")
            self.pending.append(item)
        if previous.get("owned_claim") is not None:
            raise RuntimeError("Prior active claim has no observed wait terminal state; do not guess")

    def save(self, status):
        data = {"supervisor_pid": os.getpid(),
                "ssh_pid": self.master.pid if self.master is not None else None,
                "attempt": self.attempt, "remote_port": 18791, "macbook_proxy_port": 7896,
                "updated_utc": utc(), "relay_mode": MODE, "status": status,
                "goal_id": GOAL, "session_uuid": self.session,
                "control_socket": str(self.socket) if self.socket else None,
                "owned_claim": self.claim, "pending_claims": self.pending,
                "last_master_exit": self.last_exit}
        temp = STATE.with_name(STATE.name + f".{os.getpid()}.tmp")
        fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as stream:
            json.dump(data, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, STATE)

    def request_stop(self, signum, unused):
        self.stop.set()
        if self.master is not None and self.master.poll() is None:
            try:
                self.master.terminate()  # Only the Popen master created by this instance.
            except ProcessLookupError:
                pass

    def client(self, args, program=None, timeout=CALL_TIMEOUT):
        try:
            result = subprocess.run(args, input=program, text=True,
                                    stdin=None if program is not None else subprocess.DEVNULL,
                                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                    timeout=timeout)
            return result.returncode, result.stdout
        except subprocess.TimeoutExpired:
            return None, ""
        except OSError:
            return -1, ""

    def mux_args(self):
        # Ignore helper-side Host forwards. Only the already-connected socket may
        # carry this request; an unavailable socket must not fall back to SSH.
        return ["ssh", "-F", "/dev/null", "-T", "-S", str(self.socket),
                "-o", "BatchMode=yes", "-o", "ControlMaster=no",
                "-o", "ProxyCommand=/usr/bin/false", "-o", "ProxyJump=none",
                "-o", "ClearAllForwardings=no", "-o", "ConnectTimeout=20"]

    def check_master(self):
        if self.master is None or self.master.poll() is not None:
            return False
        rc, unused = self.client(self.mux_args() + ["-O", "check", HOST], timeout=3)
        return rc == 0 and self.master.poll() is None

    def remote(self, mode, payload):
        if mode == "register":
            if not self.check_master():
                return {"status": "skip", "reason": "master_not_ready_for_registration"}
            args = self.mux_args()
            command = ["python3", "-", "register", self.session,
                       str(self.master.pid), str(os.getpid())]
        else:
            # Fresh management connection, never the old ControlMaster.
            args = ["ssh", "-T", "-S", "none", "-o", "BatchMode=yes",
                    "-o", "ControlMaster=no", "-o", "ControlPath=none",
                    "-o", "ClearAllForwardings=yes", "-o", "RemoteCommand=none",
                    "-o", "ConnectTimeout=20", "-o", "ConnectionAttempts=1",
                    "-o", "ServerAliveInterval=10", "-o", "ServerAliveCountMax=2"]
            encoded = base64.b64encode(json.dumps(payload).encode()).decode()
            command = ["python3", "-", "cleanup", encoded]
        rc, output = self.client(args + [HOST, shlex.join(command)], REMOTE)
        if rc != 0:
            return {"status": "skip", "reason": "management_timeout_or_failure", "exit": rc}
        try:
            value = json.loads(output)
            if not isinstance(value, dict):
                raise ValueError()
            return value
        except ValueError:
            return {"status": "skip", "reason": "invalid_management_response"}

    def cleanup_pending(self):
        remaining = []
        for item in self.pending:
            result = self.remote("cleanup", item)
            self.log("owned_cleanup_result", session_uuid=item["claim"].get("session_uuid"),
                     result=result)
            if result.get("status") == "released":
                continue
            if result.get("status") == "term_sent":
                item = {**item, "term_sent_utc": utc()}
            remaining.append(item)
        self.pending = remaining
        self.save("pending_cleanup" if remaining else "disconnected")
        return not remaining

    def finish_master(self):
        if self.master is None:
            return
        if self.master.poll() is None:
            self.master.terminate()
        # SIGTERM may only target our own local master. No remote cleanup before
        # this exact Popen's wait has observed its real terminal status.
        try:
            code = self.master.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.master.kill()  # Our local child only, never a remote sshd.
            code = self.master.wait(timeout=5)
        self.last_exit = {"master_pid": self.master.pid, "exit_code": code, "wait_utc": utc()}
        if self.claim is not None:
            self.pending.append({"claim": self.claim, "terminal_observed": True,
                                 "master_exit_code": code, "master_wait_utc": utc()})
        self.claim = None
        self.log("master_terminal_observed", **self.last_exit)
        self.save("pending_cleanup" if self.pending else "disconnected")
        self.master = None

    def attempt_once(self):
        self.attempt += 1
        self.session = str(uuid.uuid4())
        self.socket = WORK / f"mx-{os.getpid()}-{self.attempt}.sock"
        if len(os.fsencode(self.socket)) >= 100 or self.socket.exists():
            raise RuntimeError("Control socket is too long or already exists; refusing to adopt")
        command = ["ssh", "-N", "-T", "-M", "-S", str(self.socket),
                   "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
                   "-o", "ConnectionAttempts=1", "-o", "ControlPersist=no",
                   "-o", "ForkAfterAuthentication=no", "-o", "RemoteCommand=none",
                   "-o", "ClearAllForwardings=yes", "-o", "ExitOnForwardFailure=yes",
                   "-o", "ForwardAgent=no", "-o", "ForwardX11=no",
                   "-o", "PermitLocalCommand=no", "-o", "ServerAliveInterval=20",
                   "-o", "ServerAliveCountMax=3", HOST]
        self.master = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.save("connecting")
        self.log("master_started", attempt=self.attempt, master_pid=self.master.pid,
                 session_uuid=self.session)
        forward_requested = False
        try:
            deadline = time.monotonic() + CALL_TIMEOUT
            while not self.stop.is_set() and self.master.poll() is None:
                if self.check_master():
                    break
                if time.monotonic() >= deadline:
                    return
                self.stop.wait(0.25)
            if self.stop.is_set() or not self.check_master():
                return
            # The server may create the forward even if its reply is lost.
            forward_requested = True
            rc, unused = self.client(self.mux_args() + ["-O", "forward", "-R", FORWARD, HOST])
            if rc != 0 or self.master.poll() is not None:
                self.log("forward_request_failed_no_owner_guess", exit=rc)
                return
            result = self.remote("register", None)
            if result.get("status") != "registered" and self.check_master():
                self.log("retrying_same_mux_registration_once", session_uuid=self.session)
                result = self.remote("register", None)
            claim = result.get("claim")
            valid = (result.get("status") == "registered" and isinstance(claim, dict)
                     and claim.get("goal_id") == GOAL and claim.get("session_uuid") == self.session
                     and claim.get("local_master_pid") == self.master.pid
                     and claim.get("local_supervisor_pid") == os.getpid()
                     and claim.get("registration_method") == "same_controlmaster_exec_ancestor")
            if not valid:
                self.log("registration_failed_no_unregistered_cleanup", status=result.get("status"),
                         reason=result.get("reason"))
                return
            # A valid response is the ownership proof even if the master exited
            # immediately afterwards; finally will wait, persist, then clean it.
            self.claim = claim
            self.save("owned" if self.master.poll() is None else "ownership_registered_master_exiting")
            self.log("ownership_registered", session_uuid=self.session,
                     remote_pid=claim.get("remote_pid"), start_ticks=claim.get("remote_start_ticks"))
            if not self.check_master():
                return
            while not self.stop.is_set() and self.master.poll() is None:
                self.stop.wait(1)
        finally:
            if forward_requested and self.claim is None and self.check_master():
                self.client(self.mux_args() + ["-O", "cancel", "-R", FORWARD, HOST])
            self.finish_master()

    def run(self):
        self.log("supervisor_started", supervisor_pid=os.getpid(), goal_id=GOAL)
        try:
            while not self.stop.is_set():
                if self.pending and not self.cleanup_pending():
                    self.stop.wait(10)
                    continue
                if self.stop.is_set():
                    break
                try:
                    self.attempt_once()
                except Exception as exc:
                    self.log("attempt_failed_closed", error_type=type(exc).__name__)
                    self.finish_master()
                if self.pending:
                    self.cleanup_pending()
                if not self.stop.is_set():
                    self.stop.wait(10)
        finally:
            self.finish_master()
            if self.pending:
                self.cleanup_pending()  # One bounded pass; retain failures for recovery.
            self.save("stopped_pending_cleanup" if self.pending else "stopped")
            self.log("supervisor_stopped", pending_claims=len(self.pending))


def main():
    with LOCK.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Another task-owned ControlMaster supervisor is active.")
        relay = Relay()
        signal.signal(signal.SIGTERM, relay.request_stop)
        signal.signal(signal.SIGINT, relay.request_stop)
        relay.run()


if __name__ == "__main__":
    main()
