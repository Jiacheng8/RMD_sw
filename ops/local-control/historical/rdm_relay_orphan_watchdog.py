#!/usr/bin/env python3
"""Conservative sidecar for this task's existing streaming SSH reverse forward.

Nothing runs when this module is imported. The existing relay supervisor is not
changed or restarted. Only its absent SSH child can trigger the two-phase check.
"""

import datetime as dt
import errno
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


HERE = Path(__file__).resolve().parent
OLD_WORK = Path("/Users/shaoshitong/Documents/Codex/2026-11-01/ben-d/work")
RELAY_STATE = OLD_WORK / "rdm-relay-state.json"
SUPERVISOR_SCRIPT = OLD_WORK / "rdm_macbook_relay.py"
OWN_STATE = HERE / "rdm-relay-watchdog-state.json"
OWN_LOG = HERE / "rdm-relay-watchdog.log"
OWN_LOCK = HERE / "rdm-relay-watchdog.lock"
HOST = "jump_h20-1"
POLL_SECONDS = 12
SSH_TIMEOUT = 40
REPAIR_COOLDOWN = 120


REMOTE_PROGRAM = r'''
import datetime as dt
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

PORT = 18791
ENDPOINT = "127.0.0.1:18791"
CONTROL = Path("/share-evpfs/shaoshitong/rdm-sets/control")
URL = "https://huggingface.co/api/models/Qwen/Qwen-Image-Bench"

def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()

def reply(status, reason, **data):
    print(json.dumps({"status": status, "reason": reason, **data}), flush=True)

def stat_fields(pid):
    text = Path(f"/proc/{pid}/stat").read_text()
    fields = text[text.rfind(")") + 2:].split()
    return int(fields[1]), int(fields[19])  # PPID (4), starttime (22)

def ancestors():
    seen, pid = set(), os.getpid()
    while pid > 0 and pid not in seen:
        seen.add(pid)
        pid, unused = stat_fields(pid)
    return seen

def listeners(port_only=False):
    command = ["ss", "-H", "-ltnp"]
    if port_only:
        command.append("( sport = :18791 )")
    result = subprocess.run(command, capture_output=True, text=True,
                            timeout=3, check=True)
    rows = []
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) < 5:
            raise ValueError("unparseable_listener")
        rows.append((fields[3], set(map(int, re.findall(r"pid=(\d+)", line)))))
    return rows

def snapshot(expected=None):
    rows = listeners(port_only=True)
    if len(rows) != 1 or rows[0][0] != ENDPOINT or len(rows[0][1]) != 1:
        return None
    pid = next(iter(rows[0][1]))
    if pid <= 1 or pid in ancestors():
        return None
    proc = Path(f"/proc/{pid}")
    ppid, started = stat_fields(pid)
    if expected and (pid != expected[0] or started != expected[1]):
        return None
    if os.readlink(proc / "exe") != "/usr/sbin/sshd":
        return None
    command = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode().strip()
    if command not in {"sshd: root", "sshd: root@notty"}:
        return None
    uid_line = next(line for line in (proc / "status").read_text().splitlines()
                    if line.startswith("Uid:"))
    if any(int(value) != 0 for value in uid_line.split()[1:]):
        return None
    for child_file in (proc / "task").glob("*/children"):
        if child_file.read_text().strip():
            return None
    owned = [address for address, owners in listeners() if pid in owners]
    if owned != [ENDPOINT]:
        return None
    # Re-read starttime after the structural checks; no PID-only decisions.
    if stat_fields(pid)[1] != started:
        return None
    return {"pid": pid, "start_ticks": started, "ppid": ppid,
            "command": command, "listeners": owned, "checked_utc": utc()}

def failed_probes():
    records = []
    for unused in range(2):
        result = subprocess.run(
            ["curl", "--silent", "--show-error", "--proxy", "http://" + ENDPOINT,
             "--connect-timeout", "4", "--max-time", "4", "--output", "/dev/null",
             "--write-out", "%{http_code}", URL],
            capture_output=True, text=True, timeout=5)
        code = result.stdout.strip()
        records.append({"utc": utc(), "http_code": code, "curl_exit": result.returncode})
        if result.returncode == 0 or code != "000":
            return None
    return records

def audit_before_term(candidate, probes):
    if not CONTROL.is_dir():
        raise ValueError("control_directory_missing")
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    path = CONTROL / f"relay-orphan-repair-{stamp}.json"
    record = {"protocol": 1, "utc": utc(), "event": "pre_term_verified",
              "requested_signal": "SIGTERM", "candidate": candidate,
              "proxy_probes": probes,
              "scope": "Only the identity-checked orphan owning loopback TCP 18791."}
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return str(path)

def main():
    phase = sys.argv[1]
    expected = None
    if phase == "phase2":
        expected = int(sys.argv[2]), int(sys.argv[3])
    elif phase != "phase1":
        raise ValueError("unknown_phase")
    first = snapshot(expected)
    if first is None:
        reply("skip", "listener_not_an_eligible_orphan")
        return
    identity = first["pid"], first["start_ticks"]
    probes = failed_probes()
    if probes is None:
        reply("skip", "proxy_probe_responded")
        return
    current = snapshot(identity)
    if current is None:
        reply("skip", "candidate_changed_during_probes")
        return
    if phase == "phase1":
        reply("candidate", "two_failed_probes_and_structural_checks", **current)
        return
    # pidfd makes the actual signal independent of numeric PID reuse. Do not
    # fall back to a PID-only signal on an older kernel or Python runtime.
    if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
        reply("skip", "pidfd_unavailable")
        return
    descriptor = os.pidfd_open(identity[0])
    try:
        final = snapshot(identity)
        if final is None:
            reply("skip", "candidate_changed_before_signal")
            return
        audit = audit_before_term(final, probes)
        signal.pidfd_send_signal(descriptor, signal.SIGTERM)
        reply("term_sent", "verified_orphan_only", pid=identity[0],
              start_ticks=identity[1], audit_path=audit)
    finally:
        os.close(descriptor)

try:
    main()
except Exception as exc:
    # No command output, environment, proxy configuration, URLs or credentials.
    reply("skip", "remote_check_failed", error_type=type(exc).__name__)
'''


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def presence(pid):
    try:
        os.kill(pid, 0)
        return "present"
    except ProcessLookupError:
        return "absent"
    except (PermissionError, OSError):
        return "unknown"


def command_for(pid):
    result = subprocess.run(["ps", "-p", str(pid), "-o", "command="],
                            capture_output=True, text=True, timeout=2)
    if result.returncode != 0:
        return None
    return shlex.split(result.stdout.strip())


def local_state():
    """Only an absent PID qualifies; a reused/unknown live PID fails closed."""
    try:
        data = json.loads(RELAY_STATE.read_text())
        supervisor, child = data["supervisor_pid"], data["ssh_pid"]
        if (type(supervisor) is not int or type(child) is not int or
                supervisor <= 1 or child <= 1 or data.get("remote_port") != 18791):
            return {"eligible": False, "reason": "invalid_relay_state"}
        if presence(supervisor) != "present":
            return {"eligible": False, "reason": "supervisor_not_alive"}
        args = command_for(supervisor)
        if not args or str(SUPERVISOR_SCRIPT) not in args:
            return {"eligible": False, "reason": "supervisor_identity_unverified"}
        status = presence(child)
        return {"eligible": status == "absent", "reason": "ssh_" + status,
                "supervisor_pid": supervisor, "ssh_pid": child}
    except (OSError, ValueError, KeyError, subprocess.SubprocessError):
        return {"eligible": False, "reason": "relay_state_unreadable"}


class Watchdog:
    def __init__(self):
        self.stop = threading.Event()
        self.child = None
        self.last_sample = 0.0
        self.sample = {}
        self.last_repair = 0.0
        self.logger = logging.getLogger("rdm-relay-watchdog")
        self.logger.setLevel(logging.INFO)
        handler = RotatingFileHandler(OWN_LOG, maxBytes=256 * 1024, backupCount=2)
        handler.setFormatter(logging.Formatter("%(message)s"))
        self.logger.addHandler(handler)

    def log(self, event, **data):
        record = {"utc": utc(), "event": event, **data}
        message = json.dumps(record, sort_keys=True)
        self.logger.info(message)
        print(message, flush=True)

    def save_state(self, **extra):
        data = {"watchdog_pid": os.getpid(), "updated_utc": utc(),
                "local": self.sample, **extra}
        temporary = OWN_STATE.with_name(OWN_STATE.name + f".{os.getpid()}.tmp")
        with temporary.open("w") as stream:
            json.dump(data, stream, indent=2)
            stream.write("\n")
        os.replace(temporary, OWN_STATE)

    def read_local(self):
        self.sample = local_state()
        self.last_sample = time.monotonic()
        return self.sample

    def request_stop(self, signum, unused):
        self.stop.set()
        if self.child is not None and self.child.poll() is None:
            self.child.terminate()  # Only this watchdog's management SSH client.

    def remote(self, phase, candidate=None):
        args = ["python3", "-", phase]
        if candidate is not None:
            args += [str(candidate["pid"]), str(candidate["start_ticks"])]
        command = ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
                   "-o", "ConnectionAttempts=1", "-o", "ServerAliveInterval=5",
                   "-o", "ServerAliveCountMax=2", HOST, shlex.join(args)]
        if self.stop.is_set():
            return {"status": "skip", "reason": "stopping"}
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True)
        self.child = process
        started, payload = time.monotonic(), REMOTE_PROGRAM
        try:
            while True:
                if self.stop.is_set():
                    process.terminate()
                    return {"status": "skip", "reason": "stopping"}
                if time.monotonic() - started >= SSH_TIMEOUT:
                    process.terminate()
                    return {"status": "skip", "reason": "ssh_timeout"}
                if time.monotonic() - self.last_sample >= POLL_SECONDS:
                    if not self.read_local().get("eligible"):
                        process.terminate()
                        return {"status": "skip", "reason": "local_ssh_or_supervisor_changed"}
                try:
                    stdout, stderr = process.communicate(input=payload, timeout=1)
                    break
                except subprocess.TimeoutExpired:
                    payload = None
            if process.returncode != 0:
                return {"status": "skip", "reason": "management_ssh_failed",
                        "ssh_exit": process.returncode}
            result = json.loads(stdout)
            if not isinstance(result, dict):
                raise ValueError("invalid_response")
            return result
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            return {"status": "skip", "reason": "management_check_failed",
                    "error_type": type(exc).__name__}
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.communicate(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()  # Only our local management client, never remote sshd.
                    process.communicate()
            self.child = None

    def cycle(self):
        first = self.read_local()
        self.save_state(status="monitoring")
        if not first.get("eligible"):
            return
        if time.monotonic() - self.last_repair < REPAIR_COOLDOWN:
            return
        checked = self.remote("phase1")
        if checked.get("status") != "candidate":
            self.log("phase1_skipped", **checked)
            return
        if (type(checked.get("pid")) is not int or checked["pid"] <= 1 or
                type(checked.get("start_ticks")) is not int or checked["start_ticks"] <= 0):
            self.log("invalid_remote_candidate")
            return
        current = self.read_local()
        if (self.stop.is_set() or not current.get("eligible") or
                current.get("supervisor_pid") != first.get("supervisor_pid")):
            self.log("phase2_skipped", reason="local_state_recovered_or_changed")
            return
        result = self.remote("phase2", checked)
        self.log("phase2_result", **result)
        if result.get("status") == "term_sent":
            self.last_repair = time.monotonic()
        self.save_state(status="monitoring", last_result=result)

    def run(self):
        self.log("watchdog_started", watchdog_pid=os.getpid(), poll_seconds=POLL_SECONDS)
        try:
            while not self.stop.is_set():
                due = time.monotonic() + POLL_SECONDS
                try:
                    self.cycle()
                except Exception as exc:
                    self.log("cycle_skipped", error_type=type(exc).__name__)
                self.stop.wait(max(0.0, due - time.monotonic()))
        finally:
            self.save_state(status="stopped")
            self.log("watchdog_stopped", watchdog_pid=os.getpid())


def main():
    # Separate sidecar lock; never create, replace or acquire the experiment lock.
    with OWN_LOCK.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in (errno.EACCES, errno.EAGAIN):
                raise SystemExit("A relay watchdog already owns this sidecar lock.")
            raise
        watchdog = Watchdog()
        signal.signal(signal.SIGTERM, watchdog.request_stop)
        signal.signal(signal.SIGINT, watchdog.request_stop)
        watchdog.run()


if __name__ == "__main__":
    main()
