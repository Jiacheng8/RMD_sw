# H20 owned ControlMaster recovery

`rdm-owned-relay-mux.py` is the preserved task-owned supervisor, with its remote
ownership record moved from the removed `/share-evpfs` filesystem to
`/data/rdm-workspace/rdm-sets/control`. The local STATE and flock paths remain the
original `2026-11-01/ben-d/work` paths consumed by `h20-recovery-ssh.py`.

If the ownership file disappeared, cleanup only releases an already terminated
master's claim when the boot changed or the claim no longer owns port 18791.
A matching live listener with a missing ownership file is never signalled.
An existing different ownership file is never adopted or overwritten by cleanup.
The existing identity/pidfd/recheck/audit/TERM path is unchanged for valid claims.

Before replacing an old supervisor, verify its exact PID and command, the actual
master terminal witness, and the current listener/remote process identity. Keep
the old state and logs. Stop only that owned local supervisor, wait for it to
release its flock, then launch this script; never run two supervisors or kill an
unregistered remote process. A connection failure does not stop or restart training.

Offline regression check:

    python3 -m pytest -q z-image/tests/test_owned_relay_recovery.py

After recovery, verify the new owned master twice using its actual control
socket and perform real remote training/log queries through the existing helper.
