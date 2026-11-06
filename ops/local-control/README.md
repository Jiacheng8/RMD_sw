# Local orchestration source backup

`current/h20-recovery-ssh.py` is the exact Mac control entry used by this chat.
It reads only the owned ControlMaster socket from the external private state
file and transports Python stdin to the existing connection. No private state,
credentials, SSH keys or account configuration is included. The earlier
parameterized `../h20-recovery-ssh.py` is preserved separately.

`historical/` preserves prior launch, download, relay, observation and plotting
code originally kept outside Git. Several refer to the removed shared mount or
superseded experiments; these files document those historical versions. Current
training is `/data/rdm-workspace/RDM-twohead-lr10-v1` at a25e4c4; its entry/config
and exact scope are documented in `../H20-TWOHEAD-LR10.md`. Publication does not
start or restart any job. `SOURCE-MAP.json` records original paths and SHA256.

The inspected third-party Diffusers attention-dispatch source is not an authored
runtime patch and is excluded; package versions and the pinned FA3 revision are
already recorded by the environment and attention configuration in this repo.
