"""Own-reference (mechanism) ablation of SW-LMMD (see README.md).

* ``window-nodiag``       -- the window loss without its cross diagonal k(X_i, Y_i), rescaled K/(K-1);
* ``global-plus-self-b*`` -- the Table 2 Global arm plus beta * L_self (own-row attraction);
* Step 0 probe            -- the own term's share of the L_MMD feature gradient on main-run checkpoints.
"""
from .config import MAIN_ORDER_HASH, VARIANTS, MechanismConfig, mechanism_from_config
from .loss import MechanismLoss
from .trainer import MechanismTrainer

__all__ = ["MAIN_ORDER_HASH", "VARIANTS", "MechanismConfig", "MechanismLoss", "MechanismTrainer",
           "mechanism_from_config"]
