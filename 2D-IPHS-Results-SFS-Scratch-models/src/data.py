"""Shared column constants for MOF loading prediction."""

from __future__ import annotations

TARGET_COL = "loading"
ID_COL = "MOF_name"
ADS_COL = "ads"

TEXTURAL_COLS = ["LPD", "PLD", "SA_grav", "VF", "PSSD", "density"]
MOLECULAR_COLS = ["chg", "bond_length", "eps_eff", "sig_eff"]
PRESSURE_COL = "fugacity"

EXPERT_COL = "expert_prediction_loading"
DROP_DESCRIPTOR_COLS = MOLECULAR_COLS
ID_COLS = [ID_COL]
