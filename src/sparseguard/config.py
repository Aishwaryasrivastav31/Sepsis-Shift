"""Paths, variable lists and experiment settings for SparseGuard."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"  # training_setA/ and training_setB/ from PhysioNet 2019
CACHE = ROOT / "data" / "cache"
TAB = ROOT / "reports" / "tables"
FIG = ROOT / "reports" / "figures"

VITALS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2"]
LABS = [
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN", "Alkalinephos", "Calcium",
    "Chloride", "Creatinine", "Bilirubin_direct", "Glucose", "Lactate", "Magnesium", "Phosphate",
    "Potassium", "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC", "Fibrinogen", "Platelets",
]  # fmt: skip
CLINICAL = VITALS + LABS  # the 34 time-varying measurements that can be missing
STATIC = ["Age", "Gender", "Unit1", "Unit2", "HospAdmTime"]
LABEL = "SepsisLabel"

SEED = 2019
SPLIT_A_TRAIN = 0.7  # hospital A: 70% of patients train the model, 30% calibrate (A-calibration)
SPLIT_B_CAL = 0.5  # hospital B: 50% of patients calibrate (B-calibration), 50% test
SPARSITY_WINDOW_H = 24  # chart fullness = distinct lab variables measured in the previous 24 hours
N_IMPUTATIONS = 10  # multiple imputation draws from the GP posterior
ALPHAS = (0.10, 0.20)  # target share of sepsis hours that may receive a confident all-clear
N_SPLITS = 50  # repeated random calibration/test splits of hospital B
