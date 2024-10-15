import os

PACKAGE_DIR = os.path.dirname(os.path.dirname(__file__))
SIM_CONFIGS_DIR = os.path.join(PACKAGE_DIR, "configs", "simulation")
ANALYSIS_CONFIGS_DIR = os.path.join(PACKAGE_DIR, "configs", "analysis")
TRUTHS_DIR = os.path.join(PACKAGE_DIR, "results", "truths")
DATA_DIR = os.path.join(PACKAGE_DIR, "results", "simulation")
RESULTS_DIR = os.path.join(PACKAGE_DIR, "results", "analysis")
UTIL_DIR = os.path.join(PACKAGE_DIR, "utilities")
