"""
Central configuration for the REFIT NILM pipeline.

All paths, constants, and appliance mappings live here so every other module
imports a single source of truth.
"""

from pathlib import Path

# ─── Project Paths ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "REFIT_RAW_081116"
CLEAN_DIR = DATA_DIR / "CLEAN_REFIT_081116"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
PLOT_DIR = OUTPUT_DIR / "plots"
TABLE_DIR = OUTPUT_DIR / "tables"

# ─── Sampling ───────────────────────────────────────────────────────────────
RAW_SAMPLE_INTERVAL_SEC = 8          # Nominal polling interval (6-8 s)
RESAMPLE_INTERVAL = "1min"           # Target resolution after cleaning

# ─── REFIT Appliance Map ────────────────────────────────────────────────────
# Maps (house_id, appliance_column) → appliance name.
# Only the washing-machine channel per house is highlighted below; the full
# map is provided for completeness so downstream modules can look up any
# appliance by house.

APPLIANCE_MAP: dict[int, dict[str, str]] = {
    1: {
        "Appliance1": "Fridge",
        "Appliance2": "Freezer_1",
        "Appliance3": "Freezer_2",
        "Appliance4": "Washer_Dryer",
        "Appliance5": "Washing_Machine",
        "Appliance6": "Dishwasher",
        "Appliance7": "Computer",
        "Appliance8": "Television_Site",
        "Appliance9": "Electric_Heater",
    },
    2: {
        "Appliance1": "Fridge_Freezer",
        "Appliance2": "Washing_Machine",
        "Appliance3": "Dishwasher",
        "Appliance4": "Television_Site",
        "Appliance5": "Microwave",
        "Appliance6": "Toaster",
        "Appliance7": "Hi_Fi",
        "Appliance8": "Kettle",
        "Appliance9": "Overhead_Fan",
    },
    3: {
        "Appliance1": "Toaster",
        "Appliance2": "Fridge_Freezer",
        "Appliance3": "Freezer",
        "Appliance4": "Tumble_Dryer",
        "Appliance5": "Dishwasher",
        "Appliance6": "Washing_Machine",
        "Appliance7": "Television_Site",
        "Appliance8": "Microwave",
        "Appliance9": "Kettle",
    },
    4: {
        "Appliance1": "Fridge",
        "Appliance2": "Freezer",
        "Appliance3": "Fridge_Freezer",
        "Appliance4": "Washing_Machine",
        "Appliance5": "Tumble_Dryer",
        "Appliance6": "Dishwasher",
        "Appliance7": "Television_Site",
        "Appliance8": "Microwave",
        "Appliance9": "Kettle",
    },
    5: {
        "Appliance1": "Fridge_Freezer",
        "Appliance2": "Tumble_Dryer",
        "Appliance3": "Washing_Machine",
        "Appliance4": "Dishwasher",
        "Appliance5": "Food_Mixer",
        "Appliance6": "Television",
        "Appliance7": "Kettle",
        "Appliance8": "Vivarium",
        "Appliance9": "Pond_Pump",
    },
    6: {
        "Appliance1": "Freezer",
        "Appliance2": "Washing_Machine",
        "Appliance3": "Dishwasher",
        "Appliance4": "MJY_Computer",
        "Appliance5": "Television_Site",
        "Appliance6": "Microwave",
        "Appliance7": "Kettle",
        "Appliance8": "Toaster",
        "Appliance9": "PGM_Computer",
    },
    7: {
        "Appliance1": "Fridge",
        "Appliance2": "Freezer_1",
        "Appliance3": "Freezer_2",
        "Appliance4": "Tumble_Dryer",
        "Appliance5": "Washing_Machine",
        "Appliance6": "Dishwasher",
        "Appliance7": "Television_Site",
        "Appliance8": "Toaster",
        "Appliance9": "Kettle",
    },
    8: {
        "Appliance1": "Fridge",
        "Appliance2": "Freezer",
        "Appliance3": "Washer_Dryer",
        "Appliance4": "Washing_Machine",
        "Appliance5": "Toaster",
        "Appliance6": "Computer",
        "Appliance7": "Television_Site",
        "Appliance8": "Microwave",
        "Appliance9": "Kettle",
    },
    9: {
        "Appliance1": "Fridge_Freezer",
        "Appliance2": "Washer_Dryer",
        "Appliance3": "Washing_Machine",
        "Appliance4": "Dishwasher",
        "Appliance5": "Television_Site",
        "Appliance6": "Microwave",
        "Appliance7": "Kettle",
        "Appliance8": "Hi_Fi",
        "Appliance9": "Electric_Heater",
    },
    10: {
        "Appliance1": "Magimix_Blender",
        "Appliance2": "Toaster",
        "Appliance3": "Chest_Freezer",
        "Appliance4": "Fridge_Freezer",
        "Appliance5": "Washing_Machine",
        "Appliance6": "Dishwasher",
        "Appliance7": "Television_Site",
        "Appliance8": "Microwave",
        "Appliance9": "K_Mix",
    },
    11: {
        "Appliance1": "Fridge",
        "Appliance2": "Fridge_Freezer",
        "Appliance3": "Washing_Machine",
        "Appliance4": "Dishwasher",
        "Appliance5": "Computer_Site",
        "Appliance6": "Microwave",
        "Appliance7": "Kettle",
        "Appliance8": "Router",
        "Appliance9": "Hi_Fi",
    },
    12: {
        "Appliance1": "Fridge_Freezer",
        "Appliance2": "Unknown_1",
        "Appliance3": "Unknown_2",
        "Appliance4": "Computer_Site",
        "Appliance5": "Microwave",
        "Appliance6": "Kettle",
        "Appliance7": "Toaster",
        "Appliance8": "Television",
        "Appliance9": "Unknown_3",
    },
    13: {
        "Appliance1": "Television_Site",
        "Appliance2": "Freezer",
        "Appliance3": "Washing_Machine",
        "Appliance4": "Dishwasher",
        "Appliance5": "Unknown",
        "Appliance6": "Network_Site",
        "Appliance7": "Microwave_1",
        "Appliance8": "Microwave_2",
        "Appliance9": "Kettle",
    },
    15: {
        "Appliance1": "Fridge_Freezer",
        "Appliance2": "Tumble_Dryer",
        "Appliance3": "Washing_Machine",
        "Appliance4": "Dishwasher",
        "Appliance5": "Computer_Site",
        "Appliance6": "Television_Site",
        "Appliance7": "Microwave",
        "Appliance8": "Hi_Fi",
        "Appliance9": "Toaster",
    },
    16: {
        "Appliance1": "Fridge_Freezer_1",
        "Appliance2": "Fridge_Freezer_2",
        "Appliance3": "Electric_Heater_1",
        "Appliance4": "Electric_Heater_2",
        "Appliance5": "Washing_Machine",
        "Appliance6": "Dishwasher",
        "Appliance7": "Computer_Site",
        "Appliance8": "Television_Site",
        "Appliance9": "Dehumidifier",
    },
    17: {
        "Appliance1": "Freezer",
        "Appliance2": "Fridge_Freezer",
        "Appliance3": "Tumble_Dryer",
        "Appliance4": "Washing_Machine",
        "Appliance5": "Computer_Site",
        "Appliance6": "Television_Site",
        "Appliance7": "Microwave",
        "Appliance8": "Kettle",
        "Appliance9": "TV_Site_Bedroom",
    },
    18: {
        "Appliance1": "Fridge_Garage",
        "Appliance2": "Freezer_Garage",
        "Appliance3": "Fridge_Freezer",
        "Appliance4": "Washer_Dryer_Garage",
        "Appliance5": "Washing_Machine",
        "Appliance6": "Dishwasher",
        "Appliance7": "Desktop_Computer",
        "Appliance8": "Television_Site",
        "Appliance9": "Microwave",
    },
    19: {
        "Appliance1": "Fridge_Freezer",
        "Appliance2": "Washing_Machine",
        "Appliance3": "Television_Site",
        "Appliance4": "Microwave",
        "Appliance5": "Kettle",
        "Appliance6": "Toaster",
        "Appliance7": "Bread_Maker",
        "Appliance8": "Games_Console",
        "Appliance9": "Hi_Fi",
    },
    20: {
        "Appliance1": "Fridge",
        "Appliance2": "Freezer",
        "Appliance3": "Tumble_Dryer",
        "Appliance4": "Washing_Machine",
        "Appliance5": "Dishwasher",
        "Appliance6": "Computer_Site",
        "Appliance7": "Television_Site",
        "Appliance8": "Microwave",
        "Appliance9": "Kettle",
    },
    21: {
        "Appliance1": "Fridge_Freezer",
        "Appliance2": "Tumble_Dryer",
        "Appliance3": "Washing_Machine",
        "Appliance4": "Dishwasher",
        "Appliance5": "Food_Mixer",
        "Appliance6": "Television",
        "Appliance7": "Kettle",
        "Appliance8": "Vivarium",
        "Appliance9": "Pond_Pump",
    },
}

# ─── Washing-Machine Channel Lookup ────────────────────────────────────────
# { house_id: column_name }  — derived from APPLIANCE_MAP for convenience.
WASHING_MACHINE_CHANNEL: dict[int, str] = {
    house: col
    for house, mapping in APPLIANCE_MAP.items()
    for col, name in mapping.items()
    if name == "Washing_Machine"
}

# Houses selected for the EDA / modelling work (≥ 4, with clear WM channel).
SELECTED_HOUSES = [1, 2, 5, 7, 20]

# ─── Physical Bounds ───────────────────────────────────────────────────────
MAX_POWER_WATTS = 10_000   # Readings above this are physically implausible
MIN_POWER_WATTS = 0        # Negative active-power readings are invalid
SPIKE_THRESHOLD_IAM = 4000 # REFIT cleaned dataset already caps IAMs at 4 kW

