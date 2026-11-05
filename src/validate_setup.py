"""
Smoke-test: validates that the data ingestion pipeline works end-to-end.

Run from the project root:
    python -m src.validate_setup
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")

from src.config import (
    RAW_DIR, CLEAN_DIR, PLOT_DIR, TABLE_DIR,
    WASHING_MACHINE_CHANNEL, SELECTED_HOUSES, APPLIANCE_MAP,
)
from src.data_ingestion import (
    load_raw_house, load_clean_house, load_clean_houses,
    get_washing_machine_col, rename_appliances, data_summary,
)

def main():
    print("=" * 60)
    print("  REFIT NILM Pipeline — Setup Validation")
    print("=" * 60)

    # 1. Check directories exist
    for d in [RAW_DIR, CLEAN_DIR, PLOT_DIR, TABLE_DIR]:
        status = "[PASS]" if d.exists() else "[FAIL] MISSING"
        print(f"  {status}  {d}")

    # 2. Check raw files for House 1
    print("\n--- Raw House 1 (first 500 rows) ---")
    df_raw = load_raw_house(1, nrows=500)
    print(f"  Shape   : {df_raw.shape}")
    print(f"  Range   : {df_raw.index.min()} → {df_raw.index.max()}")
    print(f"  Columns : {df_raw.columns.tolist()}")

    # 3. Check cleaned House 1
    print("\n--- Clean House 1 (first 500 rows) ---")
    df_clean = load_clean_house(1, nrows=500)
    print(f"  Shape   : {df_clean.shape}")
    print(f"  Range   : {df_clean.index.min()} → {df_clean.index.max()}")
    summary = data_summary(df_clean, 1)
    print(f"  Median Δt : {summary['median_interval_sec']}s")

    # 4. Washing-machine lookups
    print("\n--- Washing-Machine Channels ---")
    for h in SELECTED_HOUSES:
        col = get_washing_machine_col(h)
        name = APPLIANCE_MAP[h][col]
        print(f"  House {h:2d}: {col} → {name}")

    # 5. Multi-house load
    print("\n--- Multi-House Clean Load (first 100 rows each) ---")
    houses = load_clean_houses(SELECTED_HOUSES, nrows=100)
    for h, df in houses.items():
        print(f"  House {h:2d}: {df.shape[0]} rows, "
              f"{df.index.min().date()} → {df.index.max().date()}")

    # 6. Rename test
    df_named = rename_appliances(df_clean, 1)
    assert "Washing_Machine" in df_named.columns, "Rename failed!"
    print("\n[PASS] Appliance rename works (Washing_Machine column present)")

    print("\n" + "=" * 60)
    print("  ALL CHECKS PASSED [PASS]")
    print("=" * 60)


if __name__ == "__main__":
    main()

