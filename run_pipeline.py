"""
Master End-to-End Pipeline Runner for Flock Energy NILM Project.

Executes all project stages sequentially or individually:
  Stage 1: Setup validation and data sanity check
  Stage 2: Raw data inspection & 1-minute cleaning for House 1
  Stage 3: Multi-household washing machine EDA & cycle detection
  Stage 4: Appliance-disaggregation modeling & baseline evaluation

Usage:
  python run_pipeline.py --all
  python run_pipeline.py --stage 1
  python run_pipeline.py --stage 2
  python run_pipeline.py --stage 3
  python run_pipeline.py --stage 4
"""
import sys, os
import argparse
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, ".")
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["MPLCONFIGDIR"] = os.path.expanduser("~/.matplotlib_tmp")

def run_stage_1():
    print("\n" + "=" * 70)
    print("  RUNNING STAGE 1: Environment & Data Ingestion Setup")
    print("=" * 70)
    from src.validate_setup import main as validate_main
    validate_main()

def run_stage_2():
    print("\n" + "=" * 70)
    print("  RUNNING STAGE 2: Raw House 1 Inspection, Cleaning & Resampling")
    print("=" * 70)
    from src.inspect_raw_house1 import main as inspect_main
    from src.clean_house1 import clean_house1
    from src.plot_house1_cleaning import main as plot_cleaning_main

    print("\n--> [2.1] Inspecting raw telemetry ...")
    inspect_main()
    print("\n--> [2.2] Executing cleaning & 1-minute resampling pipeline ...")
    clean_house1(save=True)
    print("\n--> [2.3] Generating cleaning comparison plots ...")
    plot_cleaning_main()

def run_stage_3():
    print("\n" + "=" * 70)
    print("  RUNNING STAGE 3: Multi-Household Washing Machine EDA & Cycle Detection")
    print("=" * 70)
    from src.washing_machine_eda import run_eda
    run_eda([1, 2, 5, 7, 20])

def run_stage_4():
    print("\n" + "=" * 70)
    print("  RUNNING STAGE 4: NILM Disaggregation Modeling & Baseline Benchmark")
    print("=" * 70)
    from src.nilm_model import run_nilm_pipeline
    run_nilm_pipeline(house_id=1)

def main():
    parser = argparse.ArgumentParser(description="Flock Energy NILM End-to-End Pipeline")
    parser.add_argument("--all", action="store_true", help="Run entire end-to-end pipeline")
    parser.add_argument("--stage", type=int, choices=[1, 2, 3, 4], help="Run a specific stage (1-4)")
    args = parser.parse_args()

    start_time = time.time()
    
    if args.all or args.stage is None:
        print("\n" + "#" * 70)
        print("  EXECUTING COMPLETE END-TO-END FLOCK ENERGY DATA SCIENCE PIPELINE")
        print("#" * 70)
        run_stage_1()
        run_stage_2()
        run_stage_3()
        run_stage_4()
    elif args.stage == 1:
        run_stage_1()
    elif args.stage == 2:
        run_stage_2()
    elif args.stage == 3:
        run_stage_3()
    elif args.stage == 4:
        run_stage_4()

    elapsed = time.time() - start_time
    print("\n" + "#" * 70)
    print(f"  PIPELINE EXECUTION COMPLETED IN {elapsed/60:.2f} MINUTES")
    print("#" * 70)

if __name__ == "__main__":
    main()
