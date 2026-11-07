"""
Master End-to-End Pipeline Runner:
  Stage 1: Setup validation and data sanity check
  Stage 2: Raw data inspection & 1-minute cleaning for House 1
  Stage 3: Multi-household washing machine EDA & cycle detection
  Stage 4: Appliance-disaggregation modeling & baseline evaluation (Single-House or LOHO)
  Stage 5: Macro energy attribution via Non-Negative Matrix Factorization (30-min AMI data)

Usage:
  python run_pipeline.py --all
  python run_pipeline.py --stage <1-5>
  python run_pipeline.py --stage 4 --loho
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
from src.config import OUTPUT_DIR
cache_dir = OUTPUT_DIR / ".matplotlib_cache"
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(cache_dir)

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

def run_stage_4(
    max_pause: int = 5,
    stitch_method: str = "dynamic_bridge",
    active_thresh: float = 15.0,
    tolerance_min: int = 5,
    feature_strategy: str = "rolling",
    loho: bool = False,
    train_houses: list[int] = [1, 2, 5, 7],
    test_house: int = 20,
):
    print("\n" + "=" * 70)
    print("  RUNNING STAGE 4: NILM Disaggregation Modeling & Benchmark")
    print(f"  Mode: {'Leave-One-House-Out (LOHO)' if loho else 'Single Household (House 1)'}")
    print(f"  Config: method={stitch_method}, pause={max_pause}m, thresh={active_thresh}W, tolerance={tolerance_min}m, features={feature_strategy}")
    print("=" * 70)
    if loho:
        from src.nilm_model import run_loho_experiment
        run_loho_experiment(
            train_houses=train_houses,
            test_house=test_house,
            active_threshold_watts=active_thresh,
            max_pause_min=max_pause,
            tolerance_min=tolerance_min,
            cycle_stitch_method=stitch_method,
            feature_strategy=feature_strategy,
        )
    else:
        from src.nilm_model import run_nilm_pipeline
        run_nilm_pipeline(
            house_id=1,
            active_threshold_watts=active_thresh,
            max_pause_min=max_pause,
            tolerance_min=tolerance_min,
            cycle_stitch_method=stitch_method,
            feature_strategy=feature_strategy,
        )

def run_stage_5():
    print("\n" + "=" * 70)
    print("  RUNNING STAGE 5: Macro Energy Attribution via NMF (30-min AMI Data)")
    print("=" * 70)
    from src.macro_disaggregation_nmf import run_macro_disaggregation_nmf
    run_macro_disaggregation_nmf()

def main():
    parser = argparse.ArgumentParser(description="NILM Disaggregation & Energy Analytics End-to-End Pipeline")
    parser.add_argument("--all", action="store_true", help="Run entire end-to-end pipeline (Stages 1-5)")
    parser.add_argument("--stage", type=int, choices=[1, 2, 3, 4, 5], help="Run a specific stage (1-5)")
    parser.add_argument("--loho", action="store_true", help="Run Leave-One-House-Out cross-validation in Stage 4")
    parser.add_argument("--train-houses", nargs="+", type=int, default=[1, 2, 5, 7], help="Training house IDs for LOHO")
    parser.add_argument("--test-house", type=int, default=20, help="Target unseen test house ID for LOHO")
    parser.add_argument("--max-pause", type=int, default=5, help="Bridgeable pause duration in minutes (e.g. 5, 8, 10)")
    parser.add_argument("--stitch-method", choices=["dynamic_bridge", "dbscan"], default="dynamic_bridge",
                        help="Cycle stitching method: dynamic_bridge or dbscan")
    parser.add_argument("--active-thresh", type=float, default=15.0, help="Active power threshold (W)")
    parser.add_argument("--tolerance-min", type=int, default=5, help="Event matching tolerance window (minutes)")
    parser.add_argument("--feature-strategy", choices=["rolling", "ewma", "hybrid"], default="rolling",
                        help="Feature extraction strategy: rolling (O(W) window buffer), ewma (O(1) memory state), or hybrid")
    args = parser.parse_args()

    start_time = time.time()
    
    stage_4_kwargs = dict(
        max_pause=args.max_pause,
        stitch_method=args.stitch_method,
        active_thresh=args.active_thresh,
        tolerance_min=args.tolerance_min,
        feature_strategy=args.feature_strategy,
        loho=args.loho,
        train_houses=args.train_houses,
        test_house=args.test_house,
    )

    if args.all or args.stage is None:
        print("\n" + "#" * 70)
        print("  EXECUTING COMPLETE END-TO-END NILM DATA SCIENCE PIPELINE")
        print("#" * 70)
        run_stage_1()
        run_stage_2()
        run_stage_3()
        run_stage_4(**stage_4_kwargs)
        run_stage_5()
    elif args.stage == 1:
        run_stage_1()
    elif args.stage == 2:
        run_stage_2()
    elif args.stage == 3:
        run_stage_3()
    elif args.stage == 4:
        run_stage_4(**stage_4_kwargs)
    elif args.stage == 5:
        run_stage_5()

    elapsed = time.time() - start_time
    print("\n" + "#" * 70)
    print(f"  PIPELINE EXECUTION COMPLETED IN {elapsed/60:.2f} MINUTES")
    print("#" * 70)

if __name__ == "__main__":
    main()
