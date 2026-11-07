"""
Convenience entry point for the NILM Disaggregation Pipeline.

Delegates execution to run_pipeline.main() with full CLI support:
  python pipeline.py --all
  python pipeline.py --stage 4 --loho
"""
import sys
from run_pipeline import main

if __name__ == "__main__":
    main()
