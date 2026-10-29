import sys
import subprocess
import time

def run_step(step_name, command):
    print(f"\n--- Running: {step_name} ---")
    t0 = time.time()
    result = subprocess.run([sys.executable] + command, capture_output=True, text=True)
    elapsed = time.time() - t0
    
    if result.stdout:
        print(result.stdout.strip())
    if result.stderr:
        print(f"Warnings / stderr:\n{result.stderr.strip()}", file=sys.stderr)
        
    if result.returncode != 0:
        print(f"Error in {step_name} (exit code {result.returncode})", file=sys.stderr)
        sys.exit(result.returncode)
        
    print(f"Completed {step_name} in {elapsed:.2f}s")

def main():
    print("Starting End-to-End Smart-Meter Cleaning and NILM Disaggregation Pipeline")
    t_start = time.time()
    
    # Section 1: Clean raw House 1 data and plot before/after
    run_step("Section 1: Raw Data Cleaning & Resampling", ["src/01_clean_raw.py"])
    
    # Section 2: EDA and cycle detection across 4 households
    run_step("Section 2: Washing Machine EDA Across Households", ["src/02_eda_washing_machine.py"])
    
    # Section 3: NILM Disaggregation Training & Evaluation
    run_step("Section 3: NILM Training, Evaluation & Failure Analysis", ["src/03_train_evaluate_nilm.py"])
    
    total_elapsed = time.time() - t_start
    print(f"\nPipeline finished successfully in {total_elapsed:.1f}s.")
    print("All artifacts and plots generated in figures/ directory.")

if __name__ == "__main__":
    main()
