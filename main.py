import os
import subprocess
import logging
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
logging.basicConfig(
    filename="pipeline.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
BASE_DIR = Path(__file__).resolve().parent
PIPELINE = [
    ("Data Collection", "collect_all.py"),
    ("Preprocessing", "preprocessing.py"),
    ("Model Testing", "model_testing.py"),
    ("Bias Metrics", "bias_metrics.py"),
    ("Model Validation", "model_check.py"),
    ("Visualization", "visualise_results.py")
]
def run_step(step_name, script_name):
    script_path = BASE_DIR / script_name
    if not script_path.exists():
        logging.error(f"Script not found: {script_path}")
        print(f"Error: Script not found: {script_path}")
        return
    logging.info(f"Starting: {step_name}")
    print(f"\nRunning {step_name} ({script_name})")
    try:
        subprocess.run(["python", str(script_path)], check=True)
        logging.info(f"Completed: {step_name}")
        print(f"{step_name} completed successfully.")
    except subprocess.CalledProcessError as e:
        logging.error(f"{step_name} failed: {e}")
        print(f"Error: {step_name} failed. See pipeline.log for details.")
        raise
def main():
    for step_name, script_name in PIPELINE:
        choice = input(f"Run '{step_name}'? (y/n): ").strip().lower()
        if choice == "y":
            run_step(step_name, script_name)
        else:
            logging.info(f"Skipped: {step_name}")
            print(f"Skipped {step_name}.")
    print("\nPipeline execution complete.")
if __name__ == "__main__":
    main()
