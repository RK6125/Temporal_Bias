import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(
    filename="pipeline.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

base_dir = Path(__file__).resolve().parent

flow = [
    ("Data Collection", "Data_Collection", "DataCollector"),
    ("Preprocessing", "Data_Collection", "Preprocessor"),
    ("Model Testing", "Model_Analysis", "ModelTester"),
    ("Model Validation", "Model_Analysis", "Validator"),
    ("Visualization", "Model_Analysis", "Analyzer")
]


def run_step(step_name, module_name, class_name):
    logging.info(f"Starting: {step_name}")
    print(f"\nRunning {step_name}")
    try:
        module = __import__(module_name)
        step_class = getattr(module, class_name)
        step_class().run()
        logging.info(f"Completed: {step_name}")
        print(f"{step_name} completed successfully.")
    except Exception as e:
        logging.error(f"{step_name} failed: {e}")
        print(f"Error: {step_name} failed. See pipeline.log for details.")
        raise


def main():
    for step_name, module_name, class_name in flow:
        choice = input(f"Run '{step_name}'? (y/n): ").strip().lower()
        if choice == "y":
            run_step(step_name, module_name, class_name)
        else:
            logging.info(f"Skipped: {step_name}")
            print(f"Skipped {step_name}.")
    
    print("\n Execution complete.")


if __name__ == "__main__":
    main()
