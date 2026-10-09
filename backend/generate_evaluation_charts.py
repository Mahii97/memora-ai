import argparse
from pathlib import Path

from app.evaluation.charts import generate_charts


parser = argparse.ArgumentParser(description="Generate charts from a saved MemoraAI evaluation run.")
parser.add_argument("--run-dir", required=True, help="A specific evaluation_results/run_* directory")
args = parser.parse_args()

for chart in generate_charts(Path(args.run_dir)):
    print(chart)
