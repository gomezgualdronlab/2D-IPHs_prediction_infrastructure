from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from src.train import merge_configs, read_yaml, train_and_evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train adsorption DNN model.")
    parser.add_argument("--base-config", required=True, help="Path to base YAML config.")
    parser.add_argument("--model-config", required=True, help="Path to model YAML config.")
    parser.add_argument("--run-name", default=None, help="Optional run name.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_cfg = read_yaml(args.base_config)
    model_cfg = read_yaml(args.model_config)
    config = merge_configs(base_cfg, model_cfg)

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config["paths"]["results_dir"]) / config["features"]["name"] / run_name
    metrics = train_and_evaluate(config, output_dir)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

