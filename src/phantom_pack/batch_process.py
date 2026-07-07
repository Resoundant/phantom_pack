import os
import argparse
import shutil
from .phantom_pack import process_input
from .collate_outputs import collate_outputs

import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())


def batch_process(top_directory: str | os.PathLike):
    top_directory = os.fspath(top_directory)
    all_results_dir = os.path.join(top_directory, 'phantompack_results_combined')
    my_dirs = [os.path.join(top_directory, my_dir) for my_dir in os.listdir(top_directory)]
    for my_dir in my_dirs:
        if os.path.isdir(my_dir):
            print(f"phantom_pack {my_dir}")
            results = process_input(my_dir)
            shutil.copytree(os.path.join(my_dir, 'phantompack_results'), os.path.join(all_results_dir, os.path.basename(my_dir)), dirs_exist_ok=True)

    collate_outputs(all_results_dir)
    return all_results_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Phantom Pack analysis over each child directory.")
    parser.add_argument("top_directory", help="Directory containing one subdirectory per patient-exam.")
    args = parser.parse_args(argv)

    batch_process(args.top_directory)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
