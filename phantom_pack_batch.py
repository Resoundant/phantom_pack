import os
import sys
import shutil
from  phantom_pack import phantom_pack
from collate_outputs import collate_outputs

import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())

if __name__ == '__main__':
    all_results_dir = os.path.join(sys.argv[1], 'phantompack_results_combined')
    my_dirs = [os.path.join(sys.argv[1], my_dir) for my_dir in os.listdir(sys.argv[1])]
    for my_dir in my_dirs:
        if os.path.isdir(my_dir):
            print(f"phantom_pack {my_dir}")
            results = phantom_pack(my_dir)
            shutil.copytree(os.path.join(my_dir, 'phantompack_results'), os.path.join(all_results_dir, os.path.basename(my_dir)), dirs_exist_ok=True)

    collate_outputs(all_results_dir)