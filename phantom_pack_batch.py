import os
import sys
from  phantom_pack import phantom_pack

if __name__ == '__main__':
    all_results_dir = os.path.join(sys.argv[1], 'phantom_pack_results')
    my_dirs = [os.path.join(sys.argv[1], my_dir) for my_dir in os.listdir(sys.argv[1])]
    for my_dir in my_dirs:
        if os.path.isdir(my_dir):
            print(f"phantom_pack {my_dir}")
            results = phantom_pack(my_dir)
            