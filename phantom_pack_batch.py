import os
from  phantom_pack import phantom_pack

if __name__ == '__main__':
    my_dirs = os.listdir()
    for my_dir in my_dirs:
        if os.path.isdir(my_dir):
            print(f"phantom_pack {my_dir}")
            results = phantom_pack(my_dir)
            