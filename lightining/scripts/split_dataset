#!/usr/bin/env python3

from pathlib import Path
from tqdm import tqdm
import random
import shutil
import sys

if len(sys.argv) < 2:
    print("Please provide the root directory path of dataset")
    sys.exit(1)

src_dir = Path(sys.argv[1])

if not src_dir.exists(): print(f"{src_dir.as_posix()} does not exist")

train_dir_path = src_dir.parent / f"{src_dir.name}_split" /  "train"
test_dir_path = src_dir.parent / f"{src_dir.name}_split" /  "test"

split_ratio = 0.8

def split_data(src_dir, train_dir, test_dir, split):
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    for subdir in src_dir.iterdir():
        if not subdir.is_dir():  # Skip if not a directory
            continue
        
        # Create corresponding sub-directories in the destination directories
        train_subdir = train_dir / subdir.name
        test_subdir = test_dir / subdir.name
        
        train_subdir.mkdir(parents=True, exist_ok=True)
        test_subdir.mkdir(parents=True, exist_ok=True)
        
        # only searched for .jpg, modify if necessary
        files = list(subdir.glob('*.jpg'))
        
        # shuffle the files
        random.shuffle(files)
        
        split_index = int(len(files) * split)
        
        for i, file in tqdm(enumerate(files), total=len(files), desc=f"Processing {subdir.name}", unit="file"):
            if i < split_index:
                shutil.copy2(file, train_subdir)
            else:
                shutil.copy2(file, test_subdir)

split_data(src_dir, train_dir_path, test_dir_path, split_ratio)