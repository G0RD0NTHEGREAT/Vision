import os
import sys
import pdb
import time
import glob
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, help='the image directory')
	args = parser.parse_args()

	with open(args.img_dir, 'rb') as handle:
		b = pickle.load(handle)
	print(b)