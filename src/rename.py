#!/usr/bin/python3
# Roding: utf-8

""" Extract frames from video """
import argparse
import os
import cv2
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging as log
import re


def parse_args():
    parser = argparse.ArgumentParser(description='process args')
    parser.add_argument('-v', '--verbose', dest="verbose", action='store_true')
    parser.add_argument('--videos', dest="video_dir", default='../videos/')
    parser.add_argument('--prefix', dest="prefix", default='s02')
    args = parser.parse_args()
    return args


def main(args):
    video_dir = Path(args.video_dir)
    video_files = [str(v) for v in video_dir.glob('*')]
    for video_file in tqdm(video_files, total=len(video_files), unit="videos"):
        file_name = Path(video_file).stem
        suffix = Path(video_file).suffix
        m = re.search(r'(?i)episode\s(\d+)', file_name)
        if m is None:
            log.debug(f'file_name: {file_name} skipped!')
            file_name = f'_invalid_{file_name}{suffix}'
            continue
        episode = m.group(1)
        log.debug(f'episode: {episode}')
        file_name = f'{args.prefix}_{episode}{suffix}'
        file_name1 = str(video_file)
        file_name2 = str(video_dir / file_name)
        cmd = f'mv \"{file_name1}\" \"{file_name2}\"'
        print(cmd)
        os.system(cmd)
    return 0

if __name__ == '__main__':
    args = parse_args()
    if args.verbose:
        log.basicConfig(format="%(levelname)s: %(message)s", level=log.DEBUG)
        log.info("Verbose output.")
    else:
        log.basicConfig(format="%(levelname)s: %(message)s", level=log.WARNING)

    main(args)
