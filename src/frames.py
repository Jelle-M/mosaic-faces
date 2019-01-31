#!/usr/bin/python3
# Roding: utf-8

""" Extract frames from video """
import argparse
import logging as log
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def get_desired_frames(length, n_frames, uniform=True):
    if uniform:
        interval = int((length) / n_frames)
        desired_frames = np.arange(interval, length, interval)
        return desired_frames
    X1 = np.random.normal(
        loc=length / 4, scale=length / 4, size=int(n_frames / 2))
    X1 = X1.astype(int)
    X2 = np.random.normal(loc=length / 2 + length / 4,
                          scale=length / 4, size=int(n_frames / 2))
    X2 = X2.astype(int)
    X = np.hstack((X1, X2))
    return X


def write(image, out_dir, episode, index):
    out_dir = Path(out_dir / episode)
    if not out_dir.exists():
        out_dir.mkdir()
    frame_name = "{0}.jpg".format(index)
    cv2.imwrite(str(out_dir / frame_name), image)


def extract_frames(video_file, out_dir, n_frames=10, uniform=True, episode=''):
    cap = cv2.VideoCapture(video_file)
    _, image = cap.read()
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    desired_frames = get_desired_frames(length, n_frames, uniform=uniform)
    n = len(desired_frames)
    for i, index in tqdm(zip(desired_frames, range(n)),
                         total=n, unit="frames"):
        cap.set(1, i - 1)
        _, image = cap.read(1)
        cap.get(1)
        write(image, out_dir, episode, index)


def parse_args():
    parser = argparse.ArgumentParser(description='process args')
    parser.add_argument('video_file',
                        help='path to videofile')
    parser.add_argument('-n', '--n_frames', type=int, default=10,
                        help='amount of frames')
    parser.add_argument('-e', '--episode', dest='episode', default='default',
                        help='episode counter')
    parser.add_argument('-o', '--out', dest='out_dir',
                        default='../frames/', help='output dir')
    parser.add_argument('-v', '--verbose', dest="verbose", action='store_true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    if args.verbose:
        log.basicConfig(format="%(levelname)s: %(message)s", level=log.DEBUG)
        log.info("Verbose output.")
    else:
        log.basicConfig(format="%(levelname)s: %(message)s")

    # Create out_dir
    out_dir = Path(args.out_dir)
    if not out_dir.exists():
        out_dir.mkdir()

    # Extract images
    extract_frames(
        args.video_file, out_dir, episode=Path(args.video_file).stem,
        n_frames=args.n_frames, uniform=True)
