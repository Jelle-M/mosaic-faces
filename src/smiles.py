#!/usr/bin/python3
# USAGE
# python detect_faces_video.py --prototxt deploy.prototxt.txt --model
# res10_300x300_ssd_iter_140000.caffemodel

import argparse
import logging as log
import os
from pathlib import Path

import cv2
# import the necessary packages
import numpy as np
from PIL import Image
from tqdm import tqdm

SIZE_DIM = 50


def flat(*nums):
    # Credit to
    # https://snipnyet.com/adierebel/5b45b79b77da154922550e9a/crop-and-resize-image-with-aspect-ratio-using-pillow/
    """Build a tuple of ints from float or integer arguments.
    Useful because PIL crop and resize require integer points."""

    return tuple(int(round(n)) for n in nums)


class Size():
    # Credit to
    # https://snipnyet.com/adierebel/5b45b79b77da154922550e9a/crop-and-resize-image-with-aspect-ratio-using-pillow/

    def __init__(self, pair):
        self.width = float(pair[0])
        self.height = float(pair[1])

    @property
    def aspect_ratio(self):
        return self.width / self.height

    @property
    def size(self):
        return flat(self.width, self.height)


def cropped_thumbnail(img, size):
    # Credit to
    # https://snipnyet.com/adierebel/5b45b79b77da154922550e9a/crop-and-resize-image-with-aspect-ratio-using-pillow/
    """
    Builds a thumbnail by cropping out a maximal region from
    the center of the original with
    the same aspect ratio as the target size, and then resizing.
    The result is a thumbnail which is
    always EXACTLY the requested size and with no aspect
    ratio distortion (although two edges, either
    top/bottom or left/right depending whether the
    image is too tall or too wide, may be trimmed off.)
    """
    original = Size(img.size)
    target = Size(size)
    if target.aspect_ratio > original.aspect_ratio:
        # image is too tall: take some off the top and bottom
        scale_factor = target.width / original.width
        crop_size = Size((original.width, target.height / scale_factor))
        top_cut_line = (original.height - crop_size.height) / 2
        img = img.crop(flat(0, top_cut_line, crop_size.width,
                            top_cut_line + crop_size.height))
    elif target.aspect_ratio < original.aspect_ratio:
        # image is too wide: take some off the sides
        scale_factor = target.height / original.height
        crop_size = Size((target.width / scale_factor, original.height))
        side_cut_line = (original.width - crop_size.width) / 2
        img = img.crop(flat(side_cut_line, 0, side_cut_line +
                            crop_size.width, crop_size.height))
    return img.resize(target.size, Image.ANTIALIAS)


def mean_dimension(jpg_files):
    size_info = []
    for jpg_file in tqdm(jpg_files, total=len(jpg_files), unit="images"):
        img = Image.open(jpg_file)
        size_info.append(img.size)
    w = [s[0] for s in size_info]
    h = [s[1] for s in size_info]
    log.debug('w,h; avg mean min max')
    log.debug('{0} {1} {2} {3}'.format(int(np.average(w)),
                                       int(np.mean(w)), np.min(w), np.max(w)))
    log.debug('{0} {1} {2} {3}'.format(int(np.average(h)),
                                       int(np.mean(h)), np.min(h), np.max(h)))
    w_mean = int(np.mean(w))
    h_mean = int(np.mean(h))
    square_size = int((w_mean + h_mean) / 2)
    return square_size, square_size
    # return w_mean, h_mean


def write(image, out_dir, episode, index):
    out_dir = Path(out_dir / episode)
    if not out_dir.exists():
        out_dir.mkdir()
    frame_name = "{0}.jpg".format(index)
    cv2.imwrite(str(out_dir / frame_name), image)


def main(args):
    # Grab all filenames of images
    jpg_files = [str(p) for p in Path(args.faces_dir).glob('**/*.jpg')]
    # Calculate mean width and heigth
    # size = mean_dimension(jpg_files)
    if args.info:
        return 0
    # Resize and write to output
    smileCascade = cv2.CascadeClassifier('../cascades/haarcascade_smile.xml')
    for jpg_file in tqdm(jpg_files, total=len(jpg_files), unit="images"):
        try:
            img = cv2.imread(jpg_file)
        except BaseException:
            os.system(f'rm {jpg_file}')
            continue
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        except BaseException:
            continue

        smile = smileCascade.detectMultiScale(
            gray,
            scaleFactor=1.25,
            minNeighbors=27,
            minSize=(10, 10),
        )
        for (xx, yy, ww, hh) in smile:
            cv2.rectangle(gray, (xx, yy), (xx + ww, yy + hh), (0, 255, 0), 2)
        if args.verbose:
            cv2.imshow('video', gray)
            k = cv2.waitKey(0) & 0xff
            if k == ord('q'):  # press 'ESC' to quit
                cv2.destroyAllWindows()
                break
        if smile:
            face_name = Path(jpg_file).name
            episode = Path(jpg_file).parents[0].stem
            write(img, out_dir, episode, face_name)
    return 0


# construct the argument parse and parse the arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--out', dest='out_dir', default='../smiles/',
                        help='output detection dir')
    parser.add_argument('-f', '--faces', dest='faces_dir', default='../faces/',
                        help='output faces dir')
    parser.add_argument('-v', '--verbose', dest="verbose", action='store_true')
    parser.add_argument('-I', dest="info", action='store_true')
    parser.add_argument('-s', dest="size", default=50, type=int)
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

    SIZE_DIM = args.size

    exit_code = main(args)
    exit(exit_code)
