#!/usr/bin/python3
# USAGE
# python detect_faces_video.py --prototxt deploy.prototxt.txt --model
# res10_300x300_ssd_iter_140000.caffemodel

# import the necessary packages
import numpy as np
import argparse
import imutils
import time
import cv2
import logging as log
from tqdm import tqdm
from pathlib import Path

from imutils import paths
import face_recognition
import pickle
import os


def main(args):
    jpg_files = [str(p)
                 for p in Path(args.in_dir).glob(f'**/{args.pattern}*.jpg')]
    known_names = []
    known_encodings = []
    for jpg_file in tqdm(jpg_files, total=len(jpg_files), unit="images"):
        log.debug(f'Processing image {jpg_file}')
        jpg_file_path = Path(jpg_file)
        frame_name = jpg_file_path.name
        name = jpg_file_path.parents[0].stem
        log.debug(frame_name)
        face_name = jpg_file_path.stem
        log.debug(face_name)

        # read img
        image = cv2.imread(jpg_file)
        h, w = image.shape[:2]
        log.debug(f'{h}, {w}')

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # compute the facial embedding for the face
        encodings = face_recognition.face_encodings(rgb)
        if not len(encodings) > 0:
            log.warning(f'No face found in {jpg_file}; Removing image')
            Path(jpg_file).unlink()
        for encoding in encodings:
            known_encodings.append(encoding)
            known_names.append(face_name)

    # dump the facial encodings + names to disk
    data = {"encodings": known_encodings, "names": known_names}
    file_name = 'encodings.pickle'
    file_name = Path(args.out_dir) / file_name
    print(f'Writing encoding data to {file_name}')
    f = file_name.open(mode="wb")
    f.write(pickle.dumps(data))
    f.close()
    return 0


# construct the argument parse and parse the arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', dest="verbose", action='store_true')
    parser.add_argument('-fp', '--pattern', dest="pattern", default='')
    parser.add_argument("-i", "--dataset", dest="in_dir", default="../ds",
                        help="path to input directory of faces + images")
    parser.add_argument("-e", "--encodings", dest='out_dir',
                        default="../", help="path to serialized db of facial encodings")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    if args.verbose:
        log.basicConfig(format="%(levelname)s: %(message)s", level=log.DEBUG)
        print("Verbose output.")
    else:
        log.basicConfig(format="%(levelname)s: %(message)s")

    # Create out_dir
    out_dir = Path(args.out_dir)
    if not out_dir.exists():
        out_dir.mkdir()

    exit_code = main(args)

    # Exit
    exit(exit_code)
