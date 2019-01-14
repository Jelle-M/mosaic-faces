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

def to_id(name):
    names = ['Ashley', 'Laura', 'Liam', 'Marisha', 'Matthew', 'Sam', 'Talisien', 'Travis']
    return int(names.index(name))

def main(args):
    jpg_files = [str(p) for p in Path(args.in_dir).glob(f'**/{args.pattern}*.jpg')]
    known_images = []
    known_ids = []
    for jpg_file in tqdm(jpg_files, total=len(jpg_files), unit="images"):
        log.debug(f'Processing image {jpg_file}')
        jpg_file_path = Path(jpg_file)
        frame_name = jpg_file_path.name
        log.debug(f'frame_name {frame_name}')
        name = jpg_file_path.parents[0].stem
        log.debug(f'name {name}')
        face_name = jpg_file_path.stem
        log.debug(f'face_name {face_name}')
        image = cv2.imread(jpg_file)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray = np.array(gray,'uint8')
        known_ids.append(to_id(name))
        known_images.append(gray)
    known_ids = np.array(known_ids)

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(known_images, known_ids)

    # Save the model into trainer/trainer.yml
    trainer_file = '../trained_model/trainer.yml'
    recognizer.write(trainer_file)
    print(f'Written trainer to {trainer_file}')
    return 0


# construct the argument parse and parse the arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', dest="verbose", action='store_true')
    parser.add_argument('-fp', '--pattern', dest="pattern", default='')
    parser.add_argument("-i", "--dataset", dest="in_dir", default="../ds",
                        help="path to input directory of faces + images")
    parser.add_argument("-e", "--ids", dest='out_dir',
                        default="../", help="path to serialized db of facial ids")
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
