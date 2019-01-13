#!/usr/bin/python3
# USAGE
# python detect_faces_video.py --prototxt deploy.prototxt.txt --model
# res10_300x300_ssd_iter_140000.caffemodel

# import the necessary packages
import numpy as np
import argparse
import time
import cv2
import logging as log
from tqdm import tqdm
from pathlib import Path

from imutils import paths
import face_recognition
import pickle
import dlib


PADDING=0
DEFAULT_CAFFE_PROTO = '../trained_model/deploy.prototxt.txt'
DEFAULT_CAFFE_MODEL = '../trained_model/res10_300x300_ssd_iter_140000.caffemodel'
NUM_JITTERS=5
TOLERANCE=0.7


def main(args):
    net = cv2.dnn.readNetFromCaffe(DEFAULT_CAFFE_PROTO, DEFAULT_CAFFE_MODEL)
    jpg_files = [str(p)
                 for p in Path(args.in_dir).glob(f'**/{args.pattern}*.jpg')]

    # load the known faces and embeddings
    encodings = Path(args.encoding_dir) / 'encodings.pkl'
    log.debug(f'Reading encoding {encodings}')
    data = pickle.loads(encodings.open(mode="rb").read())

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
        blob = cv2.dnn.blobFromImage(
            cv2.resize(
                image, (300, 300)), 1.0,
                (300, 300), (104.0, 177.0, 123.0))

        # pass the blob through the network and obtain the detections and
        # predictions
        net.setInput(blob)
        detections = net.forward()

        boxes = []
        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > 0.50:
                # compute the (x, y)-coordinates of the bounding box for the
                # object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                #  (top, right, bottom, left)
                top = endY
                right = endX
                bottom = startY
                left = startX
                box = (top+PADDING, right+PADDING, bottom-PADDING, left-PADDING)
                boxes.append(box)


        log.debug(f'Faces detected: {len(boxes)}')
        print(boxes)
        # to encodings
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb, boxes,
                                                    num_jitters=NUM_JITTERS)
        log.debug(f'Encodings: {len(encodings)}')

        # initialize the list of names for each face detected
        names = []

        # loop over the facial embeddings
        for encoding in encodings:
            # match face
            matches = face_recognition.compare_faces(
                data["encodings"], encoding, tolerance=TOLERANCE)
            name = "Unknown"

            # check to see if we have found a match
            if True in matches:
                # find the indexes of all matched faces then initialize a
                # dictionary to count the total number of times each face
                # was matched
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}

                # loop over the matched indexes and maintain a count for
                # each recognized face face
                for i in matchedIdxs:
                    name = data["names"][i]
                    counts[name] = counts.get(name, 0) + 1

                # determine the recognized face with the largest number of
                # votes (note: in the event of an unlikely tie Python will
                # select first entry in the dictionary)
                name = max(counts, key=counts.get)

                # update the list of names
            names.append(name)
        log.debug(names)


        # loop over the recognized faces
        for ((top, right, bottom, left), name) in zip(boxes, names):
                # draw the predicted face name on the image
            cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0, 255, 0), 2)

        # show the output image
        cv2.imshow("Output", image)
        key = cv2.waitKey(0) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            cv2.destroyAllWindows()
            return 0 
    return 0


# construct the argument parse and parse the arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', dest="verbose", action='store_true')
    parser.add_argument('-t', '--tolerance', dest="tolerance", type=float,
                        default=0.6)
    parser.add_argument('-fp', '--pattern', dest="pattern", default='')
    parser.add_argument("-e", "--encodings", dest='encoding_dir',
                        default="../", help="path to serialized db of facial encodings")
    parser.add_argument('-i', '--i', dest='in_dir', default='../frames/',
                        help='input dir')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    if args.verbose:
        log.basicConfig(format="%(levelname)s: %(message)s", level=log.DEBUG)
        print("Verbose output.")
    else:
        log.basicConfig(format="%(levelname)s: %(message)s")

    TOLERANCE = args.tolerance

    # Create out_dir
    # out_dir = Path(args.out_dir)
    # if not out_dir.exists():
    #     out_dir.mkdir()

    exit_code = main(args)

    # Exit
    exit(exit_code)
