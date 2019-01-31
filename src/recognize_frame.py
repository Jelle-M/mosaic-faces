#!/usr/bin/python3
# USAGE
# python detect_faces_video.py --prototxt deploy.prototxt.txt --model
# res10_300x300_ssd_iter_140000.caffemodel

# import the necessary packages
import argparse
import logging as log
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def write(image, out_dir, episode, index):
    out_dir = Path(out_dir / episode)
    if not out_dir.exists():
        out_dir.mkdir()
    frame_name = "{0}.jpg".format(index)
    cv2.imwrite(str(out_dir / frame_name), image)


PADDING = 0
DEFAULT_CAFFE_PROTO = '../trained_model/deploy.prototxt.txt'
DEFAULT_CAFFE_MODEL = ('../trained_model/res10_300x300_ssd_ite'
                       'r_140000.caffemodel')
NUM_JITTERS = 5
TOLERANCE = 0.7


def main(args):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(f'../trained_model/{args.trainer}')
    recognizer_names = ['Ashley', 'Laura', 'Liam', 'Marisha', 'Matthew', 'Sam',
                        'Talisien', 'Travis']
    net = cv2.dnn.readNetFromCaffe(DEFAULT_CAFFE_PROTO, DEFAULT_CAFFE_MODEL)
    jpg_files = [str(p)
                 for p in Path(args.in_dir).glob(f'**/*{args.pattern}*/*.jpg')]

    for jpg_file in tqdm(jpg_files, total=len(jpg_files), unit="images"):
        log.debug(f'Processing image {jpg_file}')
        jpg_file_path = Path(jpg_file)
        frame_name = jpg_file_path.name
        name = jpg_file_path.parents[0].stem
        # face_name = jpg_file_path.stem

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
        boxes, names = [], []
        # Convert to grayscale to recognize
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > 0.20:
                # compute the (x, y)-coordinates of the bounding box for the
                # object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                #  (top, right, bottom, left)
                top = endY
                right = endX
                bottom = startY
                left = startX
                boxes.append(box)

                # label
                try:
                    name_id, confidence = recognizer.predict(
                        gray[startY:endY, startX:endX])
                except BaseException:
                    continue
                # If confidence is less them 100 ==> "0" : perfect match
                if confidence < 72:
                    confidence = "{0}%".format(round(100 - confidence))
                    names.append(recognizer_names[name_id] + confidence)
                    write(image[startY:endY, startX:endX],
                          Path(f'../ds_new'), recognizer_names[name_id],
                          frame_name)
                else:
                    name_id = "unknown"
                    confidence = "{0}%".format(round(100 - confidence))
                    names.append(name_id + confidence)

        log.debug(f'Faces detected: {len(boxes)}')

        # loop over the recognized faces
        for ((left, bottom, right, top), name) in zip(boxes, names):
            # draw the predicted face name on the image
            if all(n == "unkown" for n in names):
                continue
            cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
            y = bottom - 15 if bottom - 15 > 15 else bottom + 15
            cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0, 255, 0), 2)

        if args.verbose:
            # show the output image
            cv2.imshow("Output", image)
            # if the `q` key was pressed, break from the loop
            if (cv2.waitKey(0) & 0xFF) == ord("q"):
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
    parser.add_argument('-i', '--i', dest='in_dir', default='../frames/',
                        help='input dir')
    parser.add_argument('--trainer', dest='trainer',
                        default='trainer.yml')
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
