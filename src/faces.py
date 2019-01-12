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

DEFAULT_CAFFE_PROTO = '../trained_model/deploy.prototxt.txt'
DEFAULT_CAFFE_MODEL = '../trained_model/res10_300x300_ssd_iter_140000.caffemodel'


def write(image, out_dir, episode, index):
    out_dir = Path(out_dir / episode)
    if not out_dir.exists():
        out_dir.mkdir()
    frame_name = "{0}.jpg".format(index)
    cv2.imwrite(str(out_dir / frame_name), image)


def extract_faces(in_dir, out_dir, confidence_treshold=0.23, pattern='', verbose=False,
                  model=DEFAULT_CAFFE_MODEL, prototxt=DEFAULT_CAFFE_PROTO):
    # load our serialized model from disk
    net = cv2.dnn.readNetFromCaffe(prototxt, model)

    # loop over the frames from the input images
    # ** means recursive
    jpg_files = [str(p) for p in Path(in_dir).glob(f'**/{pattern}*.jpg')]
    index = 0
    for jpg_file in tqdm(jpg_files, total=len(jpg_files), unit="images"):
        # load the input image and construct an input blob for the image
        # by resizing to a fixed 300x300 pixels and then normalizing it
        log.debug(jpg_file)
        jpg_file_path = Path(jpg_file)
        frame_name = jpg_file_path.name
        episode = jpg_file_path.parents[0].stem
        log.debug(frame_name)
        face_name = jpg_file_path.stem
        log.debug(face_name)
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
        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > confidence_treshold:
                # compute the (x, y)-coordinates of the bounding box for the
                # object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # write face to face dir
                face = image[startY:endY, startX:endX]
                write(face, Path(out_dir), episode, index)
                index += 1

                # draw the bounding box of the face along with the associated
                # probability
                text = "{:.2f}%".format(confidence * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(image, (startX, startY), (endX, endY),
                              (0, 0, 255), 2)
                cv2.putText(image, text, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

        # show the output image
        if verbose:
            cv2.imshow("Output", image)
            key = cv2.waitKey(0) & 0xFF
            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break
    # do a bit of cleanup
    cv2.destroyAllWindows()
    return 0


# construct the argument parse and parse the arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--prototxt",
                        help="path to Caffe 'deploy' prototxt file",
                        default=DEFAULT_CAFFE_PROTO)
    parser.add_argument("-m", "--model",
                        help="path to Caffe pre-trained model",
                        default=DEFAULT_CAFFE_MODEL)
    parser.add_argument("-c", "--confidence", type=float, default=0.23,
                        help="minimum probability to filter weak detections")
    parser.add_argument('-o', '--out', dest='out_dir', default='../faces/',
                        help='output dir')
    parser.add_argument('-i', '--i', dest='in_dir', default='../frames/',
                        help='input dir')
    parser.add_argument('-v', '--verbose', dest="verbose", action='store_true')
    parser.add_argument('-fp', '--pattern', dest="pattern", default='')
    parser.add_argument('-I', dest="single_image", action='store_true')
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

    # Extract Faces
    exit_code = extract_faces(
        args.in_dir, args.out_dir, confidence_treshold=args.confidence,
        pattern=args.pattern, verbose=args.verbose,
        model=DEFAULT_CAFFE_MODEL, prototxt=DEFAULT_CAFFE_PROTO)

    # Exit
    exit(exit_code)