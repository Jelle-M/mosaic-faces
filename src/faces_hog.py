#!/usr/bin/python3
""" Extracts faces using HoG method """
# USAGE
# python detect_faces_video.py --prototxt deploy.prototxt.txt --model
# res10_300x300_ssd_iter_140000.caffemodel

# import the necessary packages
import argparse
import logging as log
from pathlib import Path

import cv2
from tqdm import tqdm
import face_recognition  # pylint: disable=E0401


def write(image, write_dir, episode, index):
    write_dir = Path(write_dir / episode)
    if not write_dir.exists():
        write_dir.mkdir()
    frame_name = "{0}.jpg".format(index)
    cv2.imwrite(str(write_dir / frame_name), image)


def extract_faces(in_dir, out_dir, pattern='', verbose=False, padding=0):
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

        # detect the (x, y)-coordinates of the bounding boxes corresponding
        # to each face in the input image, then compute the facial embeddings
        # for each face
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb, model='hog')
        # tolerance=0.6

        for box in boxes:
            # compute the (x, y)-coordinates of the bounding box for the
            # object
            startY, endX, endY, startX = box

            # Add padding with out of bounds protection
            startX = max(0, startX - padding)
            startY = max(0, startY - padding)
            endX = min(endX + padding, w)
            endY = min(endY + padding, h)

            # write face to face dir
            face = image[startY:endY, startX:endX]
            write(face, Path(out_dir), episode, index)
            index += 1

            # draw the bounding box of the face along with the associated
            # probability
            text = ""
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
    parser.add_argument("-p", "--padding", type=int, default=0,
                        help="amount of pixels padding")
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
        args.in_dir, args.out_dir, pattern=args.pattern,
        verbose=args.verbose, padding=args.padding)

    # Exit
    exit(exit_code)
