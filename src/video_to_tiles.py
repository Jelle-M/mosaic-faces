#!/usr/bin/python3
# Roding: utf-8

""" Extract tiles from video """
import argparse
from pathlib import Path
from tqdm import tqdm
import logging as log
import os

# construct the argument parse and parse the arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', dest="verbose", action='store_true')
    parser.add_argument('-n', '--num_frames', dest="num_frames", default=100)
    parser.add_argument('-f', '--force', dest="force", action='store_true',
                        help='extract frames even if already extracted before')
    parser.add_argument('-e', '--extracted', dest="extracted",
                        default='../extracted.txt')
    parser.add_argument('--videos', dest="video_dir", default='../videos/')
    parser.add_argument('-fr', '--frames', dest="frames", action='store_true')
    parser.add_argument("-p", "--padding", type=int, default=0,
                        help="amount of pixels padding")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    if args.verbose:
        log.basicConfig(format="%(levelname)s: %(message)s", level=log.DEBUG)
        log.info("Verbose output.")
    else:
        log.basicConfig(format="%(levelname)s: %(message)s")

    # Store names of already processed videos
    extracted_file = Path(args.extracted)
    extracted = extracted_file.open().read().split() if\
            extracted_file.exists() else []
    # Extract frames
    video_dir = Path(args.video_dir)
    video_files = [str(v) for v in video_dir.glob('*')]
    for video_file in tqdm(video_files, total=len(video_files), unit="videos"):
        episode = Path(video_file).stem
        # test if in extracted
        if episode in extracted and not args.force:
            continue
        n_frames = args.num_frames
        cmd = f'python3 frames.py {video_file} -n {n_frames}'
        print(cmd)
        os.system(cmd)
        # Add to extracted
        extracted.append(episode)

    # Write to extracted.txt
    extracted_file.open(mode='w').write('\n'.join(extracted))

    if args.frames:
        print('Extraction done!')
        exit(0)

    # Extract faces
    cmd = f'python3 faces.py -p {args.padding}'
    print(cmd)
    os.system(cmd)

    # Generate tiles
    cmd = 'python3 tiles.py'
    print(cmd)
    os.system(cmd)
