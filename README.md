
[![Build Status](https://travis-ci.org/MoskiMBA/mosaic-faces.svg?branch=master)](https://travis-ci.org/MoskiMBA/mosaic-faces)
# Mosaic-faces

Create a photomosaic with faces extracted from video footage.

In this project I'll be using episodes of Critical Role podcast.

![Frame 1 s02e01][cr_frame1] 
Scripts were written to extract faces and resized to square tiles.

![Faces s02e01][cr_faces1]
![Tiles s02e01][cr_tiles1]

Then used as input for a mosaic [generator][mosaic_project] with a reference image.  

![Reference 1][cr_reference1]
![Result 1][cr_result1]

## Results with recognition based approach

Output DS after training with <30 face images per person.
Output for a single frame  
![Recognition 1][cr_recognition1]

Extracted faces Travis and Marisha
![Travis 1][cr_travis1]
![Marisha 2][cr_marisha2]

Difference detection and recogniton  
![Detection][cr_detection2]
![Recognition][cr_recognition2]

Some gifs of tiles  
![Marisha 1][cr_marisha1] ![Travis 1][cr_travis2]

Mosaic Travis  
![Travis Source][cr_Travissource1]
![Travis Large][cr_travis3]

Mosaic Marisha (Keyleth)  
![Keyleth Source][cr_Keylethsource1]
![Keyleth Large][cr_keyleth1]

## Smile detector
With a lot of data, we extract a lot of faces. With a 'smile detector' I tried
to filter out the faces that show the most emotion. 

Faces after filtering with a smile detector  

![Laura smile][cr_smile1]




## Todo
- [x] Add face recognition to label faces
- [x] Find a more accurate recognition method
- [x] Cleanup/refactor recognition code
- [x] Emotion detection on faces
- [ ] Small fixes (.jpg.jpg files, label names as args)
- [ ] ~~Docker image~~
- [ ] ~~Setup pipeline, video -> labelled faces~~

## Chart
<pre>
download.sh
 +------------+      +------------+
 +videos.txt  |      +videos/     |
 | list of vod+------> video files|
 | URLs       |      |            |
 +------------+      +------------+

video_to_tiles.py
 +----------------------------------------------------+
 |                        +------------+  +---------+ |
 |            +---------+ |faces.py    |  |tiles.py | |
 | +-------+  |frames.py| | -SSD (fast)|  | reshape | |
 | | video +--> frames/ +-> -HoG (slow)+-->  tiles/ | |
 | +-------+  +---------+ |  faces/    |  +---------+ |
 |                        +------------+              |
 +----------------------------------------------------+

with recognition                      +------------+
 +--------------+   +-----------+     |recognize.py|
 |label (manual)|   |train.py   +-----> faces/     |
 | faces/       +---> ds/       |     |  ds_new    |
 |  ds/         |   |  model.yml+--+  +------------+
 +--------------+   +-----------+  |  +------------------+
                                   +-->recognize_frame.py|
mosaic                                | frames/          |
 +--------------+                     |  ds_new          |
 |mosaic.py     |                     +------------------+
 | reference img|
 | tiles/       |
 |  mosaic.jpeg |
 +--------------+
</pre>


## Getting Started

To get started you need to setup your environment.
This project is developed on Ubuntu 18.04 LTS with requirements installed
listed in Prerequisites.

### Prerequisites

I've listed the required python packages in [requirements.txt](requirements.txt). You'll also need OpenCV. I used [3.4.4](https://docs.opencv.org/3.4.4/d2/de6/tutorial_py_setup_in_ubuntu.html).


### Installing
Refer to the opencv [3.4.4](https://docs.opencv.org/3.4.4/d2/de6/tutorial_py_setup_in_ubuntu.html)
 install guide.

To install packages with pip3
`pip3 install -r requirements.txt`


## Video footage

Video footage was obtained using the [youtube-dl][youtube-dl_project] CLI utility. Simply pasting the playlist url into `youtube-dl playlist-url` downloads all episodes. Use the `--playlist-start` and `--playlist-end` arguments to select which episodes you'd like as each video has a size greater than 1GB.

The episodes are named similar to `Curious Beginnings | Critical Role | Campaign 2,
Episode 1` with the episode number mentioned. I used `src/rename.py` to parse
the episode numbers and rename the videofiles to make further processing easier.


## Deployment

The following are run inside `src/` unless specified otherwise.
The following scripts should be run in this order

1. `rename.py`
1. `frames.py`
1. `faces.py`
1. `tiles.py`
1. `mosaic.py` from [mosaic project][mosaic_project]

To avoid running these scripts manually for each video theres `
video_to_tiles.py` that call steps 3-5. 

Clean output directories, `f` option also removes `frames/` folder, this is
usally kept static and only needs to run once.
```
sh clean.sh f
```

Extract frames from videos in `videos/`. Once a video is processed it is added
to `extracted.txt` and will no longer be run again unless specified by `-f`
```
python3 video_to_tiles.py -fr -n 100 #Extract 100 frames of videos
```

Extract faces and generate tiles
```
python3 video_to_tiles.py 
```

Create mosaic from [mosaic project][mosaic_project]
```
python mosaic.py reference tiles/
```

### Recognition
First we must train a model. In `ds/` I made directories so each person has about
30 faces (extracted from faces.py).

<pre>
ds/
├── Travis/
│   ├── 0.jpg
│   ├── 1.jpg
│   ├── ...
│   └── 29.jpg
├── Marisha/
│   ├── ...
│   └── 28.jpg
└── ...
</pre>

Then I run `train.py` to create a trained model. Then extract more frames from
videos and this time run `recognize_frame.py` to create a ds_new folder with
labelled faces.

If all went well, in `ds_new/` there will be a structure similar to `ds/` with each
folder containing only faces of the respective person.

## Issues and Challenges
All faces are extracted from video footage. If you only want a select number of subjects, faces
from other people will also be extracted. Added face recognition could solve
this issue.

Colors are limited. When creating mosaic from faces it's difficult to represent
the complete color spectrum. Shifting weights of the color channels of some
faces could help create better looking mosaics.

The recognition algorithm isn't 100% correct and mislabelled faces do happen.
Luckily these mistakes are limited and can be corrected manually quite easily.

## Face recognition
After experimenting with face detection + recognition combos the best one seems
to be the HoG face detection to extract faces and label them. Then using the
same extraction method and use a .LBPHFaceRecognizer_create

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* [Critical Role][critical_role_url]
* [Mosaic project][mosaic_project]
* [youtube-dl][youtube-dl_project] 

<!-- Links to image -->
[critical_role_url]: https://critrole.com/
[youtube-dl_project]: https://github.com/rg3/youtube-dl
[mosaic_project]: https://github.com/codebox/mosaic
[cr_frame1]: images/frame1.jpg "Frame 1 s02e01"
[cr_reference1]: images/reference1.jpg "Reference 1"
[cr_result1]: images/result1.jpg "Result 1"
[cr_faces1]: images/faces_results/faces1.jpg "Faces 1 s02e01"
[cr_tiles1]: images/tiles_results/tiles1.jpg "Tiles 1 s02e01"
[cr_recognition1]: images/result_recognition.png "Recognition 1 s02e01"
[cr_marisha2]: images/Marisha.jpg "Marisha"
[cr_travis1]: images/Travis.jpg "Travis"
[cr_marisha1]: images/Marisha.gif "Marisha 1"
[cr_travis2]: images/Travis.gif "Travis 1"
[cr_travis3]: images/Travis.jpeg "Travis Large"
[cr_keyleth1]: images/Keyleth.jpeg "Keyleth Large"
[cr_Travissource1]: images/Travis_source.bmp "Travis source"
[cr_Keylethsource1]: images/Keyleth_source.jpg "Keyleth source"
[cr_smile1]: images/smile_laura.jpg "Laura smile"
[cr_detection2]: images/DETECTION.jpg "detection"
[cr_recognition2]: images/RECOGNITION.jpg "recognition"

