# Mosaic-faces

Create a photomosaic with faces extracted from video footage.

In this project I'll be using episodes of Critical Role podcast.

![Frame 1 s02e01][cr_frame1]

Scripts were written to extract faces and resized to square tiles.

![Faces s02e01][cr_faces1]
![Tiles s02e01][cr_tiles1] Then used as input for a mosaic [generator][mosaic_project] with a reference
image.

![Reference 1][cr_reference1]
![Result 1][cr_result1]

Recognition based approach

Labelled data for Liam and Sam
![Liam 1][cr_liam1]
![Sam 1][cr_sam1]

![Recognition 1][cr_recognition1]

Other outputs

![Marisha 1][cr_marisha1]
![Travis 1][cr_travis2]
![Travis Large][cr_travis3]


## Todo
- [ ] Docker image
- [x] Add face recognition to label faces
- [x] Find a more accurate recognition method
- [ ] Cleanup/refactor recognition code
- [ ] Setup pipeline, video -> labelled faces 
- [ ] Emotion detection on faces

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

I've listed the required packages in [requirements.txt](requirements.txt)
What things you need to install the software and how to install them


### Installing

Refer to the opencv install guide (google this)

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

## Issues and Challenges
All faces are extracted from video footage. If you only want a select number of subjects, faces
from other people will also be extracted. Added face recognition could solve
this issue.

Colors are limited. When creating mosaic from faces it's difficult to represent
the complete color spectrum. Shifting weights of the color channels of some
faces could help create better looking mosaics.

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
[cr_recognition1]: images/result_recognition1.png "Recognition 1 s02e01"
[cr_liam1]: images/Liam.jpg "Liam 1 s02e01"
[cr_sam1]: images/Sam.jpg "Sam 1 s02e01"
[cr_marisha1]: images/Marisha.gif "Marisha 1"
[cr_travis2]: images/Travis.gif "Travis 1"
[cr_travis3]: images/Travis.jpeg "Travis Large"

