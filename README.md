# Mosaic-faces

Create a photomosaic with faces extracted from video footage.

In this project I'll be using episodes of Critical Role podcast.

![Frame 1 s02e01][cr_frame1]

Scripts were written to extract square tiles of faces and then used as input
for a mosaic generator.

![][cr_faces1]
.caption[*Faces from s02e01*]

![][cr_tiles1]
.caption[
**Fig. 1:** Image caption 
]

Run [mosaic project][mosaic_project] with tiles and reference image

![Result 1][cr_result1]

![Reference 1][cr_reference1]

## Getting Started

To get started you need to setup your environment.
This project is developed on Ubuntu 18.04 LTS with requirements installed
listed in Prerequisites.


## Todo
- [ ] Docker image
- [ ] Write docs
- [ ] Add face recognition to label faces


## Video footage

Video footage was obtained using the [youtube-dl][youtube-dl_project] CLI utility. Simply pasting the playlist url into `youtube-dl playlist-url` downloads all episodes. Use the `--playlist-start` and `--playlist-end` arguments to select which episodes you'd like as each video has a size greater than 1GB.

The episodes are named similar to `Curious Beginnings | Critical Role | Campaign 2,
Episode 1` with the episode number mentioned. I used `src/rename.py` to parse
the episode numbers and rename the videofiles to make further processing easier.


### Prerequisites

I've listed the required packages in [requirements.txt](requirements.txt)
What things you need to install the software and how to install them


### Installing

Refer to the opencv install guide (google this)

To install packages with pip3
`pip3 install -r requirements.txt`


## Deployment

The following are run inside `src/` unless specified otherwise.
The following scripts should be run in this order
```
1. youtube-dl playlist from videos/ 
2. rename.py
3. frames.py
4. faces.py
5. tiles.py
6. mosaic.py from [mosaic project][mosaic_project]
```

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

Create the tiles
```
python3 video_to_tiles.py 
```

Create mosaic from [mosaic project][mosaic_project]
```
python mosaic.py reference tiles/
```

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
