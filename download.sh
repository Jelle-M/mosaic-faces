#!/bin/bash
# videos.txt with vod links seperated by \n
#   e.g.
#       twitch.../vod/1...
#       twitch.../vod/2...
#       twitch.../vod/3...
# ./download.sh videos.txt prefix
#
FILENAME=$1
FILELINES=`cat $FILENAME`
echo Start

ITER=0
for URL in $FILELINES ; do
    echo $URL
    LINK=`youtube-dl -f 720p -g $URL`
    ffmpeg -i ${LINK} -ss 01:00:00.00 -t 00:15:00.00 -c copy $2$ITER.mp4
    let ITER=$ITER+1
    echo $1$ITER
done


