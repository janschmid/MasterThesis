# FFMPEG
Dont use pattern_type glob, it's fucking it up, create a file list first and take it as input to ffmpeg to ensure that it's in sync!!!
- Create file list
    Run `create_ffmpeg_file_process_list` on image data dir
- Let's create the h246 video
     `/usr/bin/ffmpeg -f concat -safe 0 -i HelperFunctions/ordered_file_list_ffmpeg.csv  -colorspace 1 -codec h264 images.h264`
- Make sure the entire video has the same resolution, if not deepstream will skip frames without complaining and the entire index is srewed
    `/usr/bin/ffmpeg -i images.h264 -vf scale=1280:720 rimages.h264`


- Create preview from h264
    `/usr/bin/ffmpeg -i images.h264 -c copy -copyts -t 10  output.mp4`

# Useful commands
- copy subset of directory :
    `ls  | grep "IMG_0001.*" | xargs -d "\n" cp -t   full_images_movie/`