# coding_test-video_reordering
Takes as input a video with randomly mixed frames, and additional outlier/noise frames, and outputs a cleaned and reordered video.

## Usage

```
python rearange_mixed_frames.py --video-path ./corrupted_video.mp4 --apply-pca 25
```

or, to output a video in reverse order (if the above command created a video reading backwards through time) :

```
python rearange_mixed_frames.py --video-path ./corrupted_video.mp4 --apply-pca 25 -r
```
