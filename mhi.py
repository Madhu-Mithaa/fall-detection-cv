#!/usr/bin/env python
import os
import numpy as np
import cv2

MHI_DURATION = 30
DEFAULT_THRESHOLD = 32
def get_motion_history_images(video_src, output_folder):
    video_name = video_src.split(".mp4")[0]
    cam = cv2.VideoCapture(video_src)
    ret, frame = cam.read()
    h, w = frame.shape[:2]
    prev_frame = frame.copy()
    motion_history = np.zeros((h, w), np.float32)
    timestamp = 0
    count = 0
    while True:
        ret, frame = cam.read()
        if not ret:
            break
        frame_diff = cv2.absdiff(frame, prev_frame)
        gray_diff = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY)
        ret, fgmask = cv2.threshold(gray_diff, DEFAULT_THRESHOLD, 1, cv2.THRESH_BINARY)
        timestamp += 1
        # update motion history
        cv2.motempl.updateMotionHistory(fgmask, motion_history, timestamp, MHI_DURATION)
        # normalize motion history
        mh = np.uint8(np.clip((motion_history-(timestamp-MHI_DURATION)) / MHI_DURATION, 0, 1)*255)
        if count>MHI_DURATION:
            cv2.imwrite(os.path.join(output_folder,'{}_mhi_{}.jpg'.format(video_name, count)),mh)
        prev_frame = frame.copy()
        count += 1
    cv2.destroyAllWindows()

if __name__ == "__main__":
  #video_src = 'actor4_fall.mp4'
  input_folder = 'C:/Users/DELL/Desktop/test'
  #input_folder = 'C:/Users/DELL/Desktop/FallDatasetProcessed/not_fall/'
  out_folder = 'C:/Users/DELL/Desktop/FallDatasetProcessed/fall/'
  for video_src in os.listdir(input_folder):
    vid_path = os.path.join(input_folder, video_src)
    get_motion_history_images(vid_path, out_folder)
