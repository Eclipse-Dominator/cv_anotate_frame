import cv2
import os
import sys
import argparse
import asyncio
import numpy as np
from AnnotateFrame import AnnotateFrame
from collections import defaultdict

my_parser = argparse.ArgumentParser(description='Opens a video with a specified path and allows for annotation of up to 4 points for each frame. Press "q" to exit the program, press any other key to skip the current frame.')

my_parser.add_argument('--video', action='store', type=str, required=True) # Video path

my_parser.add_argument('--frame', action='store', type=int, required=False) # Skip to this frame

my_parser.add_argument('--filename', action='store', type=str, required=False) # Specify coords text file name

args = my_parser.parse_args()

video_path = args.video
framecounter = args.frame if args.frame else 0
savefile = args.filename if args.filename else "Annotations.txt" # Default text file name is "Annotations.txt"

if not os.path.exists(video_path):
    print('The path specified does not exist') # No video error
    sys.exit()


framePointsData = defaultdict(lambda: [])

def getFramesFromVideo(video): # Split video into frames
    vidcap = cv2.VideoCapture(video)
    success = True
    frames = []
    while success:
      success,image = vidcap.read()
      if success:
          frames.append(image)
    vidcap.release()
    return frames

async def annotateFrame(frame,frame_number,IDENTIFIER): # Annotate one frame
    af = AnnotateFrame(IDENTIFIER,40,frame)
    af.frameData = framePointsData[frame_number] 
    af.setPointLimit(4,True) 
    cv2.setMouseCallback(IDENTIFIER, af.mouseCallback,[])
    
    while 1:
        k = cv2.waitKeyEx(0)
        if k==2424832: # left arrow == go back to previous frame
            framePointsData[frame_number] = af.frameData
            return frame_number - 1 if frame_number>0 else 0
        elif k==2555904: # right arrow == go to next frame:
            framePointsData[frame_number] = af.frameData
            return frame_number+1
        elif k==99: # c == change color
            af.changeLineColor()
        elif k==8: # backspace == delete last shape
            af.undoShape()
        elif k==27: # Escape == close the program
            return -1

async def annotationProgram(video,startFrame): # Overall program
    frames = getFramesFromVideo(video)
    i = startFrame
    IDENTIFIER = ''
    cv2.namedWindow(IDENTIFIER)
    
    while i>-1 and i < len(frames):
        cv2.setWindowTitle(IDENTIFIER,f"frame_{i}")
        i = await annotateFrame(frames[i],i,IDENTIFIER)
    cv2.destroyAllWindows()

asyncio.run(annotationProgram(video_path,framecounter))