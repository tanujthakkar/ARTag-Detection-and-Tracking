#!/usr/env/bin python3

"""
ENPM673 Spring 2021: Perception for Autonomous Robots

Project 1 - 

Author(s):
Tanuj Thakkar (tanuj@umd.edu)
M. Engg Robotics
University of Maryland, College Park
"""

# Importing packages
import sys
import cv2
import os
import numpy as np
import argparse
import matplotlib.pyplot as plt


class ARTagDetector():

    def __init__(self, video_path):
        self.video_path = video_path

    def detect(self):

        video = cv2.VideoCapture(self.video_path) 
        currentframe = 0
        ret = True

        while(ret): 

            ret, frame = video.read()
            if(not ret):
                break

            frame = cv2.resize(frame, dsize=None, fx=0.4, fy=0.4, interpolation=cv2.INTER_CUBIC)


            cv2.imshow("Frame", frame)
            cv2.waitKey(0)


def main():
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--VideoPath', type=str, default="../Data/1tagvideo.mp4", help='Path to the video file')
    
    Args = Parser.parse_args()
    VideoPath = Args.VideoPath

    AR = ARTagDetector(VideoPath)
    AR.detect()

if __name__ == '__main__':
    main()