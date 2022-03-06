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
import scipy as sp
import argparse
import matplotlib.pyplot as plt

sys.dont_write_bytecode = True

class ARTagDetector():

    def __init__(self, video_path, visualize):
        self.video_path = video_path
        self.visualize = visualize

    def normalize(self, img):
        return np.uint8(cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX))

    def fft(self, img):
        dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
        return magnitude_spectrum

    def detect(self):

        video = cv2.VideoCapture(self.video_path) 
        currentframe = 0
        ret = True

        while(ret): 

            ret, frame = video.read()
            if(not ret):
                break

            frame = cv2.resize(frame, dsize=None, fx=0.4, fy=0.4, interpolation=cv2.INTER_CUBIC)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_fft = self.fft(frame_gray)

            if(self.visualize):
                cv2.imshow("Frame", frame)
                cv2.imshow("Frame Gray", frame_gray)
                cv2.imshow("FFT", self.normalize(frame_fft))
                cv2.waitKey(0)


def main():
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--VideoPath', type=str, default="../Data/1tagvideo.mp4", help='Path to the video file')
    Parser.add_argument('--Visualize', action='store_true', help='Toggle visualization')
    
    Args = Parser.parse_args()
    VideoPath = Args.VideoPath
    Visualize = Args.Visualize

    AR = ARTagDetector(VideoPath, Visualize)
    AR.detect()


if __name__ == '__main__':
    main()