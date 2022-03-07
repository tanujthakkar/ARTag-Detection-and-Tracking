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
        return dft_shift, magnitude_spectrum

    def high_pass_filter(self, img, dft_shift):

        rows, cols = img.shape
        crow, ccol = int(rows / 2), int(cols / 2)  # center

        mask = np.ones((rows, cols, 2), np.uint8)
        r = 200
        center = [crow, ccol]
        x, y = np.ogrid[:rows, :cols]
        mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
        mask[mask_area] = 0

        # apply mask and inverse DFT
        fshift = dft_shift * mask

        fshift_mask_mag = 2000 * np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1]))

        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

        return fshift_mask_mag, img_back

    def compute_corners(self, img):

        img[img < 225] = 0
        img[img >= 225] = 255
        idx = np.where(img == 255)
        r = list(idx[0])
        c = list(idx[1])

        c = np.array([(c[r.index(min(r))], min(r)),
                       (min(c), r[c.index(min(c))]),
                       (c[r.index(max(r))], max(r)),
                       (max(c), r[c.index(max(c))])])

        cv2.circle(img, c[0], 2, (0,0,255),-1)
        cv2.circle(img, c[1], 2, (0,0,255),-1)
        cv2.circle(img, c[2], 2, (0,0,255),-1)
        cv2.circle(img, c[3], 2, (0,0,255),-1)

        w = abs(corners[0][0] - corners[3][0])
        h = abs(corners[0][1] - corners[1][1])

        return c, w, h, img

    def detect(self):

        video = cv2.VideoCapture(self.video_path)
        ret, frame = video.read()
        currentframe = 0
        ret = True

        if(ret):
            frame = cv2.resize(frame, dsize=None, fx=0.4, fy=0.4, interpolation=cv2.INTER_CUBIC)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            dft_shift, frame_fft = self.fft(frame_gray)
            frame_fft_mask, edges = self.high_pass_filter(frame_gray, dft_shift)
            corners, frame = self.get_corners(frame)

            c = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(4, 2) # Points on Warped Image

            H = cv2.getPerspectiveTransform(np.float32(corners), np.float32(c))
            print(H)

            frame_ = cv2.warpPerspective(frame, H, (w, h))


            if(self.visualize):
                cv2.imshow("Frame", frame)
                cv2.imshow("Frame_", frame_)
                # cv2.imshow("Frame Gray", frame_gray)
                # cv2.imshow("FFT", self.normalize(frame_fft))
                # cv2.imshow("FFT + Mask", self.normalize(frame_fft_mask))
                # cv2.imshow("Edges", self.normalize(edges))
                # cv2.imshow("Corners", self.normalize(corners))
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