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
from PIL import Image
from skimage.filters import threshold_otsu

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
        r = 100
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

        idx = np.where(img == 255)
        r = list(idx[0])
        c = list(idx[1])

        c = np.array([(c[r.index(min(r))], min(r)),
                       (min(c), r[c.index(min(c))]),
                       (c[r.index(max(r))], max(r)),
                       (max(c), r[c.index(max(c))])])

        # cv2.circle(img, c[0], 3, (255),-1)
        # cv2.circle(img, c[1], 3, (255),-1)
        # cv2.circle(img, c[2], 3, (255),-1)
        # cv2.circle(img, c[3], 3, (255),-1)

        w = abs(c[0][0] - c[3][0])
        h = abs(c[0][1] - c[1][1])

        return c, w, h, img

    def compute_homography(self, w, h, c1):
        c2 = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(4, 2) # Points on Warped Image

        H = cv2.getPerspectiveTransform(np.float32(c1), np.float32(c2))

        return H

    def rotate_image(self, image, angle):
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result

    def detect(self):

        video = cv2.VideoCapture(self.video_path)
        
        currentframe = 0
        ret = True

        if(ret):
            ret, frame = video.read()
            frame = cv2.resize(frame, dsize=None, fx=0.4, fy=0.4, interpolation=cv2.INTER_CUBIC)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            (thresh, frame_gray) = cv2.threshold(frame_gray, 220, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            kernel = np.ones((5,5),np.uint8)
            frame_gray = cv2.morphologyEx(frame_gray, cv2.MORPH_OPEN, kernel)

            dft_shift, frame_fft = self.fft(frame_gray)
            frame_fft_mask, edges = self.high_pass_filter(frame_gray, dft_shift)
            edges = self.normalize(edges)
            edges[edges < 90] = 0
            edges[edges >= 90] = 255

            corners, w, h, frame_ = self.compute_corners(edges)
            mask = cv2.fillPoly(np.copy(frame_), pts = [corners], color =(255,255,255))
            kernel = np.ones((9, 9), np.uint8)
            mask = cv2.erode(mask, kernel) 
            mask = cv2.bitwise_not(mask)

            frame_ = cv2.bitwise_or(frame_gray, mask)
            frame_ = cv2.bitwise_not(frame_)
            corners, w, h, frame_ = self.compute_corners(frame_)
            H = self.compute_homography(w, h, corners)
            frame_ = cv2.warpPerspective(frame, H, (w, h))

            if(self.visualize):
                cv2.imshow("Frame", frame)
                cv2.imshow("Frame_", self.normalize(frame_))
                # cv2.imshow("Frame Gray", np.uint8(frame_gray))
                # cv2.imshow("Mask", np.uint8(mask))
                # cv2.imshow("FFT", self.normalize(frame_fft))
                # cv2.imshow("FFT + Mask", self.normalize(frame_fft_mask))
                # cv2.imshow("Edges", self.normalize(edges))
                # cv2.imshow("Corners", self.normalize(corners))
                cv2.waitKey(0)

        return frame_


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