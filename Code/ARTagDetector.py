#!/usr/env/bin python3

"""
ENPM673 Spring 2021: Perception for Autonomous Robots

Project 1 - April Tag Detection & Tracking

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

from utils import *

sys.dont_write_bytecode = True

class ARTag():

    def __init__(self):
        pass

    def fft(self, img):
        # Reference - https://akshaysin.github.io/fourier_transform.html

        dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))

        return dft_shift, magnitude_spectrum

    def high_pass_filter(self, img, dft_shift):
        # Reference - https://akshaysin.github.io/fourier_transform.html

        rows, cols = img.shape
        crow, ccol = int(rows / 2), int(cols / 2)

        mask = np.ones((rows, cols, 2), np.uint8)
        r = 100
        center = [crow, ccol]
        x, y = np.ogrid[:rows, :cols]
        mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
        mask[mask_area] = 0

        # apply mask and inverse DFT
        fshift = dft_shift * mask

        # fshift_mask_mag = 2000 * np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1]))
        fshift_mask_mag = None

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

        img = np.dstack((img, img, img))
        cv2.circle(img, c[0], 3, (0, 0, 255),-1)
        cv2.circle(img, c[1], 3, (0, 0, 255),-1)
        cv2.circle(img, c[2], 3, (0, 0, 255),-1)
        cv2.circle(img, c[3], 3, (0, 0, 255),-1)

        w = abs(c[0][0] - c[3][0])
        h = abs(c[0][1] - c[1][1])

        return c, w, h, img

    def compute_homography(self, w, h, c1):
        c2 = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(4, 2) # Points on Warped Image
        H = homography(np.float32(c1), np.float32(c2))

        return H

    def detect(self, frame, visualize):

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        (thresh, frame_gray) = cv2.threshold(frame_gray, 220, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        kernel = np.ones((5,5),np.uint8)
        frame_gray = cv2.morphologyEx(frame_gray, cv2.MORPH_OPEN, kernel)

        dft_shift, frame_fft = self.fft(frame_gray)
        frame_fft_mask, edges = self.high_pass_filter(frame_gray, dft_shift)
        edges = normalize(edges)
        edges[edges < 90] = 0
        edges[edges >= 90] = 255

        corners, w, h, frame_ = self.compute_corners(edges)
        mask = cv2.fillPoly(np.copy(frame_gray), pts = [corners], color =(255,255,255))
        kernel = np.ones((9, 9), np.uint8)
        mask = cv2.erode(mask, kernel) 
        mask = cv2.bitwise_not(mask)

        frame_ = cv2.bitwise_or(frame_gray, mask)
        frame_ = cv2.bitwise_not(frame_)
        corners, w, h, frame_ = self.compute_corners(frame_)
        H = self.compute_homography(w, h, corners)
        ar_tag = warp_perspective(frame_gray, H, w, h)
        ar_tag = np.flip(ar_tag, axis=1)

        cv2.polylines(frame, [corners], True, (0,0,255))

        if(visualize):
            cv2.imshow("Frame", frame)
            cv2.imshow("AR Tag", normalize(ar_tag))
            cv2.waitKey(0)

        return frame, ar_tag, corners, H

    def decode(self, ar_tag, visualize):

        ar_tag = remove_padding(ar_tag)
        h, w = ar_tag.shape[:2]
        
        rotate = 0
        for i in range(4):
            blocks = []
            for c in range(0, h, 32):
                for r in range(0, w, 32):
                    blocks.append(np.median(ar_tag[c:c+32, r:r+32]))
            if blocks[-1] >= 200:
                break
            else:
                ar_tag = cv2.rotate(ar_tag, cv2.cv2.ROTATE_90_CLOCKWISE)
            rotate += 1

        # rotate = rotate * 90

        blocks = np.array(blocks).astype(np.int32)
        blocks[blocks < 255] = 0
        blocks[blocks >= 255] = 1

        tag_value = ((blocks[5] * 1) + (blocks[6] * 2) + (blocks[10] * 4) + (blocks[9] * 8))

        ar_tag = draw_grid(ar_tag, 5)

        if(visualize):
            cv2.imshow("AR Tag Decoded", ar_tag)
            cv2.waitKey(0)

        return rotate, tag_value, ar_tag

    def superimpose(self, frame, testudo_img, ar_tag_corners, rotation, visualize):

        testudo_img = cv2.imread(testudo_img)
        testudo_img = rotate_img(testudo_img, rotation)

        h, w = testudo_img.shape[:2]
        H = self.compute_homography(w, h, ar_tag_corners)
        testudo_img = warp_perspective_(testudo_img, np.linalg.inv(H), frame.shape[1], frame.shape[0])

        _, warpedTestudo_mask = cv2.threshold(cv2.cvtColor(testudo_img, cv2.COLOR_BGR2GRAY), 20, 255, cv2.THRESH_BINARY_INV)
        warpedTestudo_mask = np.dstack((warpedTestudo_mask,warpedTestudo_mask,warpedTestudo_mask))
        testudo_img = cv2.bitwise_or(testudo_img, warpedTestudo_mask)

        _, ar_tag_mask = cv2.threshold(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 220, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((9, 9), np.uint8)
        ar_tag_mask = cv2.erode(ar_tag_mask, kernel)
        ar_tag_mask = cv2.fillPoly(np.copy(ar_tag_mask), pts = [ar_tag_corners], color=(125))
        ar_tag_mask[ar_tag_mask != 125] = 0
        ar_tag_mask[ar_tag_mask == 125] = 255
        ar_tag_mask = np.dstack((ar_tag_mask,ar_tag_mask,ar_tag_mask))
        ar_tag_mask = cv2.bitwise_and(testudo_img, ar_tag_mask)

        frame_mask = cv2.fillPoly(np.copy(frame), pts = [ar_tag_corners], color=(0))

        superimposed_img = cv2.bitwise_or(frame_mask, ar_tag_mask)

        if(visualize):
            cv2.imshow("Superimpose", superimposed_img)
            # cv2.imshow("AR Tag Mask", np.uint8(ar_tag_mask))
            cv2.waitKey(0)

        return superimposed_img

    def project(self, frame, ar_tag_corners, cube_size, visualize):

        H = self.compute_homography(cube_size, cube_size, ar_tag_corners)

        K = np.array([[1346.100595, 0, 932.1633975],
                      [0, 1355.933136, 654.8986796],
                      [0, 0, 1]])

        P = projection_matrix(np.linalg.pinv(H), K)
        
        c = compute_cube_corners(P, cube_size)
        cube = draw_cube(frame, c)

        if(visualize):
            cv2.imshow("Projection", normalize(cube))
            cv2.waitKey(0)

        return cube

    def process_frame(self, video_path, action, visualize):
        video = cv2.VideoCapture(video_path)
        ret = True

        if(ret):

            try:
                ret, frame = video.read()
                frame = cv2.resize(frame, dsize=None, fx=0.4, fy=0.4, interpolation=cv2.INTER_CUBIC)

                frame, ar_tag, corners, H = self.detect(frame, False)
                if(action == 'Decode'):
                    rotation, value, ar_tag_decoded = self.decode(ar_tag, False)
                    print("The value of the AR-Tag is {}".format(value))
                    cv2.imshow("AR Tag Decoded", normalize(ar_tag_decoded))

                cv2.imshow("Frame", frame)
                cv2.imshow("AR Tag", normalize(ar_tag))
                cv2.waitKey(0)

            except Exception as e:
                print(e)
                pass


    def process_video(self, video_path, testudo_path, action, save_path, visualize):

        video = cv2.VideoCapture(video_path)
        ret = True
        ret, frame = video.read()
        frame = cv2.resize(frame, dsize=None, fx=0.4, fy=0.4, interpolation=cv2.INTER_CUBIC)

        video_writer = cv2.VideoWriter(os.path.join(save_path, action +'.avi'), cv2.VideoWriter_fourcc('F','M','P','4'), 24, (frame.shape[1], frame.shape[0]))

        while(ret):

            try:
                ret, frame = video.read()
                frame = cv2.resize(frame, dsize=None, fx=0.4, fy=0.4, interpolation=cv2.INTER_CUBIC)

                frame, ar_tag, corners, H = self.detect(frame, False)
                if(action == 'Project'):
                    cube = self.project(frame, corners, 128, False)
                    cv2.imshow("Cube", cube)
                    video_writer.write(np.uint8(cube))
                else:
                    rotation, value, ar_tag = self.decode(ar_tag, False)
                    superimposed = self.superimpose(frame, testudo_path, corners, rotation, False)
                    cv2.imshow("Superimpose", superimposed)
                    video_writer.write(np.uint8(superimposed))

                cv2.waitKey(1)
            except Exception as e:
                print(e)
                pass

        cv2.destroyAllWindows()
        video_writer.release()


def main():
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--Action', type=str, default="detect", help='Select action to perform from [Detect, Decode, Superimpose, Project]', choices=('Detect', 'Decode', 'Superimpose', 'Project'))
    Parser.add_argument('--VideoPath', type=str, default="../Data/1tagvideo.mp4", help='Path to the video file')
    Parser.add_argument('--TestudoPath', type=str, default="../Data/testudo.png", help='Path to the testudo image')
    Parser.add_argument('--SavePath', type=str, default="../Results/", help='Path to the results folder')
    Parser.add_argument('--SaveResult', action='store_true', help='Toggle to save results')
    Parser.add_argument('--Visualize', action='store_true', help='Toggle visualization')
    
    Args = Parser.parse_args()
    Action = Args.Action
    VideoPath = Args.VideoPath
    TestudoPath = Args.TestudoPath
    SavePath = Args.SavePath
    SaveResult = Args.SaveResult
    Visualize = Args.Visualize

    AR = ARTag()
    if(Action in ['Detect', 'Decode']):
        AR.process_frame(VideoPath, Action, Visualize)
    elif(Action in ['Superimpose', 'Project']):
        AR.process_video(VideoPath, TestudoPath, Action, SavePath, Visualize)
    else:
        print("INVALID ACTION!")


if __name__ == '__main__':
    main()