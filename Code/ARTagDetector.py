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
from PIL import Image
from skimage.filters import threshold_otsu

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
        mask = cv2.fillPoly(np.copy(frame_), pts = [corners], color =(255,255,255))
        kernel = np.ones((9, 9), np.uint8)
        mask = cv2.erode(mask, kernel) 
        mask = cv2.bitwise_not(mask)

        frame_ = cv2.bitwise_or(frame_gray, mask)
        frame_ = cv2.bitwise_not(frame_)
        corners, w, h, frame_ = self.compute_corners(frame_)
        H = self.compute_homography(w, h, corners)
        # ar_tag = cv2.warpPerspective(frame_gray, H, (w, h))
        ar_tag = warp_perspective(H, frame_gray, w, h)

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
            if blocks[-1] == 255:
                break
            else:
                ar_tag = cv2.rotate(ar_tag, cv2.cv2.ROTATE_90_CLOCKWISE)
            rotate += 1

        # rotate = rotate * 90

        blocks = np.array(blocks).astype(np.int32)
        blocks[blocks < 255] = 0
        blocks[blocks >= 255] = 1

        tag_value = ((blocks[5] * 8) + (blocks[6] * 4) + (blocks[10] * 2) + (blocks[9] * 1))

        ar_tag = draw_grid(ar_tag, 5)

        if(visualize):
            cv2.imshow("AR Tag Decoded", ar_tag)
            cv2.waitKey(0)

        return rotate, tag_value

    def superimpose(self, frame, testudo_img, ar_tag_corners, rotation, visualize):

        testudo_img = cv2.imread(testudo_img)
        testudo_img = rotate_img(testudo_img, rotation)

        h, w = testudo_img.shape[:2]
        H = self.compute_homography(w, h, ar_tag_corners)
        testudo_img = cv2.warpPerspective(testudo_img, np.linalg.inv(H), (frame.shape[1], frame.shape[0]))

        _, warpedTestudo_mask = cv2.threshold(cv2.cvtColor(testudo_img, cv2.COLOR_BGR2GRAY), 20, 255, cv2.THRESH_BINARY_INV)
        warpedTestudo_mask = np.dstack((warpedTestudo_mask,warpedTestudo_mask,warpedTestudo_mask))
        testudo_img = cv2.bitwise_or(testudo_img, warpedTestudo_mask)

        _, ar_tag_mask = cv2.threshold(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 220, 255, cv2.THRESH_BINARY_INV)
        ar_tag_mask = cv2.fillPoly(np.copy(ar_tag_mask), pts = [ar_tag_corners], color=(125))
        ar_tag_mask[ar_tag_mask != 125] = 0
        ar_tag_mask[ar_tag_mask == 125] = 255
        ar_tag_mask = np.dstack((ar_tag_mask,ar_tag_mask,ar_tag_mask))
        ar_tag_mask = cv2.bitwise_and(testudo_img, ar_tag_mask)

        frame_mask = cv2.fillPoly(np.copy(frame), pts = [ar_tag_corners], color=(0))

        testudo_img = cv2.bitwise_or(frame_mask, ar_tag_mask)

        if(visualize):
            cv2.imshow("Testudo", testudo_img)
            # cv2.imshow("AR Tag Mask", np.uint8(ar_tag_mask))
            cv2.waitKey(0)


    def ProjectionMatrix(self, H, K):

        h1, h2, h3 = H[:,0], H[:,1], H[:,2]
        K_inv = np.linalg.inv(K) 
        lamda = 2/(np.linalg.norm(K_inv.dot(h1)) + np.linalg.norm(K_inv.dot(h2)) )
        
        B_ = lamda*K_inv.dot(H)

        if np.linalg.det(B_) > 0 :
            B = B_
        else:
            B = - B_

        r1, r2, r3 = B[:,0], B[:,1], np.cross(B[:,0], B[:,1])
        t = B[:,2]

        RTmatrix = np.dstack((r1,r2,r3,t)).squeeze()
        P = K.dot(RTmatrix)
        return P

    def getCubeCoordinates(self, P, cube_size = 128):

        x1,y1,z1 = P.dot([0,0,0,1])
        x2,y2,z2 = P.dot([0,cube_size,0,1])
        x3,y3,z3 = P.dot([cube_size,0,0,1])
        x4,y4,z4 = P.dot([cube_size,cube_size,0,1])

        x5,y5,z5 = P.dot([0,0,-cube_size,1])
        x6,y6,z6 = P.dot([0,cube_size,-cube_size,1])
        x7,y7,z7 = P.dot([cube_size,0,-cube_size,1])
        x8,y8,z8 = P.dot([cube_size,cube_size,-cube_size,1])

        X = [x1/z1 ,x2/z2 ,x3/z3 ,x4/z4 ,x5/z5 ,x6/z6 ,x7/z7 ,x8/z8] 
        Y = [y1/z1 ,y2/z2 ,y3/z3 ,y4/z4 ,y5/z5 ,y6/z6 ,y7/z7 ,y8/z8] 
        XY = np.dstack((X,Y)).squeeze().astype(np.int32)
        
        return XY

    def drawCube(self, im_org, XY):
        im_print = im_org.copy()
        for xy_pts in XY:
            x,y = xy_pts
            cv2.circle(im_print,(x,y), 3, (0,0,255), -1)

        im_print = cv2.line(im_print,tuple(XY[0]),tuple(XY[1]), (0,255,255), 2)
        im_print = cv2.line(im_print,tuple(XY[0]),tuple(XY[2]), (0,255,255), 2)
        im_print = cv2.line(im_print,tuple(XY[0]),tuple(XY[4]), (0,255,255), 2)
        im_print = cv2.line(im_print,tuple(XY[1]),tuple(XY[3]), (0,225,255), 2)
        im_print = cv2.line(im_print,tuple(XY[1]),tuple(XY[5]), (0,225,255), 2)
        im_print = cv2.line(im_print,tuple(XY[2]),tuple(XY[6]), (0,200,255), 2)
        im_print = cv2.line(im_print,tuple(XY[2]),tuple(XY[3]), (0,200,255), 2)
        im_print = cv2.line(im_print,tuple(XY[3]),tuple(XY[7]), (0,175,255), 2)
        im_print = cv2.line(im_print,tuple(XY[4]),tuple(XY[5]), (0,150,255), 2)
        im_print = cv2.line(im_print,tuple(XY[4]),tuple(XY[6]), (0,150,255), 2)
        im_print = cv2.line(im_print,tuple(XY[5]),tuple(XY[7]), (0,125,255), 2)
        im_print = cv2.line(im_print,tuple(XY[6]),tuple(XY[7]), (0,100,255), 2)

        return im_print

    def project(self, frame, ar_tag_corners, cube_size, visualize):

        H = self.compute_homography(cube_size, cube_size, ar_tag_corners)

        K = np.array([[1346.100595, 0, 932.1633975],
                      [0, 1355.933136, 654.8986796],
                      [0, 0, 1]])

        P = self.ProjectionMatrix(np.linalg.pinv(H), K)
        
        XY = self.getCubeCoordinates(P, cube_size)
        cube = self.drawCube(frame, XY)

        if(visualize):
            cv2.imshow("Cube", normalize(cube))
            cv2.waitKey(0)

        return cube

    def process_video(self, video_path, testudo_path, action, visualize):

        video = cv2.VideoCapture(video_path)
        ret = True

        while(ret):
            ret, frame = video.read()
            frame = cv2.resize(frame, dsize=None, fx=0.4, fy=0.4, interpolation=cv2.INTER_CUBIC)

            frame, ar_tag, corners, H = self.detect(frame, visualize)
            rotation, value = self.decode(ar_tag, visualize)
            self.superimpose(frame, testudo_path, corners, rotation, visualize)
            self.project(frame, corners, 128, visualize)


def main():
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--VideoPath', type=str, default="../Data/1tagvideo.mp4", help='Path to the video file')
    Parser.add_argument('--TestudoPath', type=str, default="../Data/testudo.png", help='Path to the testudo image')
    Parser.add_argument('--Action', type=str, default="detect", help='Select action to perform from [Detect, Decode, Superimpose, Project]', choices=('Detect', 'Decode', 'Superimpose', 'Project'))
    Parser.add_argument('--Visualize', action='store_true', help='Toggle visualization')
    Parser.add_argument('--SaveResult', action='store_true', help='Toggle to save results')
    
    Args = Parser.parse_args()
    VideoPath = Args.VideoPath
    Action = Args.Action
    TestudoPath = Args.TestudoPath
    Visualize = Args.Visualize
    SaveResult = Args.SaveResult

    AR = ARTag()
    AR.process_video(VideoPath, TestudoPath, Action, Visualize)


if __name__ == '__main__':
    main()