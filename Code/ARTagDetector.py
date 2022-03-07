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

        # H = cv2.getPerspectiveTransform(np.float32(c1), np.float32(c2))
        H = homography(np.float32(c1), np.float32(c2))

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
            frame_ = cv2.warpPerspective(frame_gray, H, (w, h))

            if(self.visualize):
                cv2.imshow("Frame", frame)
                cv2.imshow("Frame_", self.normalize(frame_))
                cv2.waitKey(0)

        return frame_

    def drawGrids(self, block, step = 8):
        """
        ref: http://study.marearts.com/2018/11/python-opencv-draw-grid-example-source.html
        """
    
        # block  = cv2.resize(block, (512,512))
        h,w = block.shape[:2]
        
        x = np.linspace(0, w, step).astype(np.int32)
        y = np.linspace(0, h, step).astype(np.int32)

        v_lines = []
        h_lines = []
        for i in range(step):
            v_lines.append( [x[i], 0, x[i], w-1] )
            h_lines.append( [0, int(y[i]), h-1, int(y[i])] )


        for i in range(step):
            [vx1, vy1, vx2, vy2] = v_lines[i]
            [hx1, hy1, hx2, hy2] = h_lines[i]

            block = cv2.line(block, (vx1,vy1), (vx2, vy2), (0,255,255),1 )
            block = cv2.line(block, (hx1,hy1), (hx2, hy2), (0,255,255),1 )
            
        return block

    def decode(self, img):

        frame = remove_padding(img)
        h, w = frame.shape
        
        blocks = []
        rotate = 0
        while True:
            for c in range(0, h, 32):
                for r in range(0, w, 32):
                    blocks.append(np.median(frame[c:c+32, r:r+32]))
            if blocks[-1] == 255:
                print(rotate * 90)
                break
            else:
                frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)
            rotate += 1

        blocks = np.array(blocks).astype(np.int32)
        blocks[blocks < 255] = 0
        blocks[blocks >= 255] = 1

        tag_value = ((blocks[5] * 8) + (blocks[6] * 4) + (blocks[10] * 2) + (blocks[9] * 1))        

        if(self.visualize):
            cv2.imshow("Frame", frame)
            cv2.waitKey(0)

        return rotate, tag_value

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

    def getCubeCoordinates(self, P,cube_size = 128):

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

    def project(self):
        video = cv2.VideoCapture(self.video_path)
        
        currentframe = 0
        ret = True

        while(ret):
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
            mask = cv2.fillPoly(np.copy(frame_), pts = [corners], color = (255,255,255))
            kernel = np.ones((9, 9), np.uint8)
            mask = cv2.erode(mask, kernel) 
            mask = cv2.bitwise_not(mask)

            frame_ = cv2.bitwise_or(frame_gray, mask)
            frame_ = cv2.bitwise_not(frame_)
            corners, w, h, frame_ = self.compute_corners(frame_)
            H = self.compute_homography(128, 128, corners)

            K = np.array([[1346.100595, 0, 932.1633975],
                          [0, 1355.933136, 654.8986796],
                          [0, 0, 1]])

            P = self.ProjectionMatrix(np.linalg.pinv(H), K)
            
            XY = self.getCubeCoordinates(P, cube_size = 128)
            frame_ = self.drawCube(frame, XY)

            if(self.visualize):
                cv2.imshow("Frame", frame)
                cv2.imshow("Frame_", self.normalize(frame_))
                cv2.waitKey(0)

        return frame_

def main():
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--VideoPath', type=str, default="../Data/1tagvideo.mp4", help='Path to the video file')
    Parser.add_argument('--Visualize', action='store_true', help='Toggle visualization')
    
    Args = Parser.parse_args()
    VideoPath = Args.VideoPath
    Visualize = Args.Visualize

    AR = ARTag(VideoPath, Visualize)
    frame = AR.detect()
    rotation, value = AR.decode(frame)
    # AR.project()


if __name__ == '__main__':
    main()