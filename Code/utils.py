#!/usr/env/bin python3

"""
ENPM673 Spring 2021: Perception for Autonomous Robots

Project 1 - April Tag Detection & Tracking

Author(s):
Tanuj Thakkar (tanuj@umd.edu)
M. Engg Robotics
University of Maryland, College Park
"""

import sys
import cv2
import os
import numpy as np

sys.dont_write_bytecode = True

def normalize(img):
    return np.uint8(cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX))

def warp_perspective(img, H, w, h):

    H_inv=np.linalg.inv(H)
    warped=np.zeros((h,w,3),np.uint8)
    for a in range(h):
        for b in range(w):
            f = [a,b,1]
            f = np.reshape(f,(3,1))
            x, y, z = np.matmul(H_inv,f)
            warped[a][b] = img[int(y/z)][int(x/z)]

    return warped

def warp_perspective_(img, H, w, h):

    h_, w_ = img.shape[:2]

    warped=np.zeros((h,w,3),np.uint8)
    for a in range(w_):
        for b in range(h_):
            mat = np.dot(H, [a,b,1])
            i, j, _ = (mat/mat[2]).astype(int)

            if 0 <= i < w-1 and 0 <= j < h-1:
                warped[j,i,:] = img[b,a,:]
            
    return warped

def remove_padding(img):

    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img[3:64, 3:67]
    img = cv2.resize(img, (128, 128), cv2.INTER_CUBIC)
    height, width = img.shape[:2]

    w_left = -1
    for w in range(width):
        col = img[:, w]
        if np.where(col != 0)[0].shape[0] != 0:
            w_left = w
            break

    w_right = -1
    for w in range(width - 1, -1, -1):
        col = img[:, w]
        if np.where(col != 0)[0].shape[0] != 0:
            w_right = w
            break

    h_top = -1
    for h in range(height):
        row = img[h, :]
        if np.where(row != 0)[0].shape[0] != 0:
            h_top = h
            break

    h_bot = -1
    for h in range(height - 1, -1, -1):
        row = img[h, :]
        if np.where(row != 0)[0].shape[0] != 0:
            h_bot = h
            break

    img = img[h_top+3:h_bot-3, w_left+3:w_right-3]
    img = cv2.resize(img, (128, 128), cv2.INTER_CUBIC)

    return img

def draw_grid(img, step):

    h,w = img.shape[:2]

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

        img = cv2.line(img, (vx1,vy1), (vx2, vy2), (0,0,255),1 )
        img = cv2.line(img, (hx1,hy1), (hx2, hy2), (0,0,255),1 )

    return img

def projection_matrix(H, K):

    h1, h2, h3 = H[:,0], H[:,1], H[:,2]
    K_inv = np.linalg.inv(K) 
    lamda = 2/(np.linalg.norm(K_inv.dot(h1)) + np.linalg.norm(K_inv.dot(h2)))
    
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

def compute_cube_corners(P, cube_size):

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
    C = np.dstack((X,Y)).squeeze().astype(np.int32)
    
    return C

def draw_cube(frame, corners):
    
    frame = np.copy(frame)
    for xy_pts in corners:
        x,y = xy_pts
        cv2.circle(frame,(x,y), 3, (0,0,255), -1)

    frame = cv2.line(frame,tuple(corners[0]),tuple(corners[1]), (0,0,255), 2)
    frame = cv2.line(frame,tuple(corners[0]),tuple(corners[2]), (0,0,255), 2)
    frame = cv2.line(frame,tuple(corners[0]),tuple(corners[4]), (0,0,255), 2)
    frame = cv2.line(frame,tuple(corners[1]),tuple(corners[3]), (0,0,255), 2)
    frame = cv2.line(frame,tuple(corners[1]),tuple(corners[5]), (0,0,255), 2)
    frame = cv2.line(frame,tuple(corners[2]),tuple(corners[6]), (0,0,255), 2)
    frame = cv2.line(frame,tuple(corners[2]),tuple(corners[3]), (0,0,255), 2)
    frame = cv2.line(frame,tuple(corners[3]),tuple(corners[7]), (0,0,255), 2)
    frame = cv2.line(frame,tuple(corners[4]),tuple(corners[5]), (0,0,255), 2)
    frame = cv2.line(frame,tuple(corners[4]),tuple(corners[6]), (0,0,255), 2)
    frame = cv2.line(frame,tuple(corners[5]),tuple(corners[7]), (0,0,255), 2)
    frame = cv2.line(frame,tuple(corners[6]),tuple(corners[7]), (0,0,255), 2)

    return frame

def homography(p1, p2):
    p1, p2 = p1.squeeze(), p2.squeeze()
    
    X,Y = p1[:,0],p1[:,1]
    Xp,Yp =  p2[:,0], p2[:,1]

    startFlag = 1

    for (x,y,xp,yp) in zip(X,Y,Xp,Yp):

        if (startFlag == 1) :
            A = np.array([[-x,-y,-1,0,0,0, x*xp, y*xp,xp], [0,0,0,-x,-y,-1, x*yp, y*yp, yp]])
        else:
            tmp = np.array([[-x,-y,-1,0,0,0, x*xp, y*xp,xp], [0,0,0,-x,-y,-1, x*yp, y*yp, yp]])
            A = np.vstack((A, tmp))

        startFlag+=1    

    U,S,Vt = np.linalg.svd(A.astype(np.float32))

    H_ = Vt[8,:]/Vt[8][8]
    H_ = H_.reshape(3,3)
    
    return H_

def rotate_img(img, rotate):
    for i in range((rotate)):
        img = cv2.rotate(img, cv2.cv2.ROTATE_90_CLOCKWISE)
    
    return img