import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
import sys
from sklearn.feature_extraction import image


def computeDisp(Il, Ir, max_disp):
    h, w, ch = Il.shape
    labels = np.zeros((h, w), dtype=np.float32)
    Il = Il.astype(np.float32)
    Ir = Ir.astype(np.float32)

    # >>> Cost computation
    # TODO: Compute matching cost from Il and Ir
    print('[Cost computation]...')
    ts = time.time()

    ## init
    thresBorder = 5 / 255
    thresColor = 15 / 255
    thresGrad = 2 / 255
    alpha = 0.95
    cost = np.ones((h, w, max_disp)) * thresBorder

    Il_g = cv2.cvtColor(Il, cv2.COLOR_BGR2GRAY) / 255
    Ir_g = cv2.cvtColor(Ir, cv2.COLOR_BGR2GRAY) / 255
    Il = Il / 255
    Ir = Ir / 255
    fx_l = np.gradient(Il_g, axis=1)
    fx_r = np.gradient(Ir_g, axis=1)
    fx_l = fx_l + 0.5
    fx_r = fx_r + 0.5

    for d in range(1, max_disp+1):
        tmp = np.ones((h,w,ch)) * thresBorder
        tmp[:, d:w, :] = Ir[:, :w-d, :]
        p_color = abs(tmp - Il)
        p_color = np.mean(p_color, axis=2)
        p_color = np.minimum(p_color, thresColor)

        tmp = np.ones((h,w))*thresBorder
        tmp[:, d:w] = fx_r[:, :w-d]
        p_grad = abs(tmp - fx_l)
        p_grad = np.minimum(p_grad, thresGrad)

        p = (1-alpha) * p_color + alpha * p_grad
        cost[:, :, d-1] = p

    te = time.time()
    print('Elapse time: {}...'.format(te-ts))

  
    # >>> Cost aggregation
    # TODO: Refine cost by aggregate nearby costs
    print('[Cost aggregation]...')
    ts = time.time()

    ## Use guided filter
    cost = cost.astype(np.float32)
    Il = Il * 255
    for d in range(max_disp):
        cost[:, :, d] = cv2.ximgproc.guidedFilter(guide=Il, src=cost[:, :, d:d+1], radius=17, eps=1e-4)

    te = time.time()
    print('Elapse time: {}...'.format(te-ts))


    # >>> Disparity optimization
    # TODO: Find optimal disparity based on estimated cost. Usually winner-take-all.
    print('[Disparity optimization]...')
    ts = time.time()

    final_labels = cost.argmin(axis=2)
    te = time.time()
    print('Elapse time: {}...'.format(te-ts))


    # >>> Disparity refinement
    # TODO: Do whatever to enhance the disparity map
    # ex: Left-right consistency check + hole filling + weighted median filtering
    print('[Disparity refinement]...')
    ts = time.time()

    ## hole filling
    input_labels = final_labels.copy()
    final_labels = holeFilling(input_labels, max_disp)

    ## weighted median filter
    labels = cv2.ximgproc.weightedMedianFilter(joint=Il.astype(np.uint8), src=final_labels.astype(np.uint8), r=3)

    return labels.astype(np.uint8)


def holeFilling(final_labels, max_disp):
    h,w = final_labels.shape
    occPix = np.zeros((h,w))
    occPix[final_labels==0] = 1
    
    fillVals = np.ones((h)) * max_disp
    FL = final_labels.copy()

    for col in range(w):
        curCol = final_labels[:,col].copy()
        curCol[curCol == 0] = fillVals[curCol == 0]
        fillVals[curCol != 0] = curCol[curCol != 0]
        FL[:,col] = curCol
    
    fillVals = np.ones((h)) * max_disp
    FR = final_labels.copy()
    for col in reversed(range(w)):
        curCol = final_labels[:,col].copy()
        curCol[curCol == 0] = fillVals[curCol == 0]
        fillVals[curCol != 0] = curCol[curCol != 0]
        FR[:,col] = curCol

    final_labels = np.fmin(FL, FR)
    return final_labels