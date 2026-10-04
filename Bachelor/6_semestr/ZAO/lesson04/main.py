#!/usr/bin/python

import sys
import cv2
import numpy as np
import math
import struct
from datetime import datetime
import glob

def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect

def four_point_transform(image, one_c):
    #https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
    
    pts = [((float(one_c[0])), float(one_c[1])),
            ((float(one_c[2])), float(one_c[3])),
            ((float(one_c[4])), float(one_c[5])),
            ((float(one_c[6])), float(one_c[7]))]
    
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(np.array(pts))
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
	    [0, 0],
	    [maxWidth - 1, 0],
	    [maxWidth - 1, maxHeight - 1],
	    [0, maxHeight - 1]], dtype = "float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped

def load_parking_map(path):
    spots = []
    with open(path, 'r') as f:
        for line in f:
            vals = list(map(int, line.split()))
            if len(vals) >= 8:
                pts = np.array([[vals[0], vals[1]],
                                [vals[2], vals[3]],
                                [vals[4], vals[5]],
                                [vals[6], vals[7]]], dtype=np.int32)
                spots.append(pts)
    return spots

def preprocess(img_color):
    img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    img = cv2.GaussianBlur(img, (5,5), 0)
    return img

def edges_from_gray(gray):
    edges = cv2.Canny(gray, 100, 200)           # doladit prahy
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    # edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
    return edges

def check_spots(edges, spots, occ_thresh=0.12):
    results = []
    h, w = edges.shape
    for i, pts in enumerate(spots):
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        # edge density = procento pixelů hran uvnitř masky
        edge_pixels = cv2.countNonZero(cv2.bitwise_and(edges, edges, mask=mask))
        area = cv2.countNonZero(mask)
        density = edge_pixels / (area + 1e-6)
        occupied = density > occ_thresh
        results.append((i, occupied, density, pts))
    return results
    
def load_ground_truth(img_path):
    gt_path = img_path.replace('.jpg', '.txt').replace('.png', '.txt') # uprav podle tvých souborů
    try:
        with open(gt_path, 'r') as f:
            # Předpokládáme, že co řádek, to stav jednoho místa (0 nebo 1)
            return [int(line.strip()) for line in f if line.strip()]
    except FileNotFoundError:
        return None
    
def main(argv):
    avg_accuracy = 0
    # load parking map from repo file
    spots = load_parking_map('parking_map_python.txt')

    # find test images (accept common extensions)
    exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
    test_images = []
    for e in exts:
        test_images.extend(glob.glob(f"test_images_zao/{e}"))
    test_images.sort()

    if len(test_images) == 0:
        print("No test images found in test_images_zao/ - make sure images exist and extensions are supported.")
        return

    for img_path in test_images:
        print("Processing", img_path)
        img_color = cv2.imread(img_path)
        if img_color is None:
            print(f"Could not read image: {img_path} — skipping")
            continue

        # preprocess / edge detection / check
        gray = preprocess(img_color)
        edges = edges_from_gray(gray)
        res = check_spots(edges, spots)
        gt = load_ground_truth(img_path)
        
        if gt is not None:
            tp = tn = fp = fn = 0
            for i, result in enumerate(res):
                occupied_pred = result[1]  # tvůj odhad (True/False)
                occupied_gt = bool(gt[i])  # realita z TXT

                if occupied_pred and occupied_gt: tp += 1
                elif not occupied_pred and not occupied_gt: tn += 1
                elif occupied_pred and not occupied_gt: fp += 1
                elif not occupied_pred and occupied_gt: fn += 1

            accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-6)
            avg_accuracy += accuracy
            print(f"Accuracy for {img_path}: {accuracy:.2%}")

        # draw results on a copy and show/save
        out = img_color.copy()
        for i, occupied, density, pts in res:
            color = (0,0,255) if occupied else (0,255,0)
            cv2.polylines(out, [pts], True, color, 2)
            cx = int(np.mean(pts[:,0])); cy = int(np.mean(pts[:,1]))
            cv2.putText(out, f'{i}:{density:.3f}', (cx-20, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # show result (press any key to go to next), or save to disk
        cv2.imshow('parking result edges', edges)
        cv2.imshow('parking result', out)
        key = cv2.waitKey(0)
        if key == 27:  # ESC to quit early
            break

    cv2.destroyAllWindows()
    
    print(f"Average accuracy over {len(test_images)} images: {avg_accuracy/len(test_images):.2%}")

if __name__ == "__main__":
   main(sys.argv[1:])
