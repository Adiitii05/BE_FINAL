import matplotlib.pyplot as plt
import matplotlib.image as img
import cv2 as cv

def ridge_count(im):
	thresh = cv.threshold(im, 120, 255, cv.THRESH_BINARY_INV)[1]
	cnts = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
	cnts = cnts[0] if len(cnts) == 2 else cnts[1]
	lines = 0
	for c in cnts:
		cv.drawContours(im, [c], -1, (36,255,12), 3)
		lines += 1
		results = print(lines)

	return results
