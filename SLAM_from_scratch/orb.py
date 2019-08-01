import cv2
import numpy as np
img = cv2.imread("left-0009", 0)
sift = cv2.xfeatures2d.SIFT_create()
surf = cv2.xfeatures2d.SURF_create()
orb = cv2.ORB_create(nfeatures=1500)
keypoints, descriptors = surf.detectAndCompute(img, None)
img = cv2.drawKeypoints(img, keypoints, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()