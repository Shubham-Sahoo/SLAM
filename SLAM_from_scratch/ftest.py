import cv2
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


img1 = cv2.imread( "000000.png" , 0)
img2 = cv2.imread( "000006.png" , 0)
 

# K = np.matrix([[546.024414,0.000000,319.711258],
#               [0.000000,542.211182,251.374926],
#               [0.000000,0.000000,1.000000]])  

K = np.matrix([[7.188560000000e+02,0.000000000000e+00,6.071928000000e+02],
[0.000000000000e+00,7.188560000000e+02,1.852157000000e+02],
[0.000000000000e+00,0.000000000000e+00,1.000000000000e+00]])

# dc= np.matrix([0.262383,-0.953104,-0.005358,0.002628,1.163314])

# h,  w = imgd1.shape[:2]
# K2, roi=cv2.getOptimalNewCameraMatrix(K,dc,(w,h),1,(w,h))


# img1 = cv2.undistort(imgd1, K, dc, None, K2)
# plt.imshow(img1) 


surf = cv2.xfeatures2d.SURF_create()
#surf.hessianThreshold = 500
kp1,des1 =surf.detectAndCompute(img1,None)
kp2,des2 =surf.detectAndCompute(img2,None)

#img1 = cv2.drawKeypoints(img1,kp1,0,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)



FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)
  
flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)  


good = []
pts1 = []
pts2 = []

for i,(m,n) in enumerate(matches):
  if m.distance < 0.8*n.distance:
    good.append(m)
    pts2.append(kp2[m.trainIdx].pt)
    pts1.append(kp1[m.queryIdx].pt)

pts1 = np.float32(pts1)
pts2 = np.float32(pts2)
F,mask= cv2.findFundamentalMat(pts1,pts2,cv2.FM_RANSAC)
#F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)

# We select only inlier points

pts1 = pts1[mask.ravel()==1]
pts2 = pts2[mask.ravel()==1]





# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# for j in range(X.shape[1]):
# #Axes3D.plot(X[0,0],X[1,0],X[2,0])
# 	
#     ax.scatter(X[0,j], X[1,j], X[2,j], c='r', marker='o')
#     ax.scatter(X2[0,j], X2[1,j], X2[2,j], c='b', marker='o')
#     ax.scatter(Xf[0,j], Xf[1,j], Xf[2,j], c='g', marker='o')

# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
# plt.show()


for i in range(pts1.shape[0]):
	cv2.circle(img1,(pts1[i,0],pts1[i,1]), 3, (0,0,255), 2)
for i in range(pts2.shape[0]):
	cv2.circle(img2,(pts2[i,0],pts2[i,1]), 3, (0,0,255), 2)	

cv2.imshow('XYZ ',img1)
cv2.imshow('XYZ 2',img2)
cv2.waitKey(0)
cv2.destroyAllWindows()



# cv2.waitKey(0)
# cv2.destroyAllWindows()