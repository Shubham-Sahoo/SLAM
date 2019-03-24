import cv2
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


img1 = cv2.imread( "0000000075.png" , 0);
img2 = cv2.imread( "0000000078.png" , 0);
  
 
sift = cv2.xfeatures2d.SIFT_create()
kp1,des1 =sift.detectAndCompute(img1,None)
kp2,des2 =sift.detectAndCompute(img2,None)


  
  #detector->detectAndCompute( src,noArray(), keypoints_1,desc1 );
  #detector->detectAndCompute( dst,noArray(), keypoints_2,desc2 );

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
print pts1.shape
# We select only inlier points
pts1 = pts1[mask.ravel()==1]
pts2 = pts2[mask.ravel()==1]


def drawlines(img1,img2,lines,pts1,pts2):
  ''' img1 - image on which we draw the epilines for the points in img2
  lines - corresponding epilines '''
  r,c = img1.shape
  img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
  img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
  for r,pt1,pt2 in zip(lines,pts1,pts2):
    color = tuple(np.random.randint(0,255,3).tolist())
    x0,y0 = map(int, [0, -r[2]/r[1] ])
    x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
    img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
    img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
    img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
  return img1,img2

# Find epilines corresponding to points in right image (second image) and
# drawing its lines on left image
lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
lines1 = lines1.reshape(-1,3)
img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)

# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
lines2 = lines2.reshape(-1,3)
img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)

stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
disparity = stereo.compute(img1,img2)
#plt.imshow(disparity,'gray')
#plt.show()

P1 = np.eye(3,4)
K = np.matrix([[9.011007e+02,0.000000e+00,6.982947e+02],
              [0.000000e+00,8.970639e+02,2.377447e+02],
              [0.000000e+00,0.000000e+00,1.000000e+00]])


E= np.multiply(np.multiply(np.transpose(K),F),K)


u, s, vt = np.linalg.svd(E,full_matrices=True)

W = np.matrix([[0,-1,0],[1,0,0],[0,0,1]])
R1 = np.multiply(np.multiply(u,np.linalg.inv(W)),vt);
T1 = u[:,2];
np.squeeze(np.asarray(T1))

P2 = np.matrix([[R1[0,0],R1[0,1],R1[0,2],T1[0,0]],
                [R1[1,0],R1[1,1],R1[1,2],T1[1,0]],
                [R1[2,0],R1[2,1],R1[2,2],T1[2,0]]])




pt1=np.reshape(np.ravel(pts1,'F'),(2,-1))


pt2=np.reshape(np.ravel(pts2,'F'),(2,-1))

X=cv2.triangulatePoints(P1[:3],P2[:3],pt1[:2],pt2[:2])
print X



plt.imshow(X,'gray')
plt.show()
x1=np.dot(P1,X)
x2=np.dot(P2,X)
print x2.shape
plt.imshow(x2,'gray')
plt.show()
j=0
print X.shape
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for j in range(X.shape[1]):
#Axes3D.plot(X[0,0],X[1,0],X[2,0])
    ax.scatter(x1[0,j], x1[1,j], x1[2,j], c='r', marker='o')
    ax.scatter(x2[0,j], x2[1,j], x2[2,j], c='b', marker='o')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()

plt.subplot(121),plt.imshow(img5)
plt.subplot(122),plt.imshow(img3)
plt.show()
  


