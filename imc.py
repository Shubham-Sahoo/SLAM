import cv2
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


img1 = cv2.imread( "0000000060.png" , 0);
img2 = cv2.imread( "0000000075.png" , 0);
img3 = cv2.imread( "0000000090.png" , 0)
  
 
sift = cv2.xfeatures2d.SIFT_create()
kp1,des1 =sift.detectAndCompute(img1,None)
kp2,des2 =sift.detectAndCompute(img2,None)
kp3,des3 =sift.detectAndCompute(img3,None)


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
pts1[1,0]
pts1 = pts1[mask.ravel()==1]
pts2 = pts2[mask.ravel()==1]

#############.............2nd.....................#######################
flann2 = cv2.FlannBasedMatcher(index_params,search_params)
matches2 = flann2.knnMatch(des2,des3,k=2)  


good2 = []
pts12 = []
pts22 = []

for i,(m,n) in enumerate(matches2):
  if m.distance < 0.8*n.distance:
    good2.append(m)
    pts22.append(kp3[m.trainIdx].pt)
    pts12.append(kp2[m.queryIdx].pt)

pts12 = np.float32(pts12)
pts22 = np.float32(pts22)
F2,mask2= cv2.findFundamentalMat(pts12,pts22,cv2.FM_RANSAC)
#F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)

# We select only inlier points
pts12 = pts12[mask2.ravel()==1]
pts22 = pts22[mask2.ravel()==1]

#############............2nd....................########################

#############.............3rd.....................#######################
flann3 = cv2.FlannBasedMatcher(index_params,search_params)
matches3 = flann2.knnMatch(des1,des3,k=2)  


good3 = []
pts13 = []
pts23 = []

for i,(m,n) in enumerate(matches3):
  if m.distance < 0.8*n.distance:
    good3.append(m)
    pts23.append(kp3[m.trainIdx].pt)
    pts13.append(kp1[m.queryIdx].pt)

pts13 = np.float32(pts13)
pts23 = np.float32(pts23)
F3,mask3= cv2.findFundamentalMat(pts13,pts23,cv2.FM_RANSAC)
#F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)

# We select only inlier points
pts13 = pts13[mask3.ravel()==1]
pts23 = pts23[mask3.ravel()==1]
#pts1f = pts13[mask3.ravel()==1&&mask2.ravel()==1&&mask1.ravel()==1]
#pts2f = pts23[mask3.ravel()==1&&mask2.ravel()==1&&mask1.ravel()==1]

#############............3rd....................########################

#############............pnp...................#########################

goodf = []
pts1f = []
pts2f = []
pts3f = []

for i,(m,n) in enumerate(matches):
  for j,(m2,n2) in enumerate(matches2): 
    if (kp2[m.trainIdx].pt)==(kp2[m2.trainIdx].pt):
      goodf.append(m)
      pts2f.append(kp2[m.trainIdx].pt)
      pts1f.append(kp1[m.queryIdx].pt)
      pts3f.append(kp3[m2.trainIdx].pt)





#############............pnp...................#########################





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
plt.imshow(disparity,'gray')
plt.show()
###############.............1st................##############################
P1 = np.eye(3,4)
K = np.matrix([[9.011007e+02,0.000000e+00,6.982947e+02],
              [0.000000e+00,8.970639e+02,2.377447e+02],
              [0.000000e+00,0.000000e+00,1.000000e+00]])


E= np.dot(np.dot(np.transpose(K),F),K)


u, s, vt = np.linalg.svd(E,full_matrices=True)

W = np.matrix([[0,-1,0],[1,0,0],[0,0,1]])

R1 = np.dot(np.dot(u,np.linalg.inv(W)),vt);

T1 = u[:,2]
np.squeeze(np.asarray(T1))

P2 = np.matrix([[R1[0,0],R1[0,1],R1[0,2],T1[0,0]],
                [R1[1,0],R1[1,1],R1[1,2],T1[1,0]],
                [R1[2,0],R1[2,1],R1[2,2],T1[2,0]]])




pt1=np.reshape(np.ravel(pts1,'F'),(2,-1))


pt2=np.reshape(np.ravel(pts2,'F'),(2,-1))

X=cv2.triangulatePoints(P1[:3],P2[:3],pt1[:2],pt2[:2])
X[3]=1.0
x1=np.dot(P1,X)
x2=np.dot(P2,X)

###############.....................1st....................############################

E2 = np.dot(np.dot(np.transpose(K),F2),K)

u2, s2, vt2 = np.linalg.svd(E2,full_matrices=True)
R2=np.dot(np.dot(u2,np.linalg.inv(W)),vt2)
T2=u[:,2]
np.squeeze(np.asarray(T2))
P3 = np.matrix([[R2[0,0],R2[0,1],R2[0,2],T2[0,0]],
                [R2[1,0],R2[1,1],R2[1,2],T2[1,0]],
                [R2[2,0],R2[2,1],R2[2,2],T2[2,0]]])

pt12=np.reshape(np.ravel(pts1,'F'),(2,-1))


pt22=np.reshape(np.ravel(pts2,'F'),(2,-1))

pt1f=np.reshape(np.ravel(pts1f,'F'),(2,-1))


pt2f=np.reshape(np.ravel(pts2f,'F'),(2,-1))
pt3f=np.reshape(np.ravel(pts3f,'F'),(2,-1))

X1=cv2.triangulatePoints(P1[:3],P2[:3],pt1f[:2],pt2f[:2])
X1[3]=1.0
X2=cv2.triangulatePoints(P2[:3],P3[:3],pt2f[:2],pt3f[:2])
X2[3]=1.0

s= ((((X1[0,0]-X1[0,10])**2)+((X1[1,0]-X1[1,10])**2)+((X1[2,0]-X1[2,10])**2))**0.5)/(((X2[0,0]-X2[0,10])**2)+((X2[1,0]-X2[1,10])**2)+((X2[2,0]-X2[2,10])**2))**0.5
print s

P3[:,3]*=s
Xf=cv2.triangulatePoints(P2[:3],P3[:3],pt2f[:2],pt3f[:2])
xf=np.dot(P3,Xf)
print np.transpose(Xf)[:,0:3]
# rvec=np.zeros((3,3),dtype=np.float32)
# tvec=np.zeros((3,1),dtype=np.float32)
_ret, rvec, tvec = cv2.solvePnP(np.transpose(Xf)[:,0:3], np.transpose(pt3f)[:,0:2], K, None)


nrvec,jac=cv2.Rodrigues(rvec)

print tvec

j=0
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for j in range(X.shape[1]):
#Axes3D.plot(X[0,0],X[1,0],X[2,0])
    ax.scatter(X[0,j], X[1,j], X[2,j], c='r', marker='o')
    ax.scatter(X2[0,j], X2[1,j], X2[2,j], c='b', marker='o')
    ax.scatter(Xf[0,j], Xf[1,j], Xf[2,j], c='g', marker='o')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()

stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
disparity = stereo.compute(pt1f,pt2f)
plt.imshow(disparity,'gray')
plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# for j in range(X2.shape[1]):
# #Axes3D.plot(X[0,0],X[1,0],X[2,0])
#     ax.scatter(X2[0,j], X2[1,j], X2[2,j], c='r', marker='o')
    
# ax.set_xlabel('X2 Label')
# ax.set_ylabel('Y2 Label')
# ax.set_zlabel('Z2 Label')
# plt.show()


# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# for j in range(Xf.shape[1]):
# #Axes3D.plot(X[0,0],X[1,0],X[2,0])
#     ax.scatter(Xf[0,j], Xf[1,j], Xf[2,j], c='r', marker='o')
    
# ax.set_xlabel('Xf Label')
# ax.set_ylabel('Yf Label')
# ax.set_zlabel('Zf Label')
# plt.show()

plt.subplot(121),plt.imshow(img5)
plt.subplot(122),plt.imshow(img3)
plt.show()
  


