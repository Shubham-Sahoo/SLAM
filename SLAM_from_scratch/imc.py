import cv2
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import glob
#import natsort

trans = np.empty((3,51),dtype=np.float64)
# K = np.matrix([[546.02441,0.000000,319.711258],
#               [0.000000,542.211182,251.374926],
#               [0.000000,0.000000,1.000000]])  
K = np.matrix([[9.842439e+02,0.000000e+00,6.900000e+02],
              [0.000000e+00,9.808141e+02,2.331966e+02],
              [0.000000e+00,0.000000e+00,1.000000e+00]])

# K = np.matrix([[7.188560000000e+02,0.000000000000e+00,6.071928000000e+02],
# [0.000000000000e+00,7.188560000000e+02,1.852157000000e+02],
# [0.000000000000e+00,0.000000000000e+00,1.000000000000e+00]])

# dc= np.matrix([0.262383,-0.953104,-0.005358,0.002628,1.163314])

# h,  w = imgd1.shape[:2]
# K2, roi=cv2.getOptimalNewCameraMatrix(K,dc,(w,h),1,(w,h))


# img1 = cv2.undistort(imgd1, K, dc, None, K2)
# plt.imshow(img1) 
#filenames = [img for img in sorted(glob.glob("images_3/*.png"))] 
#natsort.natsorted(filenames,reverse=False)
#filenames.sort()
# images = []
# for img in glob.glob("images_3/*.png"):
#     n= cv2.imread(img)
#     images.append(n)
#     print (img)



l=10
P1 = np.zeros((3,4),dtype=np.float64)
P1[0,0]=1
P1[1,1]=1
P1[2,2]=1

P2 = np.zeros((3,4),dtype=np.float64)
P2[0,0]=1
P2[1,1]=1
P2[2,2]=1

P3 = np.zeros((3,4),dtype=np.float64)
P3[0,0]=1
P3[1,1]=1
P3[2,2]=1

Tr1 = np.zeros((4,4),dtype=np.float64)
Prev_Tr = np.zeros((4,4),dtype=np.float64)
Prev_Tr[0,0]=1
Prev_Tr[1,1]=1
Prev_Tr[2,2]=1
Prev_Tr[3,3]=1
T1 = np.zeros((3,1),dtype=np.float64)

Tr2 = np.zeros((4,4),dtype=np.float64)
T2 = np.zeros((3,),dtype=np.float64)

while l<15:
  
  #img1 = images[l] #cv2.imread( "../images_2/00000000"+str(l)+".png" , 0)
  #img2 = images[l+2] #cv2.imread( "../images_2/00000000"+str(l+2)+".png" , 0)
  #img3 = images[l+4] #cv2.imread( "../images_2/00000000"+str(l+4)+".png" , 0)

  img1 = cv2.imread( "images_3/00000000"+str(l)+".png" , 0)
  img2 = cv2.imread( "images_3/00000000"+str(l+1)+".png" , 0)
  img3 = cv2.imread( "images_3/00000000"+str(l+2)+".png" , 0)

  surf = cv2.xfeatures2d.SURF_create()
  kp1,des1 =surf.detectAndCompute(img1,None)
  kp2,des2 =surf.detectAndCompute(img2,None)
  kp3,des3 =surf.detectAndCompute(img3,None)

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

  for i,(o,p) in enumerate(matches):
    for j,(o2,p2) in enumerate(matches2): 
      if (kp2[o.trainIdx].pt)==(kp2[o2.queryIdx].pt):
        goodf.append(o)
        pts2f.append(kp2[o.trainIdx].pt)
        pts1f.append(kp1[o.queryIdx].pt)
        pts3f.append(kp3[o2.trainIdx].pt)





  #############............pnp...................#########################





  # def drawlines(img1,img2,lines,pts1,pts2):
  #   ''' img1 - image on which we draw the epilines for the points in img2
  #   lines - corresponding epilines '''
  #   r,c = img1.shape
  #   img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
  #   img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
  #   for r,pt1,pt2 in zip(lines,pts1,pts2):
  #     color1 = tuple(np.random.randint(0,255,3).tolist())
  #     color2 = tuple(np.random.randint(0,100,3).tolist())
  #     x0,y0 = map(int, [0, -r[2]/r[1] ])
  #     x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
  #     img1 = cv2.line(img1, (x0,y0), (x1,y1), color2,1)
  #     img1 = cv2.circle(img1,tuple(pt1),5,color1,-1)
  #     img2 = cv2.circle(img2,tuple(pt2),5,color1,-1)
  #   return img1,img2

  # Find epilines corresponding to points in right image (second image) and
  # drawing its lines on left image
  # lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
  # lines1 = lines1.reshape(-1,3)
  # img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)

  # Find epilines corresponding to points in left image (first image) and
  # drawing its lines on right image
  # lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
  # lines2 = lines2.reshape(-1,3)
  # img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)

  # stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
  # disparity = stereo.compute(img1,img2)
  # plt.imshow(disparity,'gray')
  # plt.show()
  ###############.............1st................##############################

  #K = np.matrix([[9.011007e+02,0.000000e+00,6.982947e+02],
  #             [0.000000e+00,8.970639e+02,2.377447e+02],
  #             [0.000000e+00,0.000000e+00,1.000000e+00]])




  E= np.dot(np.dot(np.transpose(K),F),K)


  u, s, vt = np.linalg.svd(E,full_matrices=True)

  W = np.matrix([[0,-1,0],[1,0,0],[0,0,1]])

  R1 = np.dot(np.dot(u,np.linalg.inv(W)),vt);
  T1 = u[:,2]
  #np.squeeze(np.asarray(T1))

  Tr1[0,0] = R1[0,0]
  Tr1[0,1] = R1[0,1]
  Tr1[0,2] = R1[0,2]  
  Tr1[1,0] = R1[1,0]
  Tr1[1,1] = R1[1,1]
  Tr1[1,2] = R1[1,2]
  Tr1[2,0] = R1[2,0] 
  Tr1[2,1] = R1[2,1]
  Tr1[2,2] = R1[2,2]
  Tr1[0,3] = T1[0,0]
  Tr1[1,3] = T1[1,0]
  Tr1[2,3] = T1[2,0]
  Tr1[3,3] = 1

  Tr1 = np.dot(Prev_Tr,Tr1)
  loc_R1 = Tr1[0:3,0:3]
  temp_T1 = Tr1[0:3,3]
  #print loc_R1.shape
  loc_T1=np.reshape(np.ravel(temp_T1,'F'),(3,-1))
  #print loc_T1.shape
  P2[0:3,0:3] = np.transpose(loc_R1)
  P2[0:3,[3]] = -np.dot(np.transpose(loc_R1),loc_T1)

  P2 = np.dot(K,P2)
  #P2 = np.matrix([[Tr1[0,0],Tr1[0,1],Tr1[0,2],Tr1[0,3]],
  #                [Tr1[1,0],Tr1[1,1],Tr1[1,2],Tr1[1,3]],
  #               [Tr1[2,0],Tr1[2,1],Tr1[2,2],Tr1[2,3]]])



  
  pt1=np.reshape(np.ravel(pts1,'F'),(2,-1))


  pt2=np.reshape(np.ravel(pts2,'F'),(2,-1))

  X=cv2.triangulatePoints(P1[:3],P2[:3],pt1[:2],pt2[:2])
  #print X.shape
  #P1 = P2;
  Prev_Tr = Tr1
  X[3]=1.0
  x1=np.dot(P1,X)
  x2=np.dot(P2,X)




  ###############.....................1st....................############################

  E2 = np.dot(np.dot(np.transpose(K),F2),K)

  u2, s2, vt2 = np.linalg.svd(E2,full_matrices=True)
  R2=np.dot(np.dot(u2,np.linalg.inv(W)),vt2)
  T2=u[:,2]

  #np.squeeze(np.asarray(T2))
  Tr2[0,0] = R2[0,0]
  Tr2[0,1] = R2[0,1]
  Tr2[0,2] = R2[0,2]  
  Tr2[1,0] = R2[1,0]
  Tr2[1,1] = R2[1,1]
  Tr2[1,2] = R2[1,2]
  Tr2[2,0] = R2[2,0] 
  Tr2[2,1] = R2[2,1]
  Tr2[2,2] = R2[2,2]
  Tr2[0,3] = T2[0,0]
  Tr2[1,3] = T2[1,0]
  Tr2[2,3] = T2[2,0]
  Tr2[3,3] = 1

  Tr2 = np.dot(Tr1,Tr2)
  loc_R2 = Tr2[0:3,0:3]
  temp_T2 = Tr2[0:3,3]
  loc_T2=np.reshape(np.ravel(temp_T2,'F'),(3,-1))
  P3[0:3,0:3] = np.transpose(loc_R2)
  P3[0:3,[3]] = -np.dot(np.transpose(loc_R2),loc_T2)

  P3 = np.dot(K,P3)
  #P3 = np.matrix([[R2[0,0],R2[0,1],R2[0,2],T2[0,0]],
  #                [R2[1,0],R2[1,1],R2[1,2],T2[1,0]],
  #                [R2[2,0],R2[2,1],R2[2,2],T2[2,0]]])

  pt12=np.reshape(np.ravel(pts1,'F'),(2,-1))


  pt22=np.reshape(np.ravel(pts2,'F'),(2,-1))

  pt1f=np.reshape(np.ravel(pts1f,'F'),(2,-1))


  pt2f=np.reshape(np.ravel(pts2f,'F'),(2,-1))
  pt3f=np.reshape(np.ravel(pts3f,'F'),(2,-1))

  X1=cv2.triangulatePoints(P1[:3],P2[:3],pt1f[:2],pt2f[:2])
  X1=X1/X1[3]
  X2=cv2.triangulatePoints(P2[:3],P3[:3],pt2f[:2],pt3f[:2])
  X2=X2/X2[3]

  #Xtest=cv2.triangulatePoints(P1[:3],P2[:3],testp1[:2],testp2[:2])
  #Xtest[3]=1.0

  s= (((((X1[0,0]-X1[0,10])**2)+((X1[1,0]-X1[1,10])**2)+((X1[2,0]-X1[2,10])**2))**0.5)/(((X2[0,0]-X2[0,10])**2)+((X2[1,0]-X2[1,10])**2)+((X2[2,0]-X2[2,10])**2))**0.5 + 
     ((((X1[0,5]-X1[0,15])**2)+((X1[1,5]-X1[1,15])**2)+((X1[2,5]-X1[2,15])**2))**0.5)/(((X2[0,5]-X2[0,15])**2)+((X2[1,5]-X2[1,15])**2)+((X2[2,5]-X2[2,15])**2))**0.5)/2.0
  #s=1
  print "s: ", s	
  P3[:,3]*= s
  Xf=cv2.triangulatePoints(P2[:3],P3[:3],pt2f[:2],pt3f[:2])
  Xf=Xf/Xf[3]
  xf=np.dot(P3,Xf)
  
  #print np.transpose(Xf)[:,0:3]
  # rvec=np.zeros((3,3),dtype=np.float32)
  # tvec=np.zeros((3,1),dtype=np.float32)
  _ret, rvec, tvec = cv2.solvePnP(np.transpose(Xf)[:,0:3], np.transpose(pt3f)[:,0:2], K, None)


  nrvec,jac=cv2.Rodrigues(rvec)

  print "loc: " , loc_T1
  print "tvec: " , tvec
  print l
  #trans[0][l]=tvec[0][0]
  #trans[1][l]=tvec[1][0]
  #trans[2][l]=tvec[2][0]
  trans[0][l]=tvec[0]
  trans[1][l]=tvec[1]
  trans[2][l]=tvec[2]
  l=l+1
      


#print trans
j=0
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for j in range(l):#trans.shape[1]):
#Axes3D.plot(X[0,0],X[1,0],X[2,0])
    #ax.scatter(X[0,j], X[1,j], X[2,j], c='r', marker='o')
    #ax.scatter(X2[0,j], X2[1,j], X2[2,j], c='b', marker='o')
    #ax.scatter(Xf[0,j], Xf[1,j], Xf[2,j], c='g', marker='o')
    ax.scatter(trans[0,j], trans[1,j], trans[2,j], c='g', marker='o')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()


# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# for j in range(1):
# #Axes3D.plot(X[0,0],X[1,0],X[2,0])
#     ax.scatter(tvec[0,0], tvec[1,0], tvec[2,0], c='r', marker='o')
#     ax.scatter(0, 0, 0, c='b', marker='o')

# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
# plt.show()

# MIN_MATCH_COUNT = 10

# if len(good)>MIN_MATCH_COUNT:
#     src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
#     dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
#     M, mask1 = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
#     matchesMask = mask1.ravel().tolist()
#     h,w = img1.shape
#     pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
#     dst = cv2.perspectiveTransform(pts,M)
#     img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
# else:
#     print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
#     matchesMask = None

# draw_params = dict(matchColor = (0,255,0), # draw matches in green color
#                    singlePointColor = None,
#                    matchesMask = matchesMask, # draw only inliers
#                    flags = 2)
# img4 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
# plt.imshow(img4, 'gray'),plt.show()



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

#plt.subplot(121),plt.imshow(img5)
#plt.subplot(122),plt.imshow(img3)
#plt.show()
  


