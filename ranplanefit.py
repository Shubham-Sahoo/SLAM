import numpy as np
import numpy.linalg as la
#from svd_solve import svd, svd_solve
from plyfile import PlyData, PlyElement
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



def fit_plane_LSE(points):
    # points: Nx4 homogeneous 3d points
    # return: 1d array of four elements [a, b, c, d] of
    # ax+by+cz+d = 0
    assert points.shape[0] >= 3 # at least 3 points needed
    U, S, Vt = la.svd(points)
    null_space = Vt[-1, :]
    return null_space

def get_point_dist(points, plane):
    # return: 1d array of size N (number of points)
    dists = np.abs(points)/ np.sqrt(plane[0]**2 + plane[1]**2 + plane[2]**2)
    return dists

def fit_plane_LSE_RANSAC(points, iters=1000, inlier_thresh=0.05, return_outlier_list=False):
    # points: Nx4 homogeneous 3d points    inlier is 0.05
    # return: 
    #   plane: 1d array of four elements [a, b, c, d] of ax+by+cz+d = 0
    #   inlier_list: 1d array of size N of inlier points
    max_inlier_num = -1
    max_inlier_list = None
    
    N = points.shape[0]
    assert N >= 3

    for i in range(iters):
        chose_id = np.random.choice(N, 3, replace=False)
        chose_points = points[chose_id, :]
        tmp_plane = fit_plane_LSE(chose_points)
        
        dists = get_point_dist(points, tmp_plane)
        tmp_inlier_list = np.where(dists < inlier_thresh)[0]
        tmp_inliers = points[tmp_inlier_list, :]
        num_inliers = tmp_inliers.shape[0]
        if num_inliers > max_inlier_num:
            max_inlier_num = num_inliers
            max_inlier_list = tmp_inlier_list
        
        #print('iter %d, %d inliers' % (i, max_inlier_num))

    final_points = points[max_inlier_list, :]
    plane = fit_plane_LSE(final_points)
    
    fit_variance = np.var(get_point_dist(final_points, plane))
    print('RANSAC fit variance: %f' % fit_variance)
    print(plane)

    dists = get_point_dist(points, plane)

    select_thresh = inlier_thresh * 1

    inlier_list = np.where(dists < select_thresh)[0]
    if not return_outlier_list:
        return plane, inlier_list
    else:
        outlier_list = np.where(dists >= select_thresh)[0]
        return plane, inlier_list, outlier_list

def main():

    
    plyd=PlyData.read('00000015_Resection.ply')
    print(plyd.elements[0].count)  #9648
    #print(plyd.elements[0].data[])
    points=np.zeros((plyd.elements[0].count,4),dtype=np.float64)
    #points=np.zeros((,4),dtype=np.float64)

    for i in range(plyd.elements[0].count):
        points[i][0]=plyd.elements[0].data[i][0]
        points[i][1]=plyd.elements[0].data[i][1]
        points[i][2]=plyd.elements[0].data[i][2]
        points[i][3]=1
    print(points)
    plane=np.array((),dtype=np.float64)
    plane,inlier_list=fit_plane_LSE_RANSAC(points)
    print("hiiiiiiii")
    print(plane)
    a=plane[0]
    b=plane[1]
    c=plane[2]

    point  = np.array([1, 2, 3])
    normal = np.array([a, b, c],dtype=np.float64)
    #normal = np.array([1, 2, 3],dtype=np.float64)
    point2 = np.array([10, 50, 50])

    # a plane is a*x+b*y+c*z+d=0
    # [a,b,c] is the normal. Thus, we have to calculate
    # d and we're set
    #d = -point.dot(normal)
    d=plane[3]
    # create x,y
    xx, yy = np.meshgrid(range(2), range(2))

    # calculate corresponding z
    z = (-normal[0] * xx - normal[1] * yy - d) * 1. /normal[2]

    # plot the surface
    plt3d = plt.figure().gca(projection='3d')
    plt3d.plot_surface(xx, yy, z, alpha=0.2)
    ax = plt.gca()
    ax.hold(True)

    #and i would like to plot this point : 
    for j in range(0,plyd.elements[0].count,5):
        ax.scatter(points[j][0],points[j][1],points[j][2],color='red')
        ax=plt.gca()
        ax.hold(True)
    #ax.scatter(point2[0] , point2[1] , point2[2],  color='green')

    plt.show()
    #print(len(plane))

if __name__=="__main__":
    main()
    
    