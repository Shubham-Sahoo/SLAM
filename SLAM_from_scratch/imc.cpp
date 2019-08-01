#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <opencv2/viz.hpp>
#include "opencv2/calib3d.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <vector>


using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;



Mat src, src_gray,dst;
int thresh = 80;
int max_thresh = 255;
const float inlier_threshold = 8.0f;
const float nn_match_ratio = 0.8f;
const char* source_window = "Source image";
const char* dest = "DESt";
const char* corners_window = "Corners detected";
void cornerHarris_demo( int, void* );
int main( int argc, char** argv )
{

  src = imread( "0000000075.png", IMREAD_COLOR );
  dst = imread( "0000000076.png", IMREAD_COLOR );
  
  //cvtColor( src, src_gray, COLOR_BGR2GRAY );
  namedWindow( source_window, WINDOW_AUTOSIZE );
  namedWindow( dest, WINDOW_AUTOSIZE );
  
  imshow( source_window, src );
  imshow( dest , dst);

  
  std::vector<cv::Point2f> f1,f2;

 
 
 // cout<<f1;
 

  int minHessian = 400;

  Ptr<SURF> detector = SURF::create( minHessian );
  Mat desc1,desc2;
  vector<KeyPoint> keypoints_1, keypoints_2;
  Ptr<AKAZE> akaze = AKAZE::create();
  akaze->detectAndCompute(src, noArray(), keypoints_1, desc1);
  akaze->detectAndCompute(dst, noArray(), keypoints_2, desc2);


  
  //detector->detectAndCompute( src,noArray(), keypoints_1,desc1 );
  //detector->detectAndCompute( dst,noArray(), keypoints_2,desc2 );

  BFMatcher matcher(NORM_HAMMING);
  vector< vector<DMatch> >nn_matches;
  matcher.knnMatch(desc1, desc2, nn_matches, 2);
  


  // KeyPoint::convert(keypoints_1,key1);
  // KeyPoint::convert(keypoints_2,key2);
  // cout<<key1;
  vector<Point2f> keyp1,keyp2;
  int i=0;
  while(i<keypoints_1.size())
  {
  keyp1.push_back(Point2f(keypoints_1[i].pt));
        i++;
  }  
  i=0;
  while(i<keypoints_1.size())
  {
  keyp2.push_back(Point2f(keypoints_1[i].pt));
        i++;
  }

  Mat homography= findHomography(keyp1,keyp2);
  cout<<homography<<"\n";  

  Mat R = homography.colRange(0,2).rowRange(0,2);
  cout<<R<<"\n";

  Mat T = homography.colRange(2,3).rowRange(0,2);
  cout<<T<<"\n";

  vector<KeyPoint> matched1, matched2, inliers1, inliers2;
  vector<DMatch> good_matches;
    for(size_t i = 0; i < nn_matches.size(); i++) {
        DMatch first = nn_matches[i][0];
        float dist1 = nn_matches[i][0].distance;
        float dist2 = nn_matches[i][1].distance;

        if(dist1 < nn_match_ratio * dist2) {
            matched1.push_back(keypoints_1[first.queryIdx]);
            matched2.push_back(keypoints_2[first.trainIdx]);
        }
    }

    for(unsigned i = 0; i < matched1.size(); i++) {
        Mat col = Mat::ones(3, 1, CV_64F);
        col.at<double>(0) = matched1[i].pt.x;
        col.at<double>(1) = matched1[i].pt.y;

        col = homography * col;
        col /= col.at<double>(2);
        double dist = sqrt( pow(col.at<double>(0) - matched2[i].pt.x, 2) +
                            pow(col.at<double>(1) - matched2[i].pt.y, 2));

        if(dist < inlier_threshold) {
            int new_i = static_cast<int>(inliers1.size());
            inliers1.push_back(matched1[i]);
            inliers2.push_back(matched2[i]);
            good_matches.push_back(DMatch(new_i, new_i, 0));
        }
    }

  float pro[]={7.070493e+02,0.000000e+00,6.040814e+02,-3.341081e+02,
               0.000000e+00,7.070493e+02,1.805066e+02,2.330660e+00,
               0.000000e+00,0.000000e+00,1.000000e+00,3.201153e-03};
  Mat P = Mat(3, 4, CV_32F, pro);
  Mat pnts3d;
  //cam0=P.colRange(0,3).rowRange(0,3);
  Mat Rt0 = Mat::eye(3, 4, CV_64FC1);
  Mat Rt1 = Mat::eye(3, 4, CV_64FC1);
  float kd[]={9.011007e+02,0.000000e+00,6.982947e+02,
              0.000000e+00,8.970639e+02,2.377447e+02,
              0.000000e+00,0.000000e+00,1.000000e+00};
  Mat K = Mat(3, 3, CV_32F, kd);
  triangulatePoints(Rt0,P,keyp1,keyp2,pnts3d);
  cout<<pnts3d.cols<<"\n";
  //-- Draw keypoints
  Mat img_keypoints_1; Mat img_keypoints_2;

    Mat res;
    drawMatches(src, inliers1, dst, inliers2, good_matches, res);
    imwrite("res.png", res);
  
  drawKeypoints( src, keypoints_1, img_keypoints_1, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
  drawKeypoints( dst, keypoints_2, img_keypoints_2, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
  imshow("result", res);
  //-- Show detected (drawn) keypoints
  imshow("Keypoints 1", img_keypoints_1 );
  imshow("Keypoints 2", img_keypoints_2 );


  




 // Create two sets of points. Points in pts2 are moved 10pixel to the right of the points in pts1.


 
  //Mat tr2=findHomography(f1,f2,0,3);
  //float data[4] = { 1,1,1,1};
  //Mat A = Mat(4,1, CV_32F, data);
  //Mat v;
  //vconcat(tr2,A,v);
  //cout<<A<<"\n";
 
  //Mat out,in;
  
  //in=Mat::zeros(F.size(),CV_32F);
  //triangulatePoints(in,F,f1,f2,out);

  //cornerHarris_demo( 0, 0 );
  waitKey();
  return(0);
}
void cornerHarris_demo( int, void* )
{
  Mat dst, dst_norm, dst_norm_scaled;
  dst = Mat::zeros( src.size(), CV_32FC1 );
  int blockSize = 2;
  int apertureSize = 3;
  double k = 0.05;
  cornerHarris( src_gray, dst, blockSize, apertureSize, k, BORDER_DEFAULT );
  normalize( dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat() );
  convertScaleAbs( dst_norm, dst_norm_scaled );
  for( int j = 0; j < dst_norm.rows ; j++ )
     { for( int i = 0; i < dst_norm.cols; i++ )
          {
            if( (int) dst_norm.at<float>(j,i) > thresh )
              {
               circle( dst_norm_scaled, Point( i, j ), 5,  Scalar(0), 2, 8, 0 );
              }
          }
     }
  namedWindow( corners_window, WINDOW_AUTOSIZE );
  imshow( corners_window, dst_norm_scaled );
}
