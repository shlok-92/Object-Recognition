
//Include statements
#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include <ros/package.h>
#include <string>
#include <ros/ros.h>
#include "std_msgs/String.h"
#include <chrono>
#include <ctime>


//Name spaces used
using namespace cv;
using namespace std;

int morph_elem = 1;
int morph_size = 5;

Mat canny_output;
vector<vector<Point> > contours;
vector<Vec4i> hierarchy;
Mat threshold_output;
int thresh=120;
RNG rng(12345);
vector<KeyPoint> kpVidImage;
Mat desImage,img_matches;

int main()
{

    //load training image
    Mat object = imread("/home/shlok/catkin_ws/src/cv_surf/sample_images/input_image_cropped.jpg", CV_LOAD_IMAGE_GRAYSCALE);
    if (!object.data){
        cout<<"Can't open image";
        return -1;
    }

    //SURF Detector, and descriptor parameters
    vector<KeyPoint> kpObject;
    Mat desObject,kpObjectImage;

    //SURF Detector, and descriptor parameters, match object initialization
    int minHess=400;
    //Detect training interest points

   // namedWindow("Keypoints_SURF",CV_WINDOW_NORMAL);
   // resizeWindow("Keypoints_SURF",800,800);

    SurfFeatureDetector detector(minHess);
    detector.detect(object, kpObject);
    drawKeypoints(object,kpObject,kpObjectImage,Scalar::all(-1), DrawMatchesFlags::DEFAULT);
   // imshow("Keypoints_SURF",kpObjectImage);

    //Extract training interest point descriptors
    SurfDescriptorExtractor extractor;
    extractor.compute(object, kpObject, desObject);

    BFMatcher matcher(NORM_L2, true);
    std::vector<DMatch> matches;

    VideoCapture cap(0); // open the default camera
    if(!cap.isOpened())  // check if we succeeded
        return -1;

    Mat edges,segImage,morphImage,framegray;
    Mat element = getStructuringElement( morph_elem, Size( 2*morph_size + 1, 2*morph_size+1 ), Point( morph_size, morph_size ) );

    int sensitivity=20;
   // namedWindow("edges",1);
    std::chrono::system_clock::time_point t1 = std::chrono::system_clock::now();
    for(;;)
    {

        t1 = std::chrono::system_clock::now();
        Mat frame;
        cap >> frame; // get a new frame from camera
        GaussianBlur(frame, edges, Size(7,7), 1.5, 1.5);
        cvtColor(edges, edges, CV_BGR2HSV);
        cvtColor(frame,framegray,CV_RGB2GRAY);
        inRange(edges,Scalar(0,100,150),Scalar(60+sensitivity,255,255),segImage);
        morphologyEx( segImage, morphImage, 3, element );
        //imshow("Morphed Image",morphImage);

        //imshow("edges", edges);
        //imshow("SegImage",segImage);

        /// Detect edges using Threshold
        threshold( segImage, threshold_output, thresh, 255, THRESH_BINARY );
        /// Find contours
        findContours( threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

        /// Approximate contours to polygons + get bounding rects and circles
        vector<vector<Point> > contours_poly( contours.size() );
        vector<Rect> boundRect( contours.size() );
        vector<Rect> boundRectLarge( contours.size() );
        vector<Point2f>center( contours.size() );
        vector<float>radius( contours.size() );

        for( int i = 0; i < contours.size(); i++ )
        { approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
            boundRect[i] = boundingRect( Mat(contours_poly[i]) );


            //minEnclosingCircle( (Mat)contours_poly[i], center[i], radius[i] );
        }


        /// Draw polygonal contour + bonding rects + circles
        Mat drawing = Mat::zeros( threshold_output.size(), CV_8UC3 );
        for( int i = 0; i< contours.size(); i++ )
        {
            //if (contourArea( contours[i],false)>20)
            //   {//  Find the area of contour
            Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
            drawContours( drawing, contours_poly, i, color, 1, 8, vector<Vec4i>(), 0, Point() );
            rectangle( drawing, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0 );

            // }
        }

        // cout<<contours.size()<<endl;

        /// Show in a window
        namedWindow( "Contours", CV_WINDOW_AUTOSIZE );
        imshow( "Contours", drawing );

        double largest_area=1;
        int largest_contour_index=1;

        for( int i = 0; i< contours.size(); i++ ) // iterate through each contour.
        {
            double a=contourArea( contours[i],false);  //  Find the area of contour
            if(a>largest_area){
                largest_area=a;
                largest_contour_index=i;                //Store the index of largest contour

            }

        }
        Mat dst;
        if(largest_contour_index!=0 && largest_contour_index!=1)
        {
            // Scalar color( 255,255,255);
            Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
            drawContours( edges, contours_poly, largest_contour_index, color, 1, 8, vector<Vec4i>(), 0, Point() );
            rectangle( edges, boundRect[largest_contour_index].tl(), boundRect[largest_contour_index].br(), color, 2, 8, 0 );
            dst = framegray(Rect(boundRect[largest_contour_index])).clone();
            //dst = framegray(Rect(boundRect[largest_contour_index].tl().x,boundRect[largest_contour_index].tl().y,boundRect[largest_contour_index].br().x,boundRect[largest_contour_index].br().y)).clone();
            //imshow("SegmentedCameraImage",dst);
        }
        namedWindow( "Biggest box", CV_WINDOW_AUTOSIZE );
        imshow( "Biggest box", edges );
        namedWindow( "matches", CV_WINDOW_AUTOSIZE );


        if (!dst.empty())
        {  //cout<<"entered loop"<<endl;
            detector.detect(dst,kpVidImage);
            extractor.compute(dst,kpVidImage,desImage);
            if(!desImage.empty())
            {
                matcher.match(desObject, desImage, matches);
                drawMatches(object, kpObject, dst, kpVidImage, matches, img_matches);
                if(matches.size()>10)
                {
                    putText(img_matches, "Object Found", cvPoint(10,50),FONT_HERSHEY_COMPLEX_SMALL, 2, cvScalar(0,0,250), 1, CV_AA);
                }
                imshow("matches", img_matches);
            }
        }


        char k=waitKey(30);
        if (k=='p')
        {
            imshow("grabbed frame",drawing);
            waitKey(0);

        }
        if (k=='q')
        {
            break;
        }

        std::cout << "time " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - t1).count() << "ms" <<std::endl;
    }

    return 0;
}

