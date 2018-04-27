//
//  aff_match_images.cpp
//
//  Created by Evgeny on 7/22/13.
//  Copyright (c) 2013 Evgeny Toropov. All rights reserved.
//

#include <iostream>

#include <boost/filesystem.hpp>

#include <tclap/CmdLine.h>

#include <opencv2/imgproc/imgproc.hpp>

#include "aff_features2d.hpp"

#include "mediaIO.h"
#include "featuresIO.h"


using namespace std;
using namespace cv;
using namespace boost::filesystem;


int main(int argc, const char * argv[])
{
    int              maxTilt        = 3;
    float            threshold      = 0.4;  // matching threshold
    string           imageName1     = "data/adam1.jpg";
    string           imageName2     = "data/adam2.jpg";


    Mat im1, im2;
    if ( !exists(path(imageName1)) || !exists(path(imageName2)) )
    {
        cerr << "adam1.jpg or adam2.jpg not found. Try running the demo from under the Erie root directory" << endl;
        return -1;
    }

    if (!evg::loadImage(imageName1, im1)) return 0;
    if (!evg::loadImage(imageName2, im2)) return 0;


    // create underlying feature2d detector, extractor, and matcher
    
    /*
    Ptr<FeatureDetector> detector = new SIFT();
    Ptr<DescriptorExtractor> extractor = new SIFT();
    Ptr<DescriptorMatcher> matcher = new FlannBasedMatcher();
    */
    
    Ptr<FeatureDetector> detector = new ORB(10000);
    Ptr<DescriptorExtractor> extractor = new BRISK();
    Ptr<DescriptorMatcher> matcher = new BFMatcher(NORM_HAMMING);
    
    // create affine-invariant matching wrapper on top
    Ptr<cv::affma::AffMatcherHelper> affMatcherHelper = cv::affma::createAffMatcherHelper (detector, extractor, matcher);

    // variables to store results
    vector<KeyPoint> keypoints1, keypoints2;
    vector<DMatch> matches;

    if (maxTilt >= 0)
        // match with preset maxTilt
        affMatcherHelper->matchWithMaxTilt (im1, im2, keypoints1, keypoints2, matches, threshold, maxTilt);
    else
        // increase maxTilt while matching until a good result
        affMatcherHelper->matchIncreasingTilt (im1, im2, keypoints1, keypoints2, matches, threshold);


    // output
    Mat imgMatches;
    drawMatches (im1, keypoints1, im2, keypoints2, matches, imgMatches,
                 Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    imshow( "matches", imgMatches );
    waitKey(0);
    return 0;
}
