//
//  main.cpp
//  ORB_aORB_match
//
//  Created by willard on 8/18/15.
//  Copyright (c) 2015 wilard. All rights reserved.
//

#include "AOrbtDetector.h"
#include "utils.h"

int main(int argc, const char * argv[]) {

    string imgfn = "./pic/DSC_2625_resize.JPG";
    string objFileName = "./pic/DSC_2624_resize.JPG";
    Mat queryImage, objectImage;
    queryImage = imread(imgfn);
    objectImage = imread(objFileName);

    AOrbtDetector aorbDetector;
    vector<KeyPoint> aorbKeypoints_query, aorbKeypoints_object;
    Mat aorbDescriptors_query, aorbDescriptors_object;
    aorbDetector.detectAndCompute(queryImage, aorbKeypoints_query, aorbDescriptors_query);
    aorbDetector.detectAndCompute(objectImage, aorbKeypoints_object, aorbDescriptors_object);
    
    //Matching descriptor vectors using FLANN matcher, Aorb找匹配点
    vector<vector<DMatch>> matches;
    std::vector< DMatch > aorbMatches;
    BFMatcher matcher(NORM_HAMMING);

    matcher.match(aorbDescriptors_query, aorbDescriptors_object, aorbMatches);
    findInliers(aorbKeypoints_query, aorbKeypoints_object, aorbMatches, imgfn, objFileName);
    
    // 使用内置函数画匹配点对
    Mat img_matches;
    drawMatches(queryImage, aorbKeypoints_query, objectImage, aorbDescriptors_object,
                aorbMatches, img_matches, Scalar::all(-1), Scalar::all(-1),
                vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    imshow("Good Matches & Object detection", img_matches);
    waitKey(0);

    return 0;
}
