//
//  main.cpp
//  sift_asift_match
//
//  Created by willard on 8/18/15.
//  Copyright (c) 2015 wilard. All rights reserved.
//

#include "ASifttDetector.h"
#include "utils.h"

int main(int argc, const char * argv[]) {

    string imgfn = "./pic/DSC_2625_resize.JPG";
    string objFileName = "./pic/DSC_2624_resize.JPG";
    Mat queryImage, objectImage;
    queryImage = imread(imgfn);
    objectImage = imread(objFileName);

    ASifttDetector asiftDetector;
    vector<KeyPoint> asiftKeypoints_query, asiftKeypoints_object;
    Mat asiftDescriptors_query, asiftDescriptors_object;
    asiftDetector.detectAndCompute(queryImage, asiftKeypoints_query, asiftDescriptors_query);
    asiftDetector.detectAndCompute(objectImage, asiftKeypoints_object, asiftDescriptors_object);
    
    //Matching descriptor vectors using FLANN matcher, Asift找匹配点
    vector<vector<DMatch>> matches;
    std::vector< DMatch > asiftMatches;
    FlannBasedMatcher matcher;

    matcher.match(asiftDescriptors_query, asiftDescriptors_object, asiftMatches);
    findInliers(asiftKeypoints_query, asiftKeypoints_object, asiftMatches, imgfn, objFileName);
    
    // 使用内置函数画匹配点对
    Mat img_matches;
    drawMatches(queryImage, asiftKeypoints_query, objectImage, asiftDescriptors_object,
                asiftMatches, img_matches, Scalar::all(-1), Scalar::all(-1),
                vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    imshow("Good Matches & Object detection", img_matches);
    waitKey(0);

    return 0;
}
