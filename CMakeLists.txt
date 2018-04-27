cmake_minimum_required(VERSION 2.8.4)
project(Aorb_sift)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set( Boost_USE_STATIC_LIBS OFF )
set( Boost_USE_STATIC_RUNTIME OFF )
find_package( Boost REQUIRED COMPONENTS system filesystem )
find_package( OpenCV REQUIRED COMPONENTS opencv_core opencv_imgproc opencv_features2d opencv_nonfree opencv_video opencv_flann opencv_calib3d opencv_highgui)


include_directories(src)



set(ORB_HEADERS
        src/orb/AOrbtDetector.cpp
        src/orb/AOrbtDetector.h)

set(SIFT_HEADERS
        src/sift/ASifttDetector.cpp
        src/sift/ASifttDetector.h)

set(ERIE_SRC_FILES
        src/utils.cpp
        src/utils.h)

add_library(demo ${ERIE_SRC_FILES} )

add_executable(Aorb_demo src/orb/aorb_main.cpp ${ORB_HEADERS})
target_link_libraries(Aorb_demo demo ${Boost_FILESYSTEM_LIBRARY} ${Boost_SYSTEM_LIBRARY} ${OpenCV_LIBS} )

add_executable(Asift_demo src/sift/asift_main.cpp ${SIFT_HEADERS})
target_link_libraries(Asift_demo demo ${Boost_FILESYSTEM_LIBRARY} ${Boost_SYSTEM_LIBRARY} ${OpenCV_LIBS} )


