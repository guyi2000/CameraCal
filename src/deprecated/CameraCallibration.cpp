#include <iostream>
#include <fstream>
#include <string>
#include <ctime>
#include <cstdio>
#include <omp.h>
#include <filesystem>

#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include "ThreadPool.h"

using namespace cv;
using namespace std;

int main(int argc, char* argv[])
{
    filesystem::path p("./calibration");
    if (!exists(p)) exit(-1);
    vector<string> paths;
    for (auto& path : filesystem::directory_iterator(p)) {
        if (filesystem::is_regular_file(path)) {
            paths.emplace_back(path.path().u8string());
        }
    }
    size_t path_cnt = paths.size();

    TermCriteria criteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 30, 0.001);
    Size boardSize(8, 5);
    int squareSize = 1;
    vector<Point3f> objP;
    for (int i = 0; i < boardSize.height; i++) {
        for (int j = 0; j < boardSize.width; j++) {
            objP.emplace_back(Point3f(static_cast<float>(j * squareSize), static_cast<float>(i * squareSize), 0));
        }
    }
    vector<vector<Point3f> > objPoints;
    vector<vector<Point2f> > imgPoints;
    objPoints.reserve(path_cnt);
    imgPoints.reserve(path_cnt);
#pragma omp parallel for
    for (int i = 0; i < path_cnt; i++) {
        Mat img = imread(paths[i]);
        Mat gray, corners;
        cvtColor(img, gray, COLOR_BGR2GRAY);
        bool found = findChessboardCorners(gray, boardSize, corners, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);
        if (found) {
            cornerSubPix(gray, corners, Size(11, 11), Size(-1, -1), criteria);
#pragma omp critical
            {
                objPoints.emplace_back(objP);
                imgPoints.emplace_back(corners);
            }
        }
    }

    Mat cameraMatrix, distCoeffs;
    vector<Mat> rvecs, tvecs;
    calibrateCamera(objPoints, imgPoints, Size(6000, 4000), cameraMatrix, distCoeffs, rvecs, tvecs, 0, criteria);
    //calibrateCameraRO(objPoints, imgPoints, Size(6000, 4000), 3, cameraMatrix, distCoeffs, rvecs, tvecs, noArray(), 0, criteria);

    FileStorage fs("Camera_Calibration_result.xml", FileStorage::WRITE);
    write(fs, "cameraMatrix", cameraMatrix);
    write(fs, "distCoeffs", distCoeffs);
    fs.release();
    return 0;
}