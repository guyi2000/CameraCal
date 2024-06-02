#define _USE_MATH_DEFINES
#include <iostream>
#include <fstream>
#include <string>
#include <ctime>
#include <cstdio>
#include <cmath>
#include <algorithm>

#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include "calib3d/calibinit.cpp"
#include "calib3d/circlesgrid.cpp"
#include "calib3d/checkchessboard.cpp"

#define MIN_DILATIONS 0
#define MAX_DILATIONS 0

using namespace cv;
using namespace std;

typedef struct CalPointPair {
    Point2d srcPoint;
    Point2d dstPoint;
    int srcPointIndex;
    int dstPointIndex;
    double distance;
} CalPointPair;

typedef struct PointPair {
    Point2d srcPoint;
    Point2d dstPoint;
} PointPair;

double CalPairScale(vector<PointPair> pairList) {
    double temp = 0;
    for (int i = 0; i < pairList.size(); i++) {
        temp = temp + pow(pairList[i].dstPoint.x - pairList[i].srcPoint.x, 2) + pow(pairList[i].dstPoint.y - pairList[i].srcPoint.y, 2);
    }
    return pow(temp, 0.5);
}

void findCenterOfAllChessBoard(Mat img, vector<Point2d>& outputCenters) {
    outputCenters.clear();
    cvtColor(img, img, COLOR_BGR2GRAY); // To GRAY
    Mat thresh_img_new = img.clone();
    icvBinarizationHistogramBased(thresh_img_new);
    Size pattern_size(2, 2); // five quads and four corners and one inner quad
    ChessBoardDetector detector(pattern_size); // use for find quads

    for (int dilations = MIN_DILATIONS; dilations <= MAX_DILATIONS; dilations++) {
        dilate(thresh_img_new, thresh_img_new, Mat(), Point(-1, -1), 1); // dilate the chessboard
        rectangle(
            thresh_img_new,
            Point(0, 0),
            Point(thresh_img_new.cols - 1, thresh_img_new.rows - 1),
            Scalar(255, 255, 255), 3, LINE_8); // the inverse dilate of border
        detector.reset();
        Mat binarized_img = thresh_img_new;
        imwrite("1.png", binarized_img);
        detector.generateQuads(binarized_img, 0); // find quads
        size_t max_quad_buf_size = detector.all_quads.size();
        detector.findQuadNeighbors(); // check neighbour

        std::vector<ChessBoardQuad*> quad_group;
        std::vector<ChessBoardCorner*> corner_group;
        corner_group.reserve(max_quad_buf_size * 4);

        for (int group_idx = 0; ; group_idx++) {
            detector.findConnectedQuads(quad_group, group_idx);
            if (quad_group.empty()) break;
            int count = (int)quad_group.size();
            count = detector.orderFoundConnectedQuads(quad_group);
            if (count == 0) continue;
            count = detector.cleanFoundConnectedQuads(quad_group);
            count = detector.checkQuadGroup(quad_group, corner_group);
            int n = count > 0 ? pattern_size.width * pattern_size.height : -count;
            n = min(n, pattern_size.width * pattern_size.height);

            if (count > 0) {
                vector<Point2f> out_corners;
                double center_x = 0, center_y = 0;

                for (int i = 0; i < n; ++i)
                    out_corners.emplace_back(corner_group[i]->pt);

                cornerSubPix(img, out_corners, Size(2, 2), Size(-1, -1),
                    TermCriteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 15, 0.001));

                for (int i = 0; i < n; ++i) {
                    center_x += out_corners[i].x;
                    center_y += out_corners[i].y;
                }

                outputCenters.emplace_back(Point2d{ center_x / n, center_y / n });
            }
        }
    }
}

void generatePointPair(vector<Point2d>& srcPoint, vector<Point2d>& dstPoint, vector<PointPair>& outputPair) {
    outputPair.clear();
    vector<CalPointPair> tempPair;
    if (srcPoint.size() != dstPoint.size()) return;
    for (int i = 0; i < srcPoint.size(); i++) {
        for (int j = 0; j < dstPoint.size(); j++) {
            tempPair.emplace_back(CalPointPair{
                srcPoint[i], dstPoint[j], i, j,
                sqrt(
                    (srcPoint[i].x - dstPoint[j].x) * (srcPoint[i].x - dstPoint[j].x) +
                    (srcPoint[i].y - dstPoint[j].y) * (srcPoint[i].y - dstPoint[j].y)
                )
                });
        }
    }
    sort(
        tempPair.begin(), tempPair.end(),
        [](CalPointPair& p1, CalPointPair& p2) -> bool
        {
            return p1.distance < p2.distance;
        }
    );
    auto srcflag = new bool[srcPoint.size()];
    auto dstflag = new bool[srcPoint.size()];
    memset(srcflag, 0, sizeof(bool) * srcPoint.size());
    memset(dstflag, 0, sizeof(bool) * srcPoint.size());
    for (auto& p : tempPair) {
        if (!(srcflag[p.srcPointIndex] || dstflag[p.dstPointIndex])) {
            outputPair.emplace_back(PointPair{ p.srcPoint, p.dstPoint });
            srcflag[p.srcPointIndex] = true;
            dstflag[p.dstPointIndex] = true;
        }
    }
    delete[] srcflag;
    delete[] dstflag;
}

void CalMoveAndRotate(
    Mat& InputMat,
    double InputDeltaX,
    double InputDeltaY,
    double InputDeltaZ,
    double InputAlpha,
    double InputBeta,
    double InputGamma,
    Mat& OutputMat
) {
    Mat Mt = Mat::eye(Size(4, 4), CV_64F);
    Mt.at<double>(3, 0) = InputDeltaX;
    Mt.at<double>(3, 1) = InputDeltaY;
    Mt.at<double>(3, 2) = InputDeltaZ;
    OutputMat = InputMat * Mt;

    Mt = Mat::eye(Size(4, 4), CV_64F);
    Mt.at<double>(0, 0) = cos(InputGamma) * cos(InputBeta);
    Mt.at<double>(1, 0) = -sin(InputGamma) * cos(InputAlpha) + cos(InputGamma) * sin(InputBeta) * sin(InputAlpha);
    Mt.at<double>(2, 0) = sin(InputGamma) * sin(InputAlpha) + cos(InputGamma) * sin(InputBeta) * cos(InputAlpha);
    Mt.at<double>(0, 1) = sin(InputGamma) * cos(InputBeta);
    Mt.at<double>(1, 1) = cos(InputGamma) * cos(InputAlpha) + sin(InputGamma) * sin(InputBeta) * sin(InputAlpha);
    Mt.at<double>(2, 1) = -cos(InputGamma) * sin(InputAlpha) + sin(InputGamma) * sin(InputBeta) * cos(InputAlpha);
    Mt.at<double>(0, 2) = -sin(InputBeta);
    Mt.at<double>(1, 2) = cos(InputBeta) * sin(InputAlpha);
    Mt.at<double>(2, 2) = cos(InputBeta) * cos(InputAlpha);
    OutputMat = OutputMat * Mt;
}

void CalRotateAndMove(
    Mat& InputMat,
    double InputDeltaX,
    double InputDeltaY,
    double InputDeltaZ,
    double InputAlpha,
    double InputBeta,
    double InputGamma,
    Mat& OutputMat
) {
    Mat Mt = Mat::eye(Size(4, 4), CV_64F);
    Mt.at<double>(0, 0) = cos(InputGamma) * cos(InputBeta);
    Mt.at<double>(1, 0) = -sin(InputGamma) * cos(InputAlpha) + cos(InputGamma) * sin(InputBeta) * sin(InputAlpha);
    Mt.at<double>(2, 0) = sin(InputGamma) * sin(InputAlpha) + cos(InputGamma) * sin(InputBeta) * cos(InputAlpha);
    Mt.at<double>(0, 1) = sin(InputGamma) * cos(InputBeta);
    Mt.at<double>(1, 1) = cos(InputGamma) * cos(InputAlpha) + sin(InputGamma) * sin(InputBeta) * sin(InputAlpha);
    Mt.at<double>(2, 1) = -cos(InputGamma) * sin(InputAlpha) + sin(InputGamma) * sin(InputBeta) * cos(InputAlpha);
    Mt.at<double>(0, 2) = -sin(InputBeta);
    Mt.at<double>(1, 2) = cos(InputBeta) * sin(InputAlpha);
    Mt.at<double>(2, 2) = cos(InputBeta) * cos(InputAlpha);
    OutputMat = InputMat * Mt;

    Mt = Mat::eye(Size(4, 4), CV_64F);
    Mt.at<double>(3, 0) = InputDeltaX;
    Mt.at<double>(3, 1) = InputDeltaY;
    Mt.at<double>(3, 2) = InputDeltaZ;
    OutputMat = OutputMat * Mt;
}

void CalImgCoord(
    Mat& InputCameraMatrix,
    Mat& InputWorldCoord,
    double InputCameraX,
    double InputCameraY,
    double InputCameraZ,
    double InputExTheta,
    double InputExAlpha,
    Mat& InputDistortionMatrix,
    vector<Point2d>& OutputImgCoord
) {
    OutputImgCoord.clear();

    Mat cCoord;
    CalMoveAndRotate(
        InputWorldCoord, -InputCameraX, InputCameraZ,
        -InputCameraY, InputExAlpha, InputExTheta - M_PI_2, 0, cCoord
    ); // consider the tranform of revit

    double k1, k2, p1, p2, k3;
    k1 = InputDistortionMatrix.at<double>(0, 0);
    k2 = InputDistortionMatrix.at<double>(0, 1);
    p1 = InputDistortionMatrix.at<double>(0, 2);
    p2 = InputDistortionMatrix.at<double>(0, 3);
    k3 = InputDistortionMatrix.at<double>(0, 4);

    double x_, y_, r2;
    double x__, y__;
    double u, v;
    for (int i = 0; i < cCoord.rows; i++) {

        x_ = cCoord.at<double>(i, 0) / cCoord.at<double>(i, 2);
        y_ = cCoord.at<double>(i, 1) / cCoord.at<double>(i, 2);
        r2 = pow(x_, 2) + pow(y_, 2);

        x__ = x_ * (1 + k1 * r2 + k2 * pow(r2, 2) + k3 * pow(r2, 3)) + 2 * p1 * x_ * y_ + p2 * (r2 + 2 * pow(x_, 2));
        y__ = y_ * (1 + k1 * r2 + k2 * pow(r2, 2) + k3 * pow(r2, 3)) + 2 * p2 * x_ * y_ + p1 * (r2 + 2 * pow(y_, 2));

        u = InputCameraMatrix.at<double>(0, 0) * x__ + InputCameraMatrix.at<double>(0, 2);
        v = InputCameraMatrix.at<double>(1, 1) * y__ + InputCameraMatrix.at<double>(1, 2);

        OutputImgCoord.emplace_back(Point2d{ u,v });
    }
}

void getDeltaBase(
    Mat& InputCameraMatrix,
    Mat& InputWorldCoord,
    double InputCameraX,
    double InputCameraY,
    double InputCameraZ,
    double InputExTheta,
    double InputExAlpha,
    Mat& InputDistortionMatrix,
    double InputDeltaX,
    double InputDeltaY,
    double InputDeltaZ,
    double InputDeltaAlpha,
    double InputDeltaBeta,
    double InputDeltaGamma,
    Point3d& InputOriginPoint,
    vector<vector<Point2d> >& OutputDeltaCoord
) {
    for (auto& o : OutputDeltaCoord) o.clear();
    OutputDeltaCoord.clear();

    vector<Point2d> origin;
    CalImgCoord(
        InputCameraMatrix, InputWorldCoord, InputCameraX,
        InputCameraY, InputCameraZ, InputExTheta,
        InputExAlpha, InputDistortionMatrix, origin
    );
    OutputDeltaCoord.emplace_back(origin);

    vector<Point2d> delta;
    CalImgCoord(
        InputCameraMatrix, InputWorldCoord, InputCameraX - InputDeltaX,
        InputCameraY, InputCameraZ, InputExTheta,
        InputExAlpha, InputDistortionMatrix, delta
    );  // if the camera move -dx, then the true structure move dx instead.
    for (int i = 0; i < delta.size(); i++) delta[i] -= origin[i];
    OutputDeltaCoord.emplace_back(delta);

    CalImgCoord(
        InputCameraMatrix, InputWorldCoord, InputCameraX,
        InputCameraY - InputDeltaY, InputCameraZ, InputExTheta,
        InputExAlpha, InputDistortionMatrix, delta
    );
    for (int i = 0; i < delta.size(); i++) delta[i] -= origin[i];
    OutputDeltaCoord.emplace_back(delta);

    CalImgCoord(
        InputCameraMatrix, InputWorldCoord, InputCameraX,
        InputCameraY, InputCameraZ - InputDeltaZ, InputExTheta,
        InputExAlpha, InputDistortionMatrix, delta
    );
    for (int i = 0; i < delta.size(); i++) delta[i] -= origin[i];
    OutputDeltaCoord.emplace_back(delta);

    Mat changeOriginAns;
    CalMoveAndRotate(
        InputWorldCoord, -InputOriginPoint.x,
        -InputOriginPoint.y, -InputOriginPoint.z,
        0, 0, 0, changeOriginAns
    );

    Mat rotateAns;
    CalMoveAndRotate(
        changeOriginAns, 0, 0, 0, InputDeltaAlpha, 0, 0, rotateAns
    );

    CalMoveAndRotate(
        rotateAns, InputOriginPoint.x,
        InputOriginPoint.y, InputOriginPoint.z,
        0, 0, 0, rotateAns
    );

    CalImgCoord(
        InputCameraMatrix, rotateAns, InputCameraX,
        InputCameraY, InputCameraZ, InputExTheta,
        InputExAlpha, InputDistortionMatrix, delta
    );
    for (int i = 0; i < delta.size(); i++) delta[i] -= origin[i];
    OutputDeltaCoord.emplace_back(delta);

    CalMoveAndRotate(
        changeOriginAns, 0, 0, 0, 0, InputDeltaBeta, 0, rotateAns
    );

    CalMoveAndRotate(
        rotateAns, InputOriginPoint.x,
        InputOriginPoint.y, InputOriginPoint.z,
        0, 0, 0, rotateAns
    );

    CalImgCoord(
        InputCameraMatrix, rotateAns, InputCameraX,
        InputCameraY, InputCameraZ, InputExTheta,
        InputExAlpha, InputDistortionMatrix, delta
    );
    for (int i = 0; i < delta.size(); i++) delta[i] -= origin[i];
    OutputDeltaCoord.emplace_back(delta);

    CalMoveAndRotate(
        changeOriginAns, 0, 0, 0, 0, 0, InputDeltaGamma, rotateAns
    );

    CalMoveAndRotate(
        rotateAns, InputOriginPoint.x,
        InputOriginPoint.y, InputOriginPoint.z,
        0, 0, 0, rotateAns
    );

    CalImgCoord(
        InputCameraMatrix, rotateAns, InputCameraX,
        InputCameraY, InputCameraZ, InputExTheta,
        InputExAlpha, InputDistortionMatrix, delta
    );
    for (int i = 0; i < delta.size(); i++) delta[i] -= origin[i];
    OutputDeltaCoord.emplace_back(delta);
}

void CalculateOffset(
    vector<vector<Point2d> >& InputDeltaCoord,
    vector<PointPair>& InputPointPair,
    Mat& OutputAns
) {
    Mat Base(2 * (int)InputDeltaCoord[0].size(), (int)InputDeltaCoord.size() - 1, CV_64F);
    for (int i = 1; i < InputDeltaCoord.size(); i++) {
        for (int j = 0; j < InputDeltaCoord[i].size(); j++) {
            Base.at<double>(2 * j, i - 1) = InputDeltaCoord[i][j].x;
            Base.at<double>(2 * j + 1, i - 1) = InputDeltaCoord[i][j].y;
        }
    }

    Point2d temp;
    Mat Diff(2 * (int)InputPointPair.size(), 1, CV_64F);
    for (int i = 0; i < InputPointPair.size(); i++) {
        temp = InputPointPair[i].dstPoint - InputPointPair[i].srcPoint;
        Diff.at<double>(2 * i, 0) = temp.x;
        Diff.at<double>(2 * i + 1, 0) = temp.y;
    }
    OutputAns = (Base.t() * Base).inv() * Base.t() * Diff;
    Diff = Diff - Base * OutputAns;
    for (int i = 0; i < InputPointPair.size(); i++) {
        temp.x = Diff.at<double>(2 * i, 0);
        temp.y = Diff.at<double>(2 * i + 1, 0);
        InputPointPair[i].dstPoint = InputPointPair[i].srcPoint + temp;
    }
}

int main(int argc, char* argv[])
{
    Mat cameraMatrix, distCoeffs;
    FileStorage fs("Camera_Calibration_result.xml", FileStorage::READ);
    fs["cameraMatrix"] >> cameraMatrix;
    fs["distCoeffs"] >> distCoeffs;
    fs.release();
    cout << "=== Start reading camera calibration data ===" << endl 
         << "Camera Matrix:" << endl << cameraMatrix << endl 
         << "DistCoeffs:" << endl << distCoeffs << endl 
         << "===  End reading camera calibration data  ===" << endl << endl;

    vector<Point3d> revitPointsData;
    ifstream revitPoints;
    revitPoints.open("points.txt");
    string type;
	int cnt;
    double x, y, z;
    Point3d OriginPoint;
    while (revitPoints >> type) {
        if (!type.compare("Point")) {
			revitPoints >> cnt >> x >> y >> z;
			revitPointsData.emplace_back(Point3d{ x,y,z });
		} else if (!type.compare("ExPoint")) {
			revitPoints >> cnt >> OriginPoint.x >> OriginPoint.z >> OriginPoint.y;
		}
    }
    OriginPoint.y = -OriginPoint.y;
    revitPoints.close();

    Mat WCoord(Size(4, static_cast<int>(revitPointsData.size())), CV_64F);
    for (int i = 0; i < revitPointsData.size(); i++) {
        WCoord.at<double>(i, 0) = revitPointsData[i].x;
        WCoord.at<double>(i, 1) = -revitPointsData[i].z;
        WCoord.at<double>(i, 2) = revitPointsData[i].y;
        WCoord.at<double>(i, 3) = 1.0;
    }
    cout << "=== Reading points from Revit ===" << endl
        << "WCoord:" << endl << WCoord << endl
        << "OriginalPoint:" << endl << OriginPoint << endl << endl;

    vector<Point2d> imgCoord;
    CalImgCoord(cameraMatrix, WCoord, 268.8, -190.4, 61, 138.0 / 180 * M_PI, 0, distCoeffs, imgCoord);
    cout << "=== Calculate imgCoord ===" << endl
        << "imgCoord:" << endl << imgCoord << endl << endl;

    Mat img = imread("1.jpg");
    vector<Point2d> out_centers;
    findCenterOfAllChessBoard(img, out_centers);
    cout << "=== Finding points in true picture ===" << endl
        << "Centers:" << endl;
    for (auto& p : out_centers) cout << p << endl;
    cout << endl;

    vector<PointPair> pointPair;
    generatePointPair(out_centers, imgCoord, pointPair);
    cout << "=== Generating point pairs ===" << endl
        << "Point Pair:" << endl
        << "SrcPoint\t\tDstPoint" << endl;
    for (auto& p : pointPair) cout << p.srcPoint << "\t" << p.dstPoint << endl;

    cout << endl;
    cnt = 0; cnt++;
    
    vector<vector<Point2d> > deltaCoord;
    getDeltaBase(cameraMatrix, WCoord, 268.8, -190.4, 61, 138.0 / 180 * M_PI, 0, distCoeffs, 0.1, 0.1, 0.1, 1e-5, 1e-5, 1e-5, OriginPoint, deltaCoord);

    vector<PointPair> DiffVec;
    DiffVec = pointPair;
    double Diff = CalPairScale(DiffVec);

    Mat Ans;
    CalculateOffset(deltaCoord, DiffVec, Ans);

    Mat Adjustment = Ans.clone();
    Mat tempWCoord = WCoord;
    cout << "=== Start Calculate Diff and Adjustment ===" << endl
        << "Step " << cnt << ":" << endl
        << "Adjustment:" << endl << Adjustment << endl
        << "Ans:" << endl << Ans << endl
        << "Diff:" << Diff << endl << endl;

    while (Diff > 88) {
        cnt++;
        CalMoveAndRotate(
            tempWCoord,
            -0.1 * Ans.at<double>(0, 0),
            -0.1 * Ans.at<double>(1, 0),
            -0.1 * Ans.at<double>(2, 0),
            -1e-5 * Ans.at<double>(3, 0),
            -1e-5 * Ans.at<double>(4, 0),
            -1e-5 * Ans.at<double>(5, 0),
            tempWCoord
        );
        getDeltaBase(cameraMatrix, tempWCoord, 268.8, -190.4, 61, 138 / 180 * M_PI, 0, distCoeffs, 0.1, 0.1, 0.1, 1e-5, 1e-5, 1e-5, OriginPoint, deltaCoord);
        CalculateOffset(deltaCoord, DiffVec, Ans);
        
        Diff = CalPairScale(DiffVec);
        Adjustment += Ans;
        cout << "Step " << cnt << ":" << endl 
             << "Adjustment:" << endl << Adjustment << endl 
             << "Ans:" << endl << Ans << endl 
             << "Diff:" << Diff << endl << endl;
    }
    Adjustment.at<double>(0, 0) = -0.1 * Adjustment.at<double>(0, 0);
    Adjustment.at<double>(1, 0) = -0.1 * Adjustment.at<double>(1, 0);
    Adjustment.at<double>(2, 0) = -0.1 * Adjustment.at<double>(2, 0);
    Adjustment.at<double>(3, 0) = -1e-5 * Adjustment.at<double>(3, 0);
    Adjustment.at<double>(4, 0) = -1e-5 * Adjustment.at<double>(4, 0);
    Adjustment.at<double>(5, 0) = -1e-5 * Adjustment.at<double>(5, 0);
    cout << "===  End Calculate Diff and Adjustment  ===" << endl << endl;

    cout << "Final adjustment:" << endl << Adjustment << endl;
    return 0;
}