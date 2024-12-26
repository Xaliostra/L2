#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>

using namespace std;
using namespace cv;

// Algorithm for boundary tracing (Voss method)
vector<Point> traceBoundary(const Mat& binaryImage, Point startPoint) {
    vector<Point> boundary;
    Point current = startPoint;
    Point next;

    // Search directions: clockwise
    const vector<Point> directions = { Point(0, -1), Point(1, -1), Point(1, 0), Point(1, 1),
                                       Point(0, 1), Point(-1, 1), Point(-1, 0), Point(-1, -1) };
    int dirIndex = 0; // Current direction index

    do {
        boundary.push_back(current);
        for (int i = 0; i < 8; ++i) {
            int idx = (dirIndex + i) % 8;
            next = current + directions[idx];
            if (next.x >= 0 && next.x < binaryImage.cols && next.y >= 0 && next.y < binaryImage.rows &&
                binaryImage.at<uchar>(next) == 255) {
                dirIndex = (idx + 6) % 8; // Turn back 6 directions
                current = next;
                break;
            }
        }
    } while (current != startPoint);

    return boundary;
}

// Curvature calculation at a point
float computeCurvature(const Point& prev, const Point& current, const Point& next) {
    Point2f v1 = current - prev;
    Point2f v2 = next - current;

    float norm1 = sqrt(v1.x * v1.x + v1.y * v1.y);
    float norm2 = sqrt(v2.x * v2.x + v2.y * v2.y);

    if (norm1 == 0 || norm2 == 0) return 0.0f;

    float dot = v1.x * v2.x + v1.y * v2.y;
    float det = v1.x * v2.y - v1.y * v2.x;
    float angle = atan2(det, dot);

    return fabs(angle);
}

int main() {
    // Load the image
    string imagePath = "E:\\test_image.jpg";

    Mat image = imread(imagePath);

    // Check if the image was successfully loaded
    if (image.empty()) {
        cerr << "Error: Unable to load the image." << endl;
        return -1;
    }

    // Convert the image to grayscale
    Mat grayImage;
    cvtColor(image, grayImage, COLOR_BGR2GRAY);

    // Threshold the image (binarization)
    Mat binaryImage;
    int thresholdValue = 128; // Threshold value for black and white
    threshold(grayImage, binaryImage, thresholdValue, 255, THRESH_BINARY);

    // Find the first white pixel to start boundary tracing
    Point startPoint(-1, -1);
    for (int y = 0; y < binaryImage.rows; ++y) {
        for (int x = 0; x < binaryImage.cols; ++x) {
            if (binaryImage.at<uchar>(y, x) == 255) {
                startPoint = Point(x, y);
                break;
            }
        }
        if (startPoint.x != -1) break;
    }

    if (startPoint.x == -1) {
        cerr << "Error: No shapes found in the image." << endl;
        return -1;
    }

    // Trace the boundary using the Voss algorithm
    vector<Point> boundary = traceBoundary(binaryImage, startPoint);

    // Calculate the perimeter of the contour
    double perimeter = 0.0;
    for (size_t i = 0; i < boundary.size(); ++i) {
        Point p1 = boundary[i];
        Point p2 = boundary[(i + 1) % boundary.size()];
        perimeter += norm(p1 - p2);
    }

    cout << "Perimeter of the contour: " << perimeter << endl;

    // Compute the centroid
    Moments m = moments(binaryImage, true);
    Point2f centroid(m.m10 / m.m00, m.m01 / m.m00);
    cout << "Centroid: (" << centroid.x << ", " << centroid.y << ")" << endl;

    // Calculate the eccentricity
    double a = m.mu20 / m.m00; // Second central moment
    double b = m.mu11 / m.m00;
    double c = m.mu02 / m.m00;
    double discriminant = sqrt(pow(a - c, 2) + 4 * pow(b, 2));
    double lambda1 = (a + c + discriminant) / 2; // Eigenvalue 1
    double lambda2 = (a + c - discriminant) / 2; // Eigenvalue 2
    double eccentricity = sqrt(1 - lambda2 / lambda1);
    cout << "Eccentricity: " << eccentricity << endl;

    // Evaluate the curvature at boundary points
    float minCurvature = FLT_MAX, maxCurvature = -FLT_MAX;
    Point minCurvaturePoint, maxCurvaturePoint;

    for (size_t i = 0; i < boundary.size(); ++i) {
        Point prev = boundary[(i + boundary.size() - 1) % boundary.size()];
        Point current = boundary[i];
        Point next = boundary[(i + 1) % boundary.size()];

        float curvature = computeCurvature(prev, current, next);

        if (curvature < minCurvature) {
            minCurvature = curvature;
            minCurvaturePoint = current;
        }
        if (curvature > maxCurvature) {
            maxCurvature = curvature;
            maxCurvaturePoint = current;
        }
    }

    cout << "Minimum curvature: " << minCurvature << " at point (" << minCurvaturePoint.x << ", " << minCurvaturePoint.y << ")" << endl;
    cout << "Maximum curvature: " << maxCurvature << " at point (" << maxCurvaturePoint.x << ", " << maxCurvaturePoint.y << ")" << endl;

    // Perform distance transform
    Mat distTransformResult;
    distanceTransform(binaryImage, distTransformResult, DIST_L2, 3);

    double minVal, maxVal;
    minMaxLoc(distTransformResult, &minVal, &maxVal);
    Mat normalizedDist;
    distTransformResult.convertTo(normalizedDist, CV_8U, 255.0 / maxVal);

    // Detect lines using Hough Transform
   /* vector<Vec4i> lines;
    HoughLinesP(binaryImage, lines, 1, CV_PI / 180, 50, 50, 10);*/

    // Visualization
    Mat resultImage = image.clone();
    for (size_t i = 0; i < boundary.size(); ++i) {
        line(resultImage, boundary[i], boundary[(i + 1) % boundary.size()], Scalar(0, 255, 0), 2);
    }
    circle(resultImage, centroid, 5, Scalar(0, 0, 255), -1); // Draw centroid

    // Draw the detected lines
    /*for (size_t i = 0; i < lines.size(); ++i) {
        line(resultImage, Point(lines[i][0], lines[i][1]), Point(lines[i][2], lines[i][3]), Scalar(255, 0, 0), 2);
    }*/

    // Display eccentricity on the image
    string eccentricityText = "Eccentricity: " + to_string(eccentricity);
    putText(resultImage, eccentricityText, Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 0, 0), 2);

    // Display points with minimum and maximum curvature
    circle(resultImage, minCurvaturePoint, 5, Scalar(255, 0, 0), -1); // Point of minimum curvature
    circle(resultImage, maxCurvaturePoint, 5, Scalar(0, 255, 255), -1); // Point of maximum curvature

    // Display distance transform
    imshow("Distance Transform", normalizedDist);
    imshow("Result", resultImage);
    waitKey(0);

    return 0;
}
