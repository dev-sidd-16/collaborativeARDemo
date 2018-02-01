#include <jni.h>
#include <iostream>
#include <string>
#include <sys/time.h>

#include <android/log.h>
#include <android/bitmap.h>
#include <android/native_window_jni.h>
#include <android/native_window.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>

#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "myrobot", __VA_ARGS__)

using namespace cv;
using namespace std;

int numFeautresReference = 500;
int numFeaturesDest = 500;

extern "C"
JNIEXPORT jstring

JNICALL
Java_com_example_siddprakash_collaborativeardemo_MainActivity_stringFromJNI(
        JNIEnv *env,
        jobject /* this */,
        jint srcWidth, jint srcHeight,
        jobject srcBuffer, jlong address,
        jobject dstSurface, jboolean capture) {

    string hello = "--*--\n";
    uint8_t *srcLumaPtr = reinterpret_cast<uint8_t *>(env->GetDirectBufferAddress(srcBuffer));

    int dstWidth;
    int dstHeight;

    Mat &mYuv = *(Mat*)address;
    Mat mYuvGray;
    cvtColor(mYuv, mYuvGray, CV_YUV2GRAY_420);
//    Mat mYuv(srcHeight+srcHeight/2, srcWidth, CV_8UC1, srcLumaPtr, 1280); //getting all channels for display
//    cv::Mat mYuvGray(srcHeight,srcWidth,CV_8UC1,srcLumaPtr); //only getting the luma channel

//    uint8_t *srcChromaUVInterleavedPtr = nullptr;
//    bool swapDstUV;

    ANativeWindow *win = ANativeWindow_fromSurface(env, dstSurface);
    ANativeWindow_acquire(win);

    ANativeWindow_Buffer buf;
    dstWidth = srcHeight;
    dstHeight = srcWidth;


    ANativeWindow_setBuffersGeometry(win, dstWidth, dstHeight, 0 );

    //acquiring a lock on the SurfaceView to render the processed image
    if (int32_t err = ANativeWindow_lock(win, &buf, NULL)) {
        LOGE("ANativeWindow_lock failed with error code %d\n", err);
        ANativeWindow_unlockAndPost(win);
        ANativeWindow_release(win);
        return env->NewStringUTF(hello.c_str());;
    }

    uint8_t *dstLumaPtr = reinterpret_cast<uint8_t *>(buf.bits);

    Mat dstRgba(dstHeight, buf.stride, CV_8UC4, dstLumaPtr);
    Mat srcRgba(srcHeight, srcWidth, CV_8UC4);
    Mat flipRgba(dstHeight, dstWidth, CV_8UC4);
    Mat colorRgba(dstHeight, dstWidth, CV_8UC4);

    // convert YUV -> RGBA
    cv::cvtColor(mYuv, srcRgba, CV_YUV2RGBA_NV21); //colorRgba is used for display
//    srcRgba = mYuv;
    imwrite("/mnt/sdcard/Android/Data/CollaborativeAR/read.jpg", srcRgba);

    Mat ref = imread("/mnt/sdcard/Android/Data/CollaborativeAR/marker.jpg");
    Rect roi;
    Mat crop;

    /*
    cvtColor(img, mGray, CV_BGR2RGB);
    Mat img = imread("/mnt/sdcard/Android/Data/CollabAR/image1.jpg");
    resize(img, img, Size(), 0.25, 0.25);
    imwrite( "/mnt/sdcard/Android/Data/CollabAR/image1.jpg", img );
     */

    //transpose(mYuvGray, mYuvGray);
    //flip(mYuvGray, mYuvGray,1);

    Mat imgG = mYuvGray;

//    int width;
//    width = imgG.cols;
//
//    int h, w;
//    h = ref.rows;
//    w = ref.cols;
//
//    float r = width/float(w);
//
//    resize(ref, ref, Size(width, int(h*r)), 0, 0, INTER_AREA);

    // convert images to grayscale
    Mat refG;
//    cvtColor(img, imgG, CV_YUV2GRAY_420);
    cvtColor(ref, refG, CV_RGB2GRAY);

    if( !imgG.data || !refG.data )
    {
        cout<< " --(!) Error reading images " << endl;
    }

    imwrite("/mnt/sdcard/Android/Data/CollaborativeAR/readGray.jpg", imgG);

    /*
    imwrite( "/mnt/sdcard/Android/Data/CollabAR/grayIMG.jpg", imgG );
    imwrite( "/mnt/sdcard/Android/Data/CollabAR/grayREF.jpg", refG );
     */

    const double cxIMG = imgG.cols/2;
    const double cyIMG = imgG.rows/2;
    const double fxIMG = 1.73*cxIMG;
    const double fyIMG = 1.73*cyIMG;
    const double standHeight = 1/5;


    // TODO: Move reference frame parameter estimation code to another function
    const double cxREF = refG.cols/2;
    const double cyREF = refG.rows/2;
    const double fxREF = cxREF*1.73;
    const double fyREF = cyREF*1.73;
    const double radius = 0.53/5;

    struct timeval start, end, diff;
    ::gettimeofday(&start, NULL);

    //-- Step 1: Detect the keypoints using SURF Detector

    Ptr<FeatureDetector> detector = ORB::create(numFeaturesDest);
    vector<KeyPoint> keypoints_1, keypoints_2;
    detector->detect(imgG, keypoints_1);
    detector->detect(refG, keypoints_2);

    //-- Step 2: Calculate descriptors (feature vectors)
    Mat descriptors_1, descriptors_2;
    detector->compute(imgG, keypoints_1, descriptors_1);
    detector->compute(refG, keypoints_2, descriptors_2);

    //-- Step 3: Matching descriptor vectors using FLANN matcher
    FlannBasedMatcher flannBasedMatcher(new flann::LshIndexParams(20,10,2));
    vector<vector< DMatch > >matches;
    flannBasedMatcher.knnMatch(descriptors_1,descriptors_2,matches,2);

    double max_X = 0; double min_X = 100;
    double max_Y = 0; double min_Y = 100;

    vector<Point3f>  p3d;
    for (int i = 0; i< keypoints_2.size(); i++) {
        double x = keypoints_2[i].pt.x;	// 2D location in image
        double y = keypoints_2[i].pt.y;
        double X = (x - cxREF)/cxREF;
        if( X < min_X )
            min_X = X;
        if( X > max_X )
            max_X = X;
        double Y = ((y - cyREF)/cxREF);
        if( Y < min_Y )
            min_Y = Y;
        if( Y > max_Y )
            max_Y = Y;
        double Z = (radius+standHeight);
        p3d.push_back(cv::Point3f(X, Y, Z));
    }

    double max_dist = 0; double min_dist = 100;

    //-- Quick calculation of max and min distances between keypoints
    for( int i = 0; i < descriptors_1.rows; i++ )
    {
        double dist = matches[i][1].distance;
        if( dist < min_dist )
            min_dist = dist;
        if( dist > max_dist )
            max_dist = dist;
    }

    printf("-- Max dist : %f \n", max_dist );
    printf("-- Min dist : %f \n", min_dist );

    //-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist,
    //-- or a small arbitary value ( 0.02 ) in the event that min_dist is very
    //-- small)
    //-- PS.- radiusMatch can also be used here.
    vector< DMatch > good_matches;

    for( int i = 0; i < descriptors_1.rows; i++ )
    {
        if( matches[i][0].distance <= max(2*min_dist, 0.02) )
        {
            good_matches.push_back( matches[i][0]);
        }
    }

    //-- Draw only "good" matches
    Mat img_matches;
    drawMatches( imgG, keypoints_1, refG, keypoints_2,
                 good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
                 vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

    //-- Show detected matches
    if(capture) {
        imwrite("/mnt/sdcard/Android/Data/CollaborativeAR/Good_Matches.jpg", img_matches);
    }

    int nGoodMatches = good_matches.size();

    stringstream ss;
    ss << nGoodMatches;
    //hello = hello + "No. of Good Matches: " + ss.str();

    if(nGoodMatches < 4){
        hello = hello + "Not enough good matches to estimate pose";
    } else{

        vector<Point2f> pts1(nGoodMatches);
        vector<Point2f> pts2(nGoodMatches);
        for (size_t i = 0; i < nGoodMatches; i++) {
            pts1[i] = keypoints_2[good_matches[i].trainIdx].pt;
            pts2[i] = keypoints_1[good_matches[i].queryIdx].pt;
        }

        vector<unsigned char> inliersMask(pts1.size());
        Mat homography = findHomography(pts1, pts2, FM_RANSAC, 5, inliersMask);

        vector<DMatch> inliers;
        for (size_t i = 0; i < inliersMask.size(); i++) {
            if (inliersMask[i])
                inliers.push_back(good_matches[i]);
        }
        ss.str(string());
        ss << inliers.size();
        //hello = hello + "\nNo. of inliers from homography calculation: " + ss.str();

        if (inliers.size() >5) {

            vector<Point2f> p2D;
            vector<Point3f> p3D;
            for (unsigned int i = 0; i < inliers.size(); i++) {

                int i1 = inliers[i].trainIdx;
                p3D.push_back(p3d[i1]);
                int i2 = inliers[i].queryIdx;
                p2D.push_back(keypoints_1[i2].pt);
            }

            double data[9] = {fxIMG, 0, cxIMG,
                              0, fyIMG, cyIMG,
                              0, 0, 1};

            //make the camera intrinsic parameters matrix
            Mat K = Mat(3, 3, CV_64F, data);
            Mat rotVec, transVec;
            bool foundPose = solvePnPRansac(p3D, p2D, K, Mat::zeros(5, 1, CV_64F), rotVec, transVec);
            if (foundPose){
                //hello = hello + "\nPose Found! |";
                ss.str(string());
                ss << rotVec.t();
                hello = hello + "Rotation Vector: " + ss.str();
                ss.str(string());
                ss << transVec.t();
                hello = hello + "\nTranslation Vector: " + ss.str();
                double distance = norm(transVec);
                ss.str(string());
                ss << distance;
                hello = hello + "\nDistance: " + ss.str();
                const double rad = ((radius-standHeight)/(2*distance))*fxIMG;
                ss.str(string());
                ss << rad;
                hello = hello + " | Radius: " + ss.str();

                Mat rotVecFull = Mat(3, 3, CV_64F);
                Rodrigues(rotVec, rotVecFull);

                Mat extrinsic;
                hconcat(rotVecFull, transVec, extrinsic);

                Mat R;
                R = K*extrinsic;

                cv::Mat_<double> src(4/*rows*/,1 /* cols */);

                src(0,0)=0;
                src(1,0)=0;
                src(2,0)=0;
                src(3,0)=1;

                cv::Mat_<double> dst = R*src;
                dst(0,0) = dst(0,0)/dst(2,0);
                dst(1,0) = dst(1,0)/dst(2,0);
                dst(2,0) = dst(2,0)/dst(2,0);

                ss.str(string());
                ss << dst.t();
                hello = hello + "\nCenter: " + ss.str();

                /* Set Region of Interest */

                double offset_x = dst(0,0)-rad;
                double offset_y = dst(1,0)-rad;


                roi.x = offset_x;
                roi.y = offset_y;
                roi.width = 2*rad;
                roi.height = 2*rad;

                /* Check if ROI lies in the image and Crop the original image to the defined ROI */
                bool is_inside = (roi & cv::Rect(0, 0, srcRgba.cols, srcRgba.rows)) == roi;
                if(is_inside && capture) {
                    cvtColor(srcRgba, colorRgba, CV_BGRA2RGBA);
                    crop = srcRgba(roi);
                    cvtColor(crop, crop, CV_BGR2RGB);
                    imwrite("/mnt/sdcard/Android/Data/CollaborativeAR/SEM_cropped.png", crop);
                }
                rectangle(srcRgba, roi, Scalar(255,0,0), 2);
                imwrite("/mnt/sdcard/Android/Data/CollaborativeAR/frame_ROI.png", srcRgba);
            } else{
                hello = hello + "Unable to estimate pose!";
            }
        } else{

            hello = hello + "Not enough inliers to estimate pose";

        }

    }

    ::gettimeofday(&end, NULL);
    timersub(&end, &start, &diff);

    ss.str(string());
    ss << diff.tv_usec/1e6;
    hello = hello + "\nTime elapsed in Pose Estimation: " + ss.str() + " sec.";


    transpose(srcRgba, srcRgba);
    flip(srcRgba, srcRgba,1);
//    circle(srcRgba,Point(srcRgba.cols/2,srcRgba.rows/2),2,Scalar(255,0,0),5);

    // copy to TextureView surface
    uchar *dbuf;
    uchar *sbuf;
    dbuf = dstRgba.data;
    sbuf = srcRgba.data;
    int i;
    for (i = 0; i < srcRgba.rows; i++) {
        dbuf = dstRgba.data + i * buf.stride * 4;
        memcpy(dbuf, sbuf, srcRgba.cols*4);
        sbuf += srcRgba.cols * 4;
    }

    ANativeWindow_unlockAndPost(win);
    ANativeWindow_release(win);


    return env->NewStringUTF(hello.c_str());

}
