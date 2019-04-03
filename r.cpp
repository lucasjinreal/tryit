// #include "opencv/opencv.hpp"
#include "opencv4/opencv2/opencv.hpp"
#include "opencv4/opencv2/core.hpp"
#include "opencv4/opencv2/video.hpp"
#include "opencv4/opencv2/videoio.hpp"
#include "opencv4/opencv2/video.hpp"
#include "opencv4/opencv2/videoio.hpp"


#include "thor/os.h"
#include <iostream>

#include "pcl/common/common_headers.h"
#include "pcl/io/pcd_io.h"
#include "pcl/visualization/pcl_visualizer.h"
#include "pcl/visualization/cloud_viewer.h"
#include "pcl_types.h"

using namespace std;
using namespace cv;


void ReadIntrinsics(Mat &cameraMatrix, Mat &distCoeffs, Size &imageSize, char *IntrinsicsPath)
{
	bool FSflag = false;
	FileStorage readfs;
	FSflag = readfs.open(IntrinsicsPath, FileStorage::READ);
	if (FSflag == false)
		cout << "Cannot open the file" << endl;
	readfs["CameraMat"] >> cameraMatrix;
	readfs["DistCoeff"] >> distCoeffs;
	readfs["image_width"] >> imageSize.width;
	readfs["image_height"] >> imageSize.height;
	cout << cameraMatrix << endl
		 << distCoeffs << endl
		 << imageSize << endl;
	readfs.release();
}

void unDistortion(Mat image, Mat &res, Mat camera_mat, Mat dist_coeffs)
{
	if (image.empty())
		cout << "image is empty!!" << endl;
	Size image_size = image.size();
	Mat map1, map2;
	initUndistortRectifyMap(camera_mat, dist_coeffs, Mat(), Mat(),
							image_size, CV_16SC2, map1, map2);
	remap(image, res, map1, map2, INTER_LINEAR);
}

void showBasicCloud()
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr basic_cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>());
	// create a circle cloud
	for (float z = -1.0; z <= 1.0; z += 0.05)
	{
		for (float angle = 0.0; angle <= 360.0; angle += 5)
		{
			pcl::PointXYZ basic_point;
			basic_point.x = 0.5 * cosf(pcl::deg2rad(angle));
			basic_point.y = sinf(pcl::deg2rad(angle));
			basic_point.z = z;
			cout << basic_point << endl;
			basic_cloud_ptr->points.push_back(basic_point);
		}
	}
	cout << basic_cloud_ptr->points.size() << endl;
	pcl::PointCloud<pcl::PointXYZ> &cloud = *basic_cloud_ptr; //点云
	cloud.width = cloud.points.size();
	cloud.height = 1;
	cout << "cloud shape: " << cloud.width << "x" << cloud.height << endl;
	pcl::io::savePCDFileASCII("test_pcd.pcd", cloud);
	cout << "Saved " << basic_cloud_ptr->points.size() << " data points to test_pcd.pcd." << std::endl;
	// how this cloud
	pcl::visualization::CloudViewer viewer("simple cloud viewer");
	// viwer.setBackgroundColor()
	viewer.showCloud(basic_cloud_ptr);
	while (!viewer.wasStopped())
	{
	}
}

int main()
{
	// showBasicCloud();

	pcl::PointCloud<apollo::perception::pcl_util::PointXYZIH> cloud; //初始化点云类型
	cloud.points.resize(2);
	cloud.width = 2;
	cloud.height = 1; //height为1表示无组织下的点云文件，

	cloud.points[0].h = 1;
	cloud.points[1].h = 2;
	cloud.points[0].x = cloud.points[0].y = cloud.points[0].z = 0;
	cloud.points[1].x = cloud.points[1].y = cloud.points[1].z = 3;

	pcl::io::savePCDFile("test.pcd", cloud);
	return 1;
}
