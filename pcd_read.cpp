#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pclomp/ndt_omp.h>

#include "opencv2/opencv.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"

using namespace cv;
using namespace std;


// align point clouds and measure processing time
pcl::PointCloud<pcl::PointXYZ>::Ptr align(pcl::Registration<pcl::PointXYZ, pcl::PointXYZ>::Ptr registration, const pcl::PointCloud<pcl::PointXYZ>::Ptr& target_cloud, const pcl::PointCloud<pcl::PointXYZ>::Ptr& source_cloud ) {
  registration->setInputTarget(target_cloud);
  registration->setInputSource(source_cloud);
  pcl::PointCloud<pcl::PointXYZ>::Ptr aligned(new pcl::PointCloud<pcl::PointXYZ>());

  auto t1 = cv::getTickCount();
  registration->align(*aligned);
  std::cout << "single : " << (cv::getTickCount() - t1)/cv::getTickFrequency() << "[msec]" << std::endl;

  for(int i=0; i<10; i++) {
    registration->align(*aligned);
  }
  auto t3 = cv::getTickCount();
  std::cout << "10times: " << (cv::getTickCount() - t3)/cv::getTickFrequency() << "[msec]" << std::endl;
  std::cout << "fitness: " << registration->getFitnessScore() << std::endl << std::endl;
  return aligned;
}




int main(int argc, char **argv)
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

  if (argc >= 2)
  {
    string pcd_f = string(argv[1]);
    cout << pcd_f << endl;
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(pcd_f, *cloud) == -1) //* load the file
    {
      PCL_ERROR("Couldn't read file test_pcd.pcd \n");
      return (-1);
    }
    std::cout << "Loaded "
              << cloud->width * cloud->height
              << " data points from test_pcd.pcd with the following fields: "
              << std::endl;
    for (size_t i = 0; i < cloud->points.size(); ++i)
      std::cout << "    " << cloud->points[i].x
                << " " << cloud->points[i].y
                << " " << cloud->points[i].z << std::endl;

    // visualize it
    pcl::visualization::CloudViewer viewer("ff");
    viewer.showCloud(cloud);
    while (!viewer.wasStopped())
    {
    }
  }
  else
  {
    cout << "you must provide pcd file path.\n";
  }
  return (0);
}
