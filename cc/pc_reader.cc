#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
// #include <pcl_visualization/cloud_viewer.h>
#include <pcl/visualization/cloud_viewer.h>
#include "thor/os.h"



using namespace std;


typedef pcl::PointXYZRGB PointT;

int main(int argc, char **argv)
{
  pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);

  if (argc >= 2)
  {
    string pcd_f = string(argv[1]);
    cout << pcd_f << endl;

    string f_type = thor::os::suffix(pcd_f);
    if (f_type == "pcd") {
        // using pcd open it
        if (pcl::io::loadPCDFile<PointT>(pcd_f, *cloud) == -1) //* load the file
        {
        PCL_ERROR("Try using pcd_io open it but failed. \n");
        return (-1);
        }
    } else if (f_type == "ply") {
        // using ply open it, this may fail
        if (pcl::io::loadPLYFile<PointT>(pcd_f, *cloud) == -1) //* load the file
        {
        PCL_ERROR("Try using ply_io open it but failed. \n");
        return (-1);
        }
    }
    
    std::cout << "Loaded "
              << cloud->width * cloud->height
              << " data points from test_pcd.pcd with the following fields: "
              << std::endl;
    for (size_t i = 0; i < cloud->points.size(); ++i) {
      cloud->points[i].r = 0;
      cloud->points[i].g = 255;
      cloud->points[i].r = 0;
    }
    // visualize it
    // pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color(cloud, 0, 255, 0);
    pcl::visualization::CloudViewer viewer("ff");
    // change cloud color?
    viewer.showCloud(cloud);
    while (!viewer.wasStopped())
    {
    }
  }
  else
  {
    cout << "you must provide pcd/ply file path.\n";
  }
  return (0);
}
