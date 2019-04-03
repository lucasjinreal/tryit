#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
// #include <pcl_visualization/cloud_viewer.h>
#include <pcl/visualization/cloud_viewer.h>

using namespace std;


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
