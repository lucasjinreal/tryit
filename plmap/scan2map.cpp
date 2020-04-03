#include <iostream>
#include <ros/ros.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/registration/ndt.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <math.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/PointCloud2.h>

#include <visualization_msgs/Marker.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <boost/thread/thread.hpp>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <mutex>
#include <thread>
#include <cfloat>

#include <pclomp/ndt_omp.h>


ros::Publisher plmap_pub;
pcl::PointCloud<pcl::PointXYZ>::Ptr current_cloud (new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr map_cloud (new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr last_map_cloud (new pcl::PointCloud<pcl::PointXYZ>);

Eigen::Matrix4f rigid_trans = Eigen::Matrix4f::Identity(), last_map_pose = Eigen::Matrix4f::Identity();

std::mutex m_mapsave;

void PL_command()
{
    while(1)
    {
        char c = getchar();
        if (c == 's')
        {
            m_mapsave.lock();

            pcl::PCDWriter writer;
            writer.write("/home/wenpeng/catkin_ws/cross_map.pcd", *map_cloud);
            printf("pointcloud saved\n");
            m_mapsave.unlock();
        }
        std::chrono::milliseconds dura(1000);
        std::this_thread::sleep_for(dura);
    }
}



// align point clouds and measure processing time
pcl::PointCloud<pcl::PointXYZ>::Ptr align(pcl::Registration<pcl::PointXYZ, pcl::PointXYZ>::Ptr registration,
                                          const pcl::PointCloud<pcl::PointXYZ>::Ptr& target_cloud,
                                          const pcl::PointCloud<pcl::PointXYZ>::Ptr& source_cloud, 
                                          Eigen::Matrix4f &trans)
{
  registration->setInputTarget(target_cloud);
  registration->setInputSource(source_cloud);
  pcl::PointCloud<pcl::PointXYZ>::Ptr aligned(new pcl::PointCloud<pcl::PointXYZ>());

  registration->align(*aligned, trans);
  trans = registration->getFinalTransformation();

  //std::cout << registration->getFinalTransformation() << std::endl;//add for transformation result, wp 2018.11.24

  return aligned;
}

void scan_callback(const sensor_msgs::PointCloud2::ConstPtr &pl_msg)
{
    static float recv_cnt = 1.0;
    pcl::PCLPointCloud2 pcl_tmp;
    pcl_conversions::toPCL(*pl_msg,pcl_tmp);
    pcl::fromPCLPointCloud2(pcl_tmp,*current_cloud);
    
    if (static_cast<int>(map_cloud->points.size()) == 0)
    {
        m_mapsave.lock();
        pcl::copyPointCloud(*current_cloud, *map_cloud);
        m_mapsave.unlock();
    }
    else
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr alinged_pl (new pcl::PointCloud<pcl::PointXYZ>);
        // downsampling
        pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled(new pcl::PointCloud<pcl::PointXYZ>());

        pcl::VoxelGrid<pcl::PointXYZ> voxelgrid;
        
        voxelgrid.setLeafSize(0.1f/recv_cnt, 0.1f/recv_cnt, 0.1f/recv_cnt);
        voxelgrid.setInputCloud(last_map_cloud);
        voxelgrid.filter(*downsampled);
        *last_map_cloud = *downsampled;

        voxelgrid.setLeafSize(0.1f, 0.1f, 0.1f);
        voxelgrid.setInputCloud(current_cloud);//voxelgrid.setInputCloud(source_cloud);
        voxelgrid.filter(*downsampled);
        *current_cloud = *downsampled;//source_cloud = downsampled;
        
        pclomp::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ>::Ptr ndt_omp(new pclomp::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ>());
        ndt_omp->setResolution(1.0);

        ndt_omp->setNumThreads(omp_get_max_threads());
        ndt_omp->setNeighborhoodSearchMethod(pclomp::DIRECT7); //pclomp::DIRECT1
        //*map_cloud += *(align(ndt_omp, last_map_cloud, current_cloud));
        alinged_pl = align(ndt_omp, last_map_cloud, current_cloud, rigid_trans);
        
        rigid_trans = ndt_omp->getFinalTransformation();
        Eigen::AngleAxisf axis_rot;
        float dist = (rigid_trans.block<3,1>(0,3) - last_map_pose.block<3,1>(0,3)).norm();
        axis_rot.fromRotationMatrix(rigid_trans.block<3,3>(0,0) * last_map_pose.block<3,3>(0,0));
        
        if ( (abs(axis_rot.angle()) >= 0.174532925) || (dist >= 1) )
        {
            m_mapsave.lock();
            *map_cloud += *(alinged_pl);
            m_mapsave.unlock();
            last_map_pose = rigid_trans;
            recv_cnt+=1.0;
        }
        
    }
    pcl::copyPointCloud(*map_cloud, *last_map_cloud);
    
    sensor_msgs::PointCloud2 plmap_msg;
    pcl::toROSMsg(*map_cloud, plmap_msg);
    
    plmap_msg.header.stamp = pl_msg->header.stamp;
    plmap_msg.header.frame_id = pl_msg->header.frame_id;
    
    plmap_pub.publish(plmap_msg);
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "scan2map");
    ros::NodeHandle n("~");
    
    ros::Subscriber sub_scan = n.subscribe("/cti/sensor/rslidar/PointCloud2", 2000, scan_callback);
    plmap_pub = n.advertise<sensor_msgs::PointCloud2>("/pl_map", 1000);
    
    std::thread keyboard_command_process;

    keyboard_command_process = std::thread(PL_command);
    
    
    ros::spin();

//  Eigen::Matrix4f transformation;
//  transformation.setIdentity();
//  transformation.col(3) << 0.5,0.1,-0.02,1;

//pcl::transformPointCloud ( *source_cloud, *test_cloud, transformation);


    // visulization
    /*pcl::visualization::PCLVisualizer vis("vis");
    //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> test_handler(test_cloud, 255.0, 255.0, 255.0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> target_handler(target_cloud, 255.0, 0.0, 0.0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> source_handler(source_cloud, 0.0, 255.0, 0.0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> aligned_handler(aligned, 0.0, 0.0, 255.0);
    //vis.addPointCloud(test_cloud, test_handler, "test");
    vis.addPointCloud(target_cloud, target_handler, "target");
    vis.addPointCloud(source_cloud, source_handler, "source");
    vis.addPointCloud(aligned, aligned_handler, "aligned");
    vis.spin();*/

  return 0;
}
