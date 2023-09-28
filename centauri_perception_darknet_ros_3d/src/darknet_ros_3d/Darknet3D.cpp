// Copyright 2020 Intelligent Robotics Lab
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/* Author: Francisco Martín fmrico@gmail.com */
/* Author: Fernando González fergonzaramos@yahoo.es */

#include "darknet_ros_3d/Darknet3D.hpp"
#include <tf2/transform_datatypes.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.h>
#include <visualization_msgs/msg/marker_array.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/point32.hpp>
#include <sensor_msgs/point_cloud_conversion.hpp>
#include <sensor_msgs/msg/point_field.hpp>

#include <algorithm>
#include <memory>
#include <limits>
#include <gb_visual_detection_3d_msgs/msg/bounding_box3d.hpp>

// CENTAURI'S PERCEPTION SYSTEM
#include <Eigen/Geometry>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>

#include </opt/ros/foxy/include/geometry_msgs/msg/point32.h>
#include </opt/ros/foxy/include/pcl_conversions/pcl_conversions.h>

#include </usr/include/pcl-1.10/pcl/common/transforms.h>
#include </usr/include/pcl-1.10/pcl/common/centroid.h>
#include </usr/include/pcl-1.10/pcl/filters/crop_box.h>
#include </usr/include/pcl-1.10/pcl/features/moment_of_inertia_estimation.h>
#include </usr/include/pcl-1.10/pcl/point_types.h>
#include </usr/include/pcl-1.10/pcl/filters/extract_indices.h>
#include </usr/include/pcl-1.10/pcl/point_cloud.h>
#include </usr/include/pcl-1.10/pcl/point_types.h>
#include </usr/include/pcl-1.10/pcl/common/centroid.h>
#include </usr/include/pcl-1.10/pcl/visualization/pcl_visualizer.h>
#include </usr/include/pcl-1.10/pcl/features/normal_3d.h>
#include </usr/include/pcl-1.10/pcl/common/pca.h>

#include <Eigen/Dense>
#include <rclcpp/rclcpp.hpp>



using std::placeholders::_1;
using CallbackReturnT =
  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

namespace darknet_ros_3d
{
Darknet3D::Darknet3D()
: LifecycleNode("darknet3d_node"), clock_(RCL_SYSTEM_TIME),
  tfBuffer_(std::make_shared<rclcpp::Clock>(clock_)), tfListener_(tfBuffer_, true),
  pc_received_(false)
{
  // Init Params

  this->declare_parameter("darknet_ros_topic", "/darknet_ros/bounding_boxes");
  this->declare_parameter("output_bbx3d_topic", "/darknet_ros_3d/bounding_boxes");
  this->declare_parameter("point_cloud_topic", "/kinect2/hd/points");
  this->declare_parameter("working_frame", "/kinect2_link");
  this->declare_parameter("maximum_detection_threshold", 0.3f);
  this->declare_parameter("minimum_probability", 0.3f);
  this->declare_parameter("interested_classes");

  this->configure();

  pointCloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
    pointcloud_topic_, 1, std::bind(&Darknet3D::pointCloudCb, this, std::placeholders::_1));

  darknet_ros_sub_ = this->create_subscription<darknet_ros_msgs::msg::BoundingBoxes>(
    input_bbx_topic_, 1, std::bind(&Darknet3D::darknetCb, this, std::placeholders::_1));

  darknet3d_pub_ = this->create_publisher<gb_visual_detection_3d_msgs::msg::BoundingBoxes3d>(
    output_bbx3d_topic_, 100);

  markers_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
    "/darknet_ros_3d/markers", 1);

  centroid_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
    "/darknet_ros_3d/centroid", 1);

  isolated_cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
    "/darknet_ros_3d/isolated_pointcloud", 10);

  object_frame_pub_ = this->create_publisher<geometry_msgs::msg::TransformStamped>(
    "/darknet_ros_3d/object_frame", 10);

  


  last_detection_ts_ = clock_.now();

  this->activate();
}

void
Darknet3D::pointCloudCb(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
  point_cloud_ = *msg;
  pc_received_ = true;
}

void
Darknet3D::darknetCb(const darknet_ros_msgs::msg::BoundingBoxes::SharedPtr msg)
{
  original_bboxes_ = msg->bounding_boxes;
  last_detection_ts_ = clock_.now();
}

// Function to claculate the centroid of the bounding box

Eigen::Affine3f calculateTransform(const Eigen::Vector3f& centroid, const Eigen::Quaternionf& quaternion)
{
    Eigen::Affine3f transform = Eigen::Translation3f(centroid) * quaternion;
    return transform;
}


void Darknet3D::calculate_boxes(sensor_msgs::msg::PointCloud2 cloud_pc2,
                                sensor_msgs::msg::PointCloud cloud_pc,
                                gb_visual_detection_3d_msgs::msg::BoundingBoxes3d *boxes)
{
  boxes->header.stamp = cloud_pc2.header.stamp;
  boxes->header.frame_id = cloud_pc2.header.frame_id;

  for (auto bbx : original_bboxes_)
  {
    if ((bbx.probability < minimum_probability_) ||
        (std::find(interested_classes_.begin(), interested_classes_.end(), bbx.class_id) == interested_classes_.end()))
    {
      continue;
    }

    int center_x, center_y;

    center_x = (bbx.xmax + bbx.xmin) / 2;
    center_y = (bbx.ymax + bbx.ymin) / 2;

    int pc_index = (center_y * cloud_pc2.width) + center_x;

    // Convert sensor_msgs::msg::PointCloud2 to pcl::PointCloud<pcl::PointXYZ>
    // Los datos de cloud_pc2 de tipo PointCLoud2 son almacenados en un nuevo mensaje de tipo PointCloud pcl_cloud
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::fromROSMsg(cloud_pc2, *pcl_cloud);


    pcl::PointXYZRGB center_point = pcl_cloud->at(pc_index);

    if (std::isnan(center_point.x))
    {
      continue;
    }

    float maxx, minx, maxy, miny, maxz, minz;

    maxx = maxy = maxz = -std::numeric_limits<float>::max();
    minx = miny = minz = std::numeric_limits<float>::max();

    for (int i = bbx.xmin; i < bbx.xmax; i++)
    {
      for (int j = bbx.ymin; j < bbx.ymax; j++)
      {
        pc_index = (j * cloud_pc2.width) + i;
        pcl::PointXYZRGB point = pcl_cloud->at(pc_index);

    if (std::isnan(point.x) || std::isnan(point.y) || std::isnan(point.z))
    {
      continue;
    }

    // Calcula la diferencia de profundidad entre el punto y el centro
    float depth_difference = std::abs(point.z - center_point.z);

    // Establece el umbral de profundidad en 0.1 metros (10 centímetros)
    float maximum_depth_difference_threshold = 0.1;  // Umbral de profundidad deseado en metros

    if (depth_difference > maximum_depth_difference_threshold)
    {
      continue;
    }

    if (fabs(point.x - center_point.x) > maximum_detection_threshold_)
    {
      continue;
    }

    maxx = std::max(point.x, maxx);
    maxy = std::max(point.y, maxy);
    maxz = std::max(point.z, maxz);
    minx = std::min(point.x, minx);
    miny = std::min(point.y, miny);
    minz = std::min(point.z, minz);
  }
    }


    // Cálculo del centroide
    Eigen::Vector3f centroid((maxx + minx) / 2, (maxy + miny) / 2, (maxz + minz) / 2);

    Eigen::Vector3f x_dir = Eigen::Vector3f::UnitX();
    Eigen::Vector3f y_dir = Eigen::Vector3f::UnitY();
    Eigen::Vector3f z_dir = Eigen::Vector3f::UnitZ();

    Eigen::Vector3f up_vector = y_dir;
    Eigen::Vector3f bbx_dir = Eigen::Vector3f(maxx - minx, maxy - miny, maxz - minz).normalized();

    Eigen::Quaternionf quaternion;
    quaternion.setFromTwoVectors(up_vector, bbx_dir);

    Eigen::Affine3f transform = calculateTransform(centroid, quaternion);

  // Se invoca la función para aislar el objeto detectado in a nube de puntos.
    publish_isolated_pointcloud(cloud_pc2, minx, maxx, miny, maxy, minz, maxz, working_frame_);
  // Publicar Marcadores para el Centroide del Objeto detectado
    visualization_msgs::msg::MarkerArray msg_centroid;

    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = working_frame_;
    marker.header.stamp = this->get_clock()->now();
    marker.ns = "centroid_marker";
    marker.id = 0;
    marker.type = visualization_msgs::msg::Marker::SPHERE;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.pose.position.x = centroid(0);
    marker.pose.position.y = centroid(1);
    marker.pose.position.z = centroid(2);
    marker.pose.orientation.x = 0.0;
    marker.pose.orientation.y = 0.0;
    marker.pose.orientation.z = 0.0;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = 0.02;
    marker.scale.y = 0.02;
    marker.scale.z = 0.02;
    marker.color.a = 1.0; // alpha channel
    marker.color.r = 1.0; // red channel
    marker.color.g = 0.0; // green channel
    marker.color.b = 0.0; // blue channel

    msg_centroid.markers.push_back(marker);

    centroid_pub_->publish(msg_centroid);

    gb_visual_detection_3d_msgs::msg::BoundingBox3d bbx_msg;
    bbx_msg.object_name = bbx.class_id;
    bbx_msg.probability = bbx.probability;

    bbx_msg.xmin = minx;
    bbx_msg.xmax = maxx;
    bbx_msg.ymin = miny;
    bbx_msg.ymax = maxy;
    bbx_msg.zmin = minz;
    bbx_msg.zmax = maxz;


    boxes->bounding_boxes.push_back(bbx_msg);  }
}

void
Darknet3D::publish_markers(gb_visual_detection_3d_msgs::msg::BoundingBoxes3d boxes)
{
  visualization_msgs::msg::MarkerArray msg;

  int counter_id = 0;
  for (auto bb : boxes.bounding_boxes) {
    visualization_msgs::msg::Marker bbx_marker;

    bbx_marker.header.frame_id = working_frame_;
    bbx_marker.header.stamp = boxes.header.stamp;
    bbx_marker.ns = "darknet3d";
    bbx_marker.id = counter_id++;
    bbx_marker.type = visualization_msgs::msg::Marker::CUBE;
    bbx_marker.action = visualization_msgs::msg::Marker::ADD;
    bbx_marker.frame_locked = false;
    bbx_marker.pose.position.x = (bb.xmax + bb.xmin) / 2.0;
    bbx_marker.pose.position.y = (bb.ymax + bb.ymin) / 2.0;
    bbx_marker.pose.position.z = (bb.zmax + bb.zmin) / 2.0;
    bbx_marker.pose.orientation.x = 0.0;
    bbx_marker.pose.orientation.y = 0.0;
    bbx_marker.pose.orientation.z = 0.0;
    bbx_marker.pose.orientation.w = 1.0;
    bbx_marker.scale.x = (bb.xmax - bb.xmin);
    bbx_marker.scale.y = (bb.ymax - bb.ymin);
    bbx_marker.scale.z = (bb.zmax - bb.zmin);
    bbx_marker.color.b = 0;
    bbx_marker.color.g = bb.probability * 255.0;
    bbx_marker.color.r = (1.0 - bb.probability) * 255.0;
    bbx_marker.color.a = 0.4;
    bbx_marker.lifetime = rclcpp::Duration(1.0);
    bbx_marker.text = bb.object_name;

    msg.markers.push_back(bbx_marker);
  }

  if (markers_pub_->is_activated()) {
    markers_pub_->publish(msg);
  }

}

void Darknet3D::publish_isolated_pointcloud(sensor_msgs::msg::PointCloud2 input_cloud,
                                            float min_x, float max_x,
                                            float min_y, float max_y,
                                            float min_z, float max_z,
                                            std::string working_frame_)
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::fromROSMsg(input_cloud, *pcl_cloud);

    // Crear una nueva nube de puntos que contendrá solo los puntos dentro de la bounding box
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr isolated_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

    for (const auto& point : pcl_cloud->points)
    {
        if (point.x >= min_x && point.x <= max_x &&
            point.y >= min_y && point.y <= max_y &&
            point.z >= min_z && point.z <= max_z)
        {
            isolated_cloud->points.push_back(point);
        }
    }

    // Configurar el mensaje PointCloud2 para publicar
    sensor_msgs::msg::PointCloud2 isolated_cloud_output;
    pcl::toROSMsg(*isolated_cloud, isolated_cloud_output);
    isolated_cloud_output.header.frame_id = working_frame_;

    // Publicar la nube de puntos aislada
    isolated_cloud_pub_->publish(isolated_cloud_output);

    // Invocar función para aplicar PCA a la nube de puntos aislada
    Eigen::Matrix3f eigen_vectors;
    Eigen::Vector3f eigen_values;
    apply_pca_to_pointcloud(isolated_cloud, eigen_vectors, eigen_values, working_frame_);
}


void Darknet3D::apply_pca_to_pointcloud(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& isolated_cloud,
                              Eigen::Matrix3f& eigen_vectors,
                              Eigen::Vector3f& eigen_values,
                              std::string working_frame_)
{
    // Crear un objeto PCA (Análisis de Componentes Principales) para calcular los vectores y valores propios
    pcl::PCA<pcl::PointXYZRGB> pca;
    pca.setInputCloud(isolated_cloud);

    // Obtener los vectores propios (eigen_vectors) y los valores propios (eigen_values)
    eigen_vectors = pca.getEigenVectors();
    eigen_values = pca.getEigenValues();

      // Crear el mensaje para los valores de PCA
    publish_pca_values(eigen_vectors, eigen_values);

        // Publicar el marco de coordenadas en el centroide de la nube de puntos
    publish_frame_at_centroid(isolated_cloud, eigen_vectors, working_frame_);
}

void Darknet3D::publish_pca_values(const Eigen::Matrix3f& eigen_vectors, const Eigen::Vector3f& eigen_values)
{
    // Crear un nodo ROS 2 para publicar los valores de PCA
    rclcpp::Node::SharedPtr node = rclcpp::Node::make_shared("pca_publisher");

    // Crear el mensaje para publicar los valores propios y vectores propios
    auto pca_msg = std::make_shared<std_msgs::msg::Float32MultiArray>();
    pca_msg->layout.dim.push_back(std_msgs::msg::MultiArrayDimension());
    pca_msg->layout.dim[0].size = 3;  // Cantidad de valores propios
    pca_msg->layout.dim[0].stride = 1;
    pca_msg->layout.dim[0].label = "eigen_values";
    
    pca_msg->layout.dim.push_back(std_msgs::msg::MultiArrayDimension());
    pca_msg->layout.dim[1].size = 9;  // Cantidad de elementos en la matriz de vectores propios (3x3)
    pca_msg->layout.dim[1].stride = 3;
    pca_msg->layout.dim[1].label = "eigen_vectors";

    pca_msg->data.resize(12);  // 3 valores propios + 9 elementos de la matriz de vectores propios

    // Llenar el mensaje con los valores propios y vectores propios
    for (int i = 0; i < 3; ++i) {
        pca_msg->data[i] = eigen_values(i);
    }

    for (int i = 0; i < 9; ++i) {
        pca_msg->data[i + 3] = eigen_vectors(i / 3, i % 3);
    }

    // Crear un publicador para el tópico "pca_pub_"
    auto pca_pub = node->create_publisher<std_msgs::msg::Float32MultiArray>("pca_pub_", 10);

    // Publicar el mensaje
    pca_pub->publish(*pca_msg);
}

void Darknet3D::publish_frame_at_centroid(const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& isolated_cloud, const Eigen::Matrix3f& eigen_vectors, std::string working_frame_)
{
    // Calcular el centroide de la nube de puntos
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*isolated_cloud, centroid);

    // Crear el mensaje de transformación para el marco de coordenadas
    geometry_msgs::msg::TransformStamped transform_msg;
    transform_msg.header.stamp = rclcpp::Clock().now();
    transform_msg.header.frame_id = working_frame_;
    transform_msg.child_frame_id = "Object_frame"; // Puedes cambiar este nombre si lo deseas

    transform_msg.transform.translation.x = centroid[0];
    transform_msg.transform.translation.y = centroid[1];
    transform_msg.transform.translation.z = centroid[2];

    // Ajustar la orientación del marco de coordenadas utilizando los vectores propios
    Eigen::Quaternionf quat;
    quat.setFromTwoVectors(Eigen::Vector3f::UnitX(), eigen_vectors.col(0).head(3));
    transform_msg.transform.rotation.x = quat.x();
    transform_msg.transform.rotation.y = quat.y();
    transform_msg.transform.rotation.z = quat.z();
    transform_msg.transform.rotation.w = quat.w();

    // Publicar el marco de coordenadas
    
    // Crear un nodo ROS 2
    rclcpp::Node::SharedPtr node = rclcpp::Node::make_shared("frame_publisher");
    tf2_ros::TransformBroadcaster tf_broadcaster(node);

    // Publicar el mensaje de transformación
    tf_broadcaster.sendTransform(transform_msg);
    
    // Publicar el mensaje de transformación en el topic /object_frame
    object_frame_pub_->publish(transform_msg);     


}


void
Darknet3D::update()
{
  if (this->get_current_state().id() != lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE) {
    return;
  }

  if ((clock_.now() - last_detection_ts_).seconds() > 2.0 || !pc_received_) {
    return;
  }

  sensor_msgs::msg::PointCloud2 local_pointcloud;
  geometry_msgs::msg::TransformStamped transform;
  sensor_msgs::msg::PointCloud cloud_pc;
  gb_visual_detection_3d_msgs::msg::BoundingBoxes3d msg;

  try {
    transform = tfBuffer_.lookupTransform(working_frame_, point_cloud_.header.frame_id,
        point_cloud_.header.stamp, tf2::durationFromSec(2.0));
  } catch (tf2::TransformException & ex) {
    RCLCPP_ERROR(this->get_logger(), "Transform error of sensor data: %s, %s\n",
      ex.what(), "quitting callback");
    return;
  }
  tf2::doTransform<sensor_msgs::msg::PointCloud2>(point_cloud_, local_pointcloud, transform);
  sensor_msgs::convertPointCloud2ToPointCloud(local_pointcloud, cloud_pc);

  calculate_boxes(local_pointcloud, cloud_pc, &msg);
  publish_markers(msg);

  if (darknet3d_pub_->is_activated()) {
    darknet3d_pub_->publish(msg);
  }
}

CallbackReturnT
Darknet3D::on_configure(const rclcpp_lifecycle::State & state)
{
  RCLCPP_INFO(this->get_logger(), "[%s] Configuring from [%s] state...",
    this->get_name(), state.label().c_str());

  this->get_parameter("darknet_ros_topic", input_bbx_topic_);
  this->get_parameter("output_bbx3d_topic", output_bbx3d_topic_);
  this->get_parameter("point_cloud_topic", pointcloud_topic_);
  this->get_parameter("working_frame", working_frame_);
  this->get_parameter("maximum_detection_threshold", maximum_detection_threshold_);
  this->get_parameter("minimum_probability", minimum_probability_);
  this->get_parameter("interested_classes", interested_classes_);

  return CallbackReturnT::SUCCESS;
}

CallbackReturnT
Darknet3D::on_activate(const rclcpp_lifecycle::State & state)
{
  RCLCPP_INFO(this->get_logger(), "[%s] Activating from [%s] state...",
    this->get_name(), state.label().c_str());

  darknet3d_pub_->on_activate();
  markers_pub_->on_activate();
  centroid_pub_->on_activate();
  isolated_cloud_pub_->on_activate();
  object_frame_pub_->on_activate();

  return CallbackReturnT::SUCCESS;
}

CallbackReturnT
Darknet3D::on_deactivate(const rclcpp_lifecycle::State & state)
{
  RCLCPP_INFO(this->get_logger(), "[%s] Deactivating from [%s] state...",
    this->get_name(), state.label().c_str());

  darknet3d_pub_->on_deactivate();
  markers_pub_->on_deactivate();
  centroid_pub_->on_deactivate();
  isolated_cloud_pub_->on_deactivate();
  object_frame_pub_->on_deactivate();

  return CallbackReturnT::SUCCESS;
}

CallbackReturnT
Darknet3D::on_cleanup(const rclcpp_lifecycle::State & state)
{
  RCLCPP_INFO(this->get_logger(), "[%s] Cleanning Up from [%s] state...",
    this->get_name(), state.label().c_str());

  darknet3d_pub_.reset();
  markers_pub_.reset();
  centroid_pub_.reset();
  isolated_cloud_pub_.reset();
  object_frame_pub_.reset();

  return CallbackReturnT::SUCCESS;
}

CallbackReturnT
Darknet3D::on_shutdown(const rclcpp_lifecycle::State & state)
{
  RCLCPP_INFO(this->get_logger(), "[%s] Shutting Down from [%s] state...",
    this->get_name(), state.label().c_str());

  darknet3d_pub_.reset();
  markers_pub_.reset();
  centroid_pub_.reset();
  isolated_cloud_pub_.reset();
  object_frame_pub_.reset();

  return CallbackReturnT::SUCCESS;
}

CallbackReturnT
Darknet3D::on_error(const rclcpp_lifecycle::State & state)
{
  RCLCPP_INFO(this->get_logger(), "[%s] Shutting Down from [%s] state...",
    this->get_name(), state.label().c_str());
  return CallbackReturnT::SUCCESS;
}

}  // namespace darknet_ros_3d
