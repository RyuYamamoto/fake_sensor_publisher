#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>

#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/static_transform_broadcaster.h>

#include <Eigen/Dense>

#include <cmath>
#include <random>
#include <iostream>

using namespace Eigen;

double deg2rad(const double degree) { return degree * M_PI / 180.0; }

double rad2deg(const double radian) { return radian * 180.0 / M_PI; }

double normalize(double angle)
{
  while (angle >= M_PI) {
    angle -= 2 * M_PI;
  }
  while (angle <= -M_PI) {
    angle += 2 * M_PI;
  }
  return angle;
}

double sdlab_uniform()
{
  double ret = ((double)rand() + 1.0) / ((double)RAND_MAX + 2.0);
  return ret;
}

// gauss noise
double gauss(double mu, double sigma)
{
  double z = std::sqrt(-2.0 * std::log(sdlab_uniform())) * std::sin(2.0 * M_PI * sdlab_uniform());
  return mu + sigma * z;
}

double getGaussDistribution(double mean, double sigma)
{
  std::random_device seed;
  std::default_random_engine engine(seed());
  std::normal_distribution<> gauss(mean, sigma);
  return gauss(engine);
}

double getExponentialDistribution(double parameter)
{
  std::random_device seed;
  std::default_random_engine engine(seed());
  std::exponential_distribution<> exponential(parameter);
  return exponential(engine);
}

class FakeSensorPublisher : public rclcpp::Node
{
public:
  FakeSensorPublisher(const rclcpp::NodeOptions & node_options)
  : Node("fake_sensor_publisher", node_options)
  {
    Q_ << 0.1, 0, 0, deg2rad(30);
    Q_ = Q_ * Q_;
    R_ << 0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, deg2rad(5);
    R_ = R_ * R_;

    distance_until_noise_ = getExponentialDistribution(1.0 / 5.0);

    twist_subscriber_ = this->create_subscription<geometry_msgs::msg::Twist>(
      "twist", 10, std::bind(&FakeSensorPublisher::callbackTwist, this, std::placeholders::_1));
    ground_truth_publisher_ =
      this->create_publisher<geometry_msgs::msg::PoseStamped>("ground_truth", 10);
    odometry_publisher_ = this->create_publisher<nav_msgs::msg::Odometry>("odom", 10);
    gnss_publisher_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("gnss", 10);

    previous_stamp_ = rclcpp::Clock().now();

    timer_ = this->create_wall_timer(
      std::chrono::milliseconds(static_cast<int>(1000 * 0.1)),
      std::bind(&FakeSensorPublisher::timerCallback, this));
  }
  ~FakeSensorPublisher() {}

  void callbackTwist(const geometry_msgs::msg::Twist msg)
  {
    twist_ = msg;
  }

  geometry_msgs::msg::Pose inputToMsgs(Matrix<double, 3, 1> pose_2d)
  {
    geometry_msgs::msg::Pose msg;
    msg.position.x = pose_2d[0];
    msg.position.y = pose_2d[1];

    tf2::Quaternion quat;
    quat.setRPY(0.0, 0.0, normalize(pose_2d[2]));
    msg.orientation.w = quat.w();
    msg.orientation.x = quat.x();
    msg.orientation.y = quat.y();
    msg.orientation.z = quat.z();

    return msg;
  }
  void noise(Eigen::Matrix<double, 3, 1> &pose, geometry_msgs::msg::Twist twist, double time_interval)
  {
    distance_until_noise_ -=
      ((std::fabs(twist.linear.x) * time_interval + std::fabs(twist.angular.z) * time_interval));
    if (distance_until_noise_ <= 0.0) {
      distance_until_noise_ += getExponentialDistribution(1.0 / 5.0);
      pose(2) += getGaussDistribution(0.0, M_PI / 60.0);
    }
  }
  void timerCallback()
  {
    rclcpp::Time current_stamp = rclcpp::Clock().now();

    geometry_msgs::msg::PoseStamped grund_truth_msg;
    geometry_msgs::msg::PoseStamped gps_msgs;
    nav_msgs::msg::Odometry odom_msg;

    Matrix<double, 2, 1> u(twist_.linear.x, twist_.angular.z);

    const double dt = (current_stamp - previous_stamp_).seconds();

    // ground truth
    RCLCPP_INFO(get_logger(), "delta time: %f", dt);
    Matrix<double, 2, 1> ud = motionNoise(u, Q_);
    grund_truth_ = motionModel(grund_truth_, ud, dt);
    //noise(grund_truth_, twist_, dt);
    grund_truth_msg.header.frame_id = "map";
    grund_truth_msg.header.stamp = current_stamp;
    grund_truth_msg.pose = inputToMsgs(grund_truth_);

    // dead recogning
    odom_ = motionModel(odom_, ud, dt);
    odom_msg.header.frame_id = "map";
    odom_msg.header.stamp = current_stamp;
    odom_msg.pose.pose = inputToMsgs(odom_);

    // observation
    Matrix<double, 3, 1> gps = observationNoise(grund_truth_, R_);
    gps_msgs.header.frame_id = "map";
    gps_msgs.header.stamp = current_stamp;
    gps_msgs.pose = inputToMsgs(gps);

    ground_truth_publisher_->publish(grund_truth_msg);
    odometry_publisher_->publish(odom_msg);
    gnss_publisher_->publish(gps_msgs);

    previous_stamp_ = current_stamp;
  }
  Matrix<double, 3, 1> motionModel(Matrix<double, 3, 1> x, Matrix<double, 2, 1> u, double dt)
  {
    Matrix<double, 3, 3> F = Matrix<double, 3, 3>::Identity();
    Matrix<double, 3, 2> B;
    B << dt * std::cos(x[2]), 0, dt * std::sin(x[2]), 0, 0, dt;

    x = F * x + B * u;
    x[2] = normalize(x[2]);
    return x;
  }
  Matrix<double, 2, 1> motionNoise(Matrix<double, 2, 1> u, Matrix<double, 2, 2> Q)
  {
    Matrix<double, 2, 1> uw(gauss(0.0, Q(0, 0)), gauss(0.0, Q(1, 1)));
    return u + uw;
  }
  Matrix<double, 3, 1> observationNoise(Matrix<double, 3, 1> x, Matrix<double, 3, 3> R)
  {
    Matrix<double, 3, 1> xw(gauss(0.0, 1.0) * R(0, 0), gauss(0.0, 1.0) * R(1, 1), gauss(0.0, 1.0) * R(2, 2));
    return x + xw;
  }

private:
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Time previous_stamp_;

  geometry_msgs::msg::Twist twist_;

  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr twist_subscriber_;

  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr ground_truth_publisher_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odometry_publisher_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr gnss_publisher_;

  Matrix<double, 2, 2> Q_;
  Matrix<double, 3, 3> R_;
  Matrix<double, 3, 1> grund_truth_;
  Matrix<double, 3, 1> odom_;
  Matrix<double, 3, 1> gps_;

  double distance_until_noise_;
};

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(FakeSensorPublisher)
