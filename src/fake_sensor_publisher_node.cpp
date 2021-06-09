#include <nav_msgs/Odometry.h>
#include <ros/ros.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <Eigen/Core>
#include <cmath>
#include <iostream>

using namespace Eigen;

double deg2rad(double degree) { return degree * M_PI / 180.0; }

double rad2deg(double radian) { return radian * 180.0 / M_PI; }

double angle_limit_pi(double angle)
{
  while (angle >= M_PI) { angle -= 2 * M_PI; }
  while (angle <= -M_PI) { angle += 2 * M_PI; }
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

class FakeSensorPublisher
{
public:
  FakeSensorPublisher(ros::NodeHandle nh)
  : _nh(nh),
    grund_truth(Matrix<double, 3, 1>::Zero()),
    odom(Matrix<double, 3, 1>::Zero()),
    gps(Matrix<double, 3, 1>::Zero())
  {
    Q << 0.1, 0, 0, deg2rad(30);
    Q = Q * Q;
    R << 2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, deg2rad(5);
    R = R * R;

    _ground_truth_pub = _nh.advertise<nav_msgs::Odometry>("/fake_sensor_publisher/grund_truth", 10);
    _odom_pub = _nh.advertise<nav_msgs::Odometry>("/fake_sensor_publisher/odom", 10);
    _gps_pub = _nh.advertise<nav_msgs::Odometry>("/fake_sensor_publisher/gps", 10);
  }
  ~FakeSensorPublisher() {}
  nav_msgs::Odometry input2nav_msgs(Matrix<double, 3, 1> pose_2d, Matrix<double, 2, 1> input)
  {
    nav_msgs::Odometry msg;
    msg.header.frame_id = "world";
    msg.child_frame_id = "odom";
    msg.pose.pose.position.x = pose_2d[0];
    msg.pose.pose.position.y = pose_2d[1];

    tf2::Quaternion quat;
    quat.setRPY(0.0, 0.0, angle_limit_pi(pose_2d[2]));
    msg.pose.pose.orientation.w = quat.w();
    msg.pose.pose.orientation.x = quat.x();
    msg.pose.pose.orientation.y = quat.y();
    msg.pose.pose.orientation.z = quat.z();
    msg.twist.twist.linear.x = input[0];
    msg.twist.twist.angular.z = input[1];

    return msg;
  }
  void run()
  {
    nav_msgs::Odometry grund_truth_msg;
    nav_msgs::Odometry odom_msg;
    nav_msgs::Odometry gps_msgs;

    Matrix<double, 2, 1> u(1.0, deg2rad(5));  // input

    double dt = 0.02;
    ros::Time old_timestamp = ros::Time::now();
    ros::Rate rate(50);
    while (ros::ok()) {
      ros::Time now = ros::Time::now();

      // ground truth
      ROS_INFO("%f", dt);
      grund_truth = motion_model(grund_truth, u, dt);
      grund_truth_msg = input2nav_msgs(grund_truth, u);

      // dead recogning
      Matrix<double, 2, 1> ud = motion_noise(u, Q);
      odom = motion_model(odom, ud, dt);
      odom_msg = input2nav_msgs(odom, ud);

      // observation
      Matrix<double, 3, 1> gps = observation_noise(grund_truth, R);
      gps_msgs = input2nav_msgs(gps, Matrix<double, 2, 1>::Zero());

      _ground_truth_pub.publish(grund_truth_msg);
      _odom_pub.publish(odom_msg);
      _gps_pub.publish(gps_msgs);

      rate.sleep();
      dt = (now - old_timestamp).toSec();
      old_timestamp = now;
    }
  }
  Matrix<double, 3, 1> motion_model(Matrix<double, 3, 1> x, Matrix<double, 2, 1> u, double dt)
  {
    Matrix<double, 3, 3> F = Matrix<double, 3, 3>::Identity();
    Matrix<double, 3, 2> B;
    B << dt * std::cos(x[2]), 0, dt * std::sin(x[2]), 0, 0, dt;

    x = F * x + B * u;
    x[2] = angle_limit_pi(x[2]);
    return x;
  }
  Matrix<double, 2, 1> motion_noise(Matrix<double, 2, 1> u, Matrix<double, 2, 2> Q)
  {
    Matrix<double, 2, 1> uw(gauss(0.0, Q(0, 0)), gauss(0.0, Q(1, 1)));
    return u + uw;
  }
  Matrix<double, 3, 1> observation_noise(Matrix<double, 3, 1> x, Matrix<double, 3, 3> R)
  {
    Matrix<double, 3, 1> xw(gauss(0.0, R(0, 0)), gauss(0.0, R(1, 1)), gauss(0.0, R(2, 2)));
    return x + xw;
  }

private:
  ros::NodeHandle _nh;

  ros::Publisher _ground_truth_pub;
  ros::Publisher _odom_pub;
  ros::Publisher _gps_pub;

  Matrix<double, 2, 2> Q;
  Matrix<double, 3, 3> R;
  Matrix<double, 3, 1> grund_truth;
  Matrix<double, 3, 1> odom;
  Matrix<double, 3, 1> gps;
};

int main(int argc, char ** argv)
{
  ros::init(argc, argv, "fake_sensor_publisher_node");
  ros::NodeHandle nh;
  FakeSensorPublisher node(nh);
  node.run();
  ros::spin();
  return 0;
}
