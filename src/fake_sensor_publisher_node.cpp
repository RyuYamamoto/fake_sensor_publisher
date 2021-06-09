#include <nav_msgs/Odometry.h>
#include <ros/ros.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <Eigen/Core>
#include <cmath>
#include <iostream>

using namespace Eigen;

double deg2rad(const double degree) { return degree * M_PI / 180.0; }

double rad2deg(const double radian) { return radian * 180.0 / M_PI; }

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
  FakeSensorPublisher()
  : grund_truth(Matrix<double, 3, 1>::Zero()),
    odom(Matrix<double, 3, 1>::Zero()),
    gps(Matrix<double, 3, 1>::Zero())
  {
    Q << 0.1, 0, 0, deg2rad(30);
    Q = Q * Q;
    R << 2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, deg2rad(5);
    R = R * R;

    ground_truth_pub_ = pnh_.advertise<nav_msgs::Odometry>("grund_truth", 10);
    odom_pub_ = pnh_.advertise<nav_msgs::Odometry>("odom", 10);
    gps_pub_ = pnh_.advertise<nav_msgs::Odometry>("gps", 10);

    previous_stamp_ = ros::Time::now();

    timer_ = nh_.createTimer(ros::Duration(0.01), &FakeSensorPublisher::timerCallback, this);
  }
  ~FakeSensorPublisher() {}
  nav_msgs::Odometry inputToNavMsgs(Matrix<double, 3, 1> pose_2d, Matrix<double, 2, 1> input)
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
  void timerCallback(const ros::TimerEvent & e)
  {
    ros::Time current_stamp = ros::Time::now();

    nav_msgs::Odometry grund_truth_msg;
    nav_msgs::Odometry odom_msg;
    nav_msgs::Odometry gps_msgs;

    Matrix<double, 2, 1> u(1.0, deg2rad(5));

    const double dt = current_stamp.toSec() - previous_stamp_.toSec();

    // ground truth
    ROS_INFO("delta time: %f", dt);
    grund_truth = motionModel(grund_truth, u, dt);
    grund_truth_msg = inputToNavMsgs(grund_truth, u);

    // dead recogning
    Matrix<double, 2, 1> ud = motionNoise(u, Q);
    odom = motionModel(odom, ud, dt);
    odom_msg = inputToNavMsgs(odom, ud);

    // observation
    Matrix<double, 3, 1> gps = observationNoise(grund_truth, R);
    gps_msgs = inputToNavMsgs(gps, Matrix<double, 2, 1>::Zero());

    ground_truth_pub_.publish(grund_truth_msg);
    odom_pub_.publish(odom_msg);
    gps_pub_.publish(gps_msgs);

    previous_stamp_ = current_stamp;
  }
  Matrix<double, 3, 1> motionModel(Matrix<double, 3, 1> x, Matrix<double, 2, 1> u, double dt)
  {
    Matrix<double, 3, 3> F = Matrix<double, 3, 3>::Identity();
    Matrix<double, 3, 2> B;
    B << dt * std::cos(x[2]), 0, dt * std::sin(x[2]), 0, 0, dt;

    x = F * x + B * u;
    x[2] = angle_limit_pi(x[2]);
    return x;
  }
  Matrix<double, 2, 1> motionNoise(Matrix<double, 2, 1> u, Matrix<double, 2, 2> Q)
  {
    Matrix<double, 2, 1> uw(gauss(0.0, Q(0, 0)), gauss(0.0, Q(1, 1)));
    return u + uw;
  }
  Matrix<double, 3, 1> observationNoise(Matrix<double, 3, 1> x, Matrix<double, 3, 3> R)
  {
    Matrix<double, 3, 1> xw(gauss(0.0, R(0, 0)), gauss(0.0, R(1, 1)), gauss(0.0, R(2, 2)));
    return x + xw;
  }

private:
  ros::NodeHandle nh_{};
  ros::NodeHandle pnh_{"~"};
  ros::Timer timer_;
  ros::Time previous_stamp_;

  ros::Publisher ground_truth_pub_;
  ros::Publisher odom_pub_;
  ros::Publisher gps_pub_;

  Matrix<double, 2, 2> Q;
  Matrix<double, 3, 3> R;
  Matrix<double, 3, 1> grund_truth;
  Matrix<double, 3, 1> odom;
  Matrix<double, 3, 1> gps;
};

int main(int argc, char ** argv)
{
  ros::init(argc, argv, "fake_sensor_publisher_node");
  FakeSensorPublisher fake_sensor_publisher;
  ros::spin();
  return 0;
}
