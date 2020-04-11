#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <exception>

#include <ros/ros.h>

#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>

#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

// TODO:
// - add the weighted combination
// - move center camera to the center of the image
// - test the speed on an idle computer
// - test the OpenMP parallel for loop for warped combination
// - add the auto-calibration mode / get calibration from the transformation graph

// algorithms

struct ImageFeatures
{
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
};

ImageFeatures detect_and_describe(const cv::Mat &image)
{
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    auto detector = cv::ORB::create();
    ImageFeatures features;
    detector->detectAndCompute(gray, cv::Mat(), features.keypoints, features.descriptors);
    return features;
}

cv::Mat find_homography(const ImageFeatures &features1, const ImageFeatures &features2,
                        float ratio = 0.7, float reproj_thresh = 4.0)
{
    auto matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
    std::vector<cv::DMatch> matches;
    matcher->match(features1.descriptors, features2.descriptors, matches);
    std::sort(matches.begin(), matches.end());
    const int numGoodMatches = matches.size() * ratio;
    matches.erase(matches.begin() + numGoodMatches, matches.end());

    if (matches.size() < 15)
        throw std::runtime_error("no keypoints found");

    std::vector<cv::Point2f> points1, points2;
    points1.reserve(matches.size());
    points2.reserve(matches.size());

    for (auto match : matches)
    {
        points1.push_back(features1.keypoints[match.queryIdx].pt);
        points2.push_back(features2.keypoints[match.trainIdx].pt);
    }

    cv::Mat H = cv::findHomography(points1, points2, cv::RANSAC);

    return H;
}

class PanoramicNode
{
public:
    PanoramicNode(std::vector<std::string> in_topics, std::string out_topic);

private:
    void receive_img(const sensor_msgs::ImageConstPtr &img, int image_id);

    void process_imgs();

    ros::NodeHandle _nh;
    image_transport::ImageTransport _it;

    ros::Publisher _image_pub;
    std::vector<image_transport::Subscriber> _image_subs;

    std::vector<sensor_msgs::ImageConstPtr> _image_buffer;
    std::vector<cv::Mat> _image_homography;
    bool _computed_homography = false;
};

PanoramicNode::PanoramicNode(std::vector<std::string> in_topics, std::string out_topic)
    : _it(_nh)
{
    int num_cameras = in_topics.size();

    _image_pub = _nh.advertise<sensor_msgs::Image>(out_topic, 10);

    for (int i = 0; i < num_cameras; i++)
    {
        _image_subs.push_back(
            _it.subscribe(in_topics[i], 10,
                          [this, i](const sensor_msgs::ImageConstPtr &img) { this->receive_img(img, i); }));
    }

    _image_buffer.resize(num_cameras);
    std::fill(_image_buffer.begin(), _image_buffer.end(), nullptr);

    _image_homography.resize(num_cameras);
}

void PanoramicNode::receive_img(const sensor_msgs::ImageConstPtr &img, int image_id)
{
    ROS_WARN("received image %d", image_id);

    _image_buffer[image_id] = img;

    process_imgs();
}

void PanoramicNode::process_imgs()
{
    for (auto buff : _image_buffer)
    {
        if (buff == nullptr)
            return;
    }

    ROS_WARN("received all images");

    std::vector<cv::Mat> images;
    for (auto image_msg : _image_buffer)
    {
        auto img = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::BGR8);
        images.push_back(img->image);
    }

    if (!_computed_homography)
    {
        std::vector<ImageFeatures> features;
        for (const auto &img : images)
        {
            features.push_back(detect_and_describe(img));
        }

        _image_homography[0] = find_homography(features[0], features[1]);
        _image_homography[1] = cv::Mat::eye(3, 3, CV_32F);
        _image_homography[2] = find_homography(features[2], features[1]);

        ROS_WARN("computed image homography");

        _computed_homography = true;
    }

    // TODO: convert all images to float32

    cv::Mat result(720, 3000, CV_8UC3);

    #pragma omp parallel for
    for (int i = 0; i < images.size(); i++)
    {
        cv::Mat warp;
        cv::warpPerspective(images[i], warp, _image_homography[i], cv::Size(3000, 720));
        cv::warpPerspective(images[i], warp, _image_homography[i], cv::Size(3000, 720));
        cv::addWeighted(result, 1.0, warp, 1.0, 0.0, result);
    }

    cv_bridge::CvImage output;
    output.header = _image_buffer[1]->header;
    output.encoding = "bgr8";
    output.image = result;

    _image_pub.publish(output.toImageMsg());

    std::fill(_image_buffer.begin(), _image_buffer.end(), nullptr);

    ROS_WARN("processed on all images");
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "panoramic_node");

    std::vector<std::string> in_topics = {
        "/FL_camera/image_raw",
        "/FM_camera/image_raw",
        "/FR_camera/image_raw",
    };
    auto pano_node = PanoramicNode(in_topics, "panoramic");

    ros::spin();

    return 0;
}