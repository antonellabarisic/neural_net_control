#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <inference_engine.hpp>
#include <math.h>
#include "geometry_msgs/PoseStamped.h"
#include "geometry_msgs/Pose.h"
#include "geometry_msgs/Point.h"
#include "geometry_msgs/Quaternion.h"
#include <string.h>


using namespace InferenceEngine;

constexpr float F = 25;
constexpr float PI = 3.14159265358979323846;

static cv::Mat image;
static geometry_msgs::PoseStamped pose_msg;

// wrap slike
static Blob::Ptr wrapMat2Blob(const cv::Mat &mat) {
    size_t channels = mat.channels();
    size_t height = mat.size().height;
    size_t width = mat.size().width;

    TensorDesc tDesc(Precision::U8,
                     {1, channels, height, width},
                     Layout::NHWC);

    return make_shared_blob<uint8_t>(tDesc, mat.data);
}

// image callback
void image_callback(const sensor_msgs::ImageConstPtr& msg)
{
    image = cv_bridge::toCvCopy(msg, "bgr8") -> image;
}

// position callback
void pose_callback(const geometry_msgs::PoseStamped &msg)
{
    pose_msg = msg;
}

// dodavanje segmentacije
cv::Mat add_segmentation(cv::Mat& image, float* net_output, int height, int width)
{
    cv::Mat resized(height, width, CV_8U);
    cv::resize(image, resized, resized.size());
    
    cv::Mat labels(height, width, CV_32FC1, net_output);
    labels.convertTo(labels, CV_8UC1);
    labels *= 255;
    
    std::vector<cv::Mat> channels(3);
    cv::split(resized, channels);
    cv::add(labels, channels[1], channels[1]);
    
    cv::Mat result;
    cv::merge(channels, result);
    return result;
}

cv::Mat segmented_img(float* net_output, int height, int width)
{
    cv::Mat labels(height, width, CV_32FC1, net_output);
    labels.convertTo(labels, CV_8UC1, 255);
    
    cv::Mat green = cv::Mat::zeros(height, width, CV_8UC1);
    cv::Mat red = cv::Mat::zeros(height, width, CV_8UC1);
    
    cv::Mat result;
    cv::merge(std::vector<cv::Mat>{labels, green, red}, result);
    
    return result;
}


int main(int argc, char **argv)
{   
    // provjera da li je zadan model
    if (argc == 1)
    {
        std::cerr << "Not enough input arguments, specifiy model path!" << std::endl;
        return 1;
    }
    
    // init node
    ros::init(argc, argv, "uav_seg");
    ros::NodeHandle nh;
    
    // init subscriber na sliku
    image_transport::ImageTransport it(nh);
    image_transport::Subscriber image_sub = it.subscribe("/uav/camera1/image_raw", 1, image_callback);

    // init publisher pozicije
    ros::Publisher pub = nh.advertise<geometry_msgs::Pose>("/uav/pose_ref", 1);
    // init subscriber pozicije
    ros::Subscriber pose_sub = nh.subscribe("/uav/pose", 1, pose_callback);
    ros::Rate loop_rate(F);
    
    Core ie;
    
    // ucitavamo mrezu i parametre
    CNNNetReader network_reader;
    std::string model_name(argv[1]);
    network_reader.ReadNetwork(argv[1]);
    std::string bin_file = model_name.erase(model_name.rfind('.')) + ".bin";
    network_reader.ReadWeights(bin_file);
    
    CNNNetwork network = network_reader.getNetwork();

    // input postavke
    InputsDataMap input_info(network.getInputsInfo());
    InputInfo::Ptr input = input_info.begin() -> second;
    input -> getPreProcess().setResizeAlgorithm(RESIZE_BILINEAR);
    input -> setPrecision(Precision::U8);
    std::string input_name = input_info.begin() -> first;
    
    // output postavke
    OutputsDataMap output_info(network.getOutputsInfo());
    auto output = output_info.begin() -> second;
    output -> setPrecision(Precision::FP32);
    std::string output_name = output_info.begin() -> first;
    
    // load mreze na stick
    ExecutableNetwork exec_network = ie.LoadNetwork(network, "MYRIAD");
    InferRequest::Ptr next_request = exec_network.CreateInferRequestPtr();
    InferRequest::Ptr current_request = exec_network.CreateInferRequestPtr();
    
    // postavi zeljenu pocetnu poziciju
    float angle = 0.5*PI;
    geometry_msgs::Pose pose;
    pose.position.x = 250.0;
    pose.position.y = -114.0;
    pose.position.z = 14.1;
    pose.orientation.x = 0.0;
    pose.orientation.y = 0.0;
    pose.orientation.z = sin(angle/2);
    pose.orientation.w = cos(angle/2);    

    // ros petlja za postavljanje na pocetnu poziciju
    while (ros::ok())
    {
        ros::spinOnce();
        pub.publish(pose);
        if (abs(pose_msg.pose.position.x - pose.position.x) < 0.005 &&
            abs(pose_msg.pose.position.y - pose.position.y) < 0.005 &&
            abs(pose_msg.pose.position.z - pose.position.z) < 0.005 &&
            abs(pose_msg.pose.orientation.x - pose.orientation.x) < 0.005 &&
            abs(pose_msg.pose.orientation.y - pose.orientation.y) < 0.005 &&
            abs(pose_msg.pose.orientation.z - pose.orientation.z) < 0.005 &&
            abs(pose_msg.pose.orientation.w - pose.orientation.w) < 0.005)
        {
            break;
        }
        loop_rate.sleep();
    }
    
    cv::VideoWriter video_rgb("seg_out_rgb.avi", cv::VideoWriter::fourcc('M','J','P','G'), 30, cv::Size(512, 256));
    cv::VideoWriter video("seg_out.avi", cv::VideoWriter::fourcc('M','J','P','G'), 30, cv::Size(512, 256));
    
    bool first_frame = true;
    // ros petlja
    while (ros::ok())
    {
        if (first_frame)
        {
            // ovaj upit se šalje samo za prvi frame
            current_request -> SetBlob(input_name, wrapMat2Blob(image));
            current_request -> StartAsync();
            // spin jednom da pozovemo callback i dobijemo novi frame.
            ros::spinOnce();
            first_frame = false;
        }

        next_request -> SetBlob(input_name, wrapMat2Blob(image));
        next_request -> StartAsync();

        if (OK == current_request -> Wait(IInferRequest::WaitMode::RESULT_READY))
        {
            Blob::Ptr blob = current_request -> GetBlob(output_name);
            float* blob_output = blob -> buffer().as<PrecisionTrait<Precision::FP32>::value_type*>();
            cv::Mat img_rgb = add_segmentation(image, blob_output, 256, 512);
            video_rgb.write(img_rgb);
            cv::imshow("Segmentation RGB", img_rgb);
            
            cv::Mat img = segmented_img(blob_output, 256, 512);
            video.write(img);
            cv::imshow("Segmentation", img);
            cv::waitKey(1);
        }

        ros::spinOnce();
        if (image.empty()) {
            break;
        }

        current_request.swap(next_request);          
        loop_rate.sleep();
    }
    
    video.release();
    video_rgb.release();
    cv::destroyAllWindows();
       
    return 0;
}
