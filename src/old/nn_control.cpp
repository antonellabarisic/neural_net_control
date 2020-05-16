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

constexpr float F = 30;
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
    image = cv_bridge::toCvCopy(msg, "mono8") -> image;
}

// position callback
void pose_callback(const geometry_msgs::PoseStamped &msg)
{
    pose_msg = msg;
}

int main(int argc, char **argv)
{   

    // provjera da li je zadan model
    if (argc == 1)
    {
        std::cerr << "Not enough input arguments, specifiy model path!" << std::endl;
        return 1;
    }
    
    float V_max = std::stof(argv[2]);
    float alpha = std::stof(argv[3]);
    float beta = std::stof(argv[4]);
    
    // init node
    ros::init(argc, argv, "uav_nn_control");
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
    
    // postavi batch size na 1
    CNNNetwork network = network_reader.getNetwork();
    network.setBatchSize(1);
    
    // input postavke
    InputInfo::Ptr input_info = network.getInputsInfo().begin() -> second;
    std::string input_name = network.getInputsInfo().begin() -> first;   
    input_info -> getPreProcess().setResizeAlgorithm(RESIZE_BILINEAR);
    input_info -> setLayout(Layout::NHWC);
    input_info -> setPrecision(Precision::U8);
    
    // output postavke
    std::vector<std::string> output_names;
    OutputsDataMap output_info(network_reader.getNetwork().getOutputsInfo());
    for (auto &output : output_info)
    {
        output_names.push_back(output.first);
        output.second -> setPrecision(Precision::FP32);
    }
    
    // load mreze na stick
    ExecutableNetwork exec_network = ie.LoadNetwork(network, "MYRIAD");
    InferRequest infer_request = exec_network.CreateInferRequest();
    
    // postavi zeljenu pocetnu poziciju
    float angle = 0.75*PI;
    geometry_msgs::Pose pose;
    pose.position.x = 6.0;
    pose.position.y = 0.0;
    pose.position.z = 1.0;
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
    
    // postavi početne vrijednosti brzine i kutnog pomaka.
    float vel = V_max;
    float vel_vec[2] = {-1.0, 0.0};
    float theta = 0.0;
    float T = 1.0/F;
    float collision_prob;
    float yaw;
    Blob::Ptr output;

    // ros petlja
    while (ros::ok())
    {
        // primi sliku od callbacka i ponavljaj dok ne dobijes
        ros::spinOnce();
        if (image.empty())
        {
            continue;
        }
        
        // inferencija mreze
        Blob::Ptr img_blob = wrapMat2Blob(image);
        infer_request.SetBlob(input_name, img_blob);      
        infer_request.Infer();
        
        // obrada izlaza
        output = infer_request.GetBlob(output_names[0]);
        collision_prob = 1 - *(output -> buffer().as<PrecisionTrait<Precision::FP32>::value_type*>());
        output = infer_request.GetBlob(output_names[1]);
        yaw = *(output -> buffer().as<PrecisionTrait<Precision::FP32>::value_type*>());
        
        //std::cout << "collision_prob: " << collision_prob << std::endl;
        //std::cout << "yaw: " << yaw << std::endl;
        
        // mijenjaj brzinu i zakret
        vel = (1.0 - alpha)*vel + alpha*collision_prob*V_max;
        theta = (1.0 - beta)*theta + beta*PI*0.5*yaw;
        
        //std::cout << "vel: " << vel << std::endl;
        std::cout << "theta, theta+angle: " << theta << " " << theta+angle << std::endl;
        
        // mijenjaj poruku pozicije
        pose.position.x += T*vel*(cos(theta)*vel_vec[0] - sin(theta)*vel_vec[1]);
        pose.position.y += T*vel*(sin(theta)*vel_vec[0] + cos(theta)*vel_vec[1]);
        pose.orientation.z = sin((angle + theta)/2.0);
        pose.orientation.w = cos((angle + theta)/2.0);
        
        //std::cout << pose.position.x << " " << pose.position.y << " " << pose.orientation.z << " " << pose.orientation.w << std::endl;
        
        pub.publish(pose);
           
        loop_rate.sleep();
    }
       
    return 0;
}
