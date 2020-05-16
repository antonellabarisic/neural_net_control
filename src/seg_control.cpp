#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <inference_engine.hpp>
#include <math.h>
#include <string.h>
//#include <chrono>
#include "neural_net_control/AlgorithmOutput.h"
#include "geometry_msgs/PoseStamped.h"
#include "geometry_msgs/Pose.h"

using namespace InferenceEngine;

constexpr double F = 25;
constexpr double PI = 3.14159265358979323846;

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

void calculate_ref(double* ref_vel,
                   double* ref_angle_offset,
                   float* net_output,
                   int start_row,
                   int end_row,
                   int start_col,
                   int end_col,
                   int n,
                   double limit,
                   double v_max,
                   double max_angle)
{
    double image_sum = 0;
    double left_image_sum = 0;
    for (int i = start_row; i < end_row; ++i)
    {
        for (int j = start_col; j < end_col; ++j)
        {
            if (net_output[i*n + j] >= limit)
            {
                image_sum += net_output[i*n + j];
                if (j < start_col + (end_col - start_col)/2)
                {
                    left_image_sum += net_output[i*n + j];
                }
            }
        }
    }
    
    double ratio;
    if (image_sum == 0.0) 
    {
        ratio = 0.0;
    }
    else
    {
        ratio = left_image_sum / image_sum - 0.5;    
    }
    
    *ref_angle_offset = max_angle * ratio;
    *ref_vel = -v_max*abs(ratio) + v_max;
    std::cout << "Angle: " << *ref_angle_offset << " Velocity: " << *ref_vel << std::endl;
    
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
    ros::Publisher ao_pub = nh.advertise<neural_net_control::AlgorithmOutput>("/uav/algorithm_output", 1);
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
    double angle;
    nh.getParam("/ctrl1/angle", angle);
    
    geometry_msgs::Pose pose;
    nh.getParam("/ctrl1/x", pose.position.x);
    nh.getParam("/ctrl1/y", pose.position.y);
    nh.getParam("/ctrl1/z", pose.position.z);
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
    
    double T = 1.0/F;
    
    double vel_vec[2] = {1.0, 0.0};
    // rotiramo za početnu poziciju na cesti
    // 0.25
    double rotated_x = cos(0.25*PI + angle)*vel_vec[0] - sin(0.25*PI + angle)*vel_vec[1];
    vel_vec[1] = sin(0.25*PI + angle)*vel_vec[0] + cos(0.25*PI + angle)*vel_vec[1];
    vel_vec[0] = rotated_x;
    
    // trenutna brzina i kut
    double velocity = 0;
    double theta = 0;
    
    // parametri upravljanja
    double alpha, beta, bottom_limit, upper_limit, left_limit, right_limit, value_limit, v_max, max_angle;
    nh.getParam("/ctrl1/alpha", alpha);
    nh.getParam("/ctrl1/beta", beta);
    nh.getParam("/ctrl1/bottom_limit", bottom_limit);
    nh.getParam("/ctrl1/upper_limit", upper_limit);
    nh.getParam("/ctrl1/left_limit", left_limit);
    nh.getParam("/ctrl1/right_limit", right_limit);
    nh.getParam("/ctrl1/value_limit", value_limit);
    nh.getParam("/ctrl1/v_max", v_max);
    nh.getParam("/ctrl1/max_angle", max_angle);
    max_angle = max_angle/180.0 * PI;

    // vrijednosti brzine i kuta iz algoritma
    double ref_vel;
    double ref_angle_offset;
    neural_net_control::AlgorithmOutput alg_out;
    
    bool first_frame = true;
    //std::chrono::steady_clock::time_point begin;
    //std::chrono::steady_clock::time_point end;
    // ros petlja
    while (ros::ok())
    {
        if (first_frame)
        {
            //begin = std::chrono::steady_clock::now();
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
            
            calculate_ref(&ref_vel,
                          &ref_angle_offset,
                          blob_output,
                          bottom_limit*256,
                          upper_limit*256,
                          left_limit*512,
                          right_limit*512,
                          512,
                          value_limit,
                          v_max,
                          max_angle);

            //end = std::chrono::steady_clock::now();
            //double time_period = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9;
            
            velocity = alpha * velocity + (1 - alpha) * ref_vel;
            theta = beta * theta + (1 - beta) * (ref_angle_offset + theta);
            std::cout  << "Actual theta: " << theta << " Actual vel: " << velocity << std::endl;
            //begin = std::chrono::steady_clock::now();
            
            pose.position.x += T * velocity * (cos(theta) * vel_vec[0] - sin(theta) * vel_vec[1]);
            pose.position.y += T * velocity * (sin(theta) * vel_vec[0] + cos(theta) * vel_vec[1]);
            pose.orientation.z = sin((angle + theta)/2.0);
            pose.orientation.w = cos((angle + theta)/2.0);
            pub.publish(pose);
            
            alg_out.theta = theta;
            alg_out.velocity = velocity;
            alg_out.header.stamp = ros::Time::now();
            ao_pub.publish(alg_out);
        }

        ros::spinOnce();
        if (image.empty()) {
            break;
        }

        current_request.swap(next_request);
        loop_rate.sleep();
    }
       
    return 0;
}
