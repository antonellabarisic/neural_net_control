#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <inference_engine.hpp>
#include <math.h>
#include "geometry_msgs/PoseStamped.h"
#include "geometry_msgs/Pose.h"
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
    image = cv_bridge::toCvCopy(msg, "bgr8") -> image;
}

// position callback
void pose_callback(const geometry_msgs::PoseStamped &msg)
{
    pose_msg = msg;
}

void calculate_ref(float* vel,
                   float* angle,
                   float* net_output,
                   std::vector<std::vector<std::pair<float, float>>>& vectors,
                   float* weights,
                   int row_squares,
                   int col_squares,
                   int height,
                   int width,
                   float limit,
                   float v_max,
                   float max_angle)
{
    double vx = 0;
    double vy = 0;
    
    for (int i = 0; i < row_squares; ++i)
    {
        for (int j = 0; j < col_squares; ++j)
        {
            int start_row = i * (height / row_squares);
            int end_row = (i + 1) * (height / row_squares);
            int start_col = j * (width / col_squares);
            int end_col = (j + 1) * (width / col_squares);
            
            double road_sum = 0;
            for (int k = start_row; k < end_row; ++k)
            {
                for (int l = start_col; l < end_col; ++l)
                {
                    if (net_output[k*width + l] >= limit)
                    {
                        road_sum += net_output[k*width + l];
                    }
                }
            }
            
            double factor = road_sum / ((end_row - start_row) * (end_col - start_col));
            vx += vectors[i][j].first * weights[i*col_squares + j] * factor;
            vy += vectors[i][j].second * weights[i*col_squares + j] * factor;
            //vx += vectors[i][j].first * weights[i*col_squares + j] * road_sum;
            //vy += vectors[i][j].second * weights[i*col_squares + j] * road_sum; 
        }   
    }
    
    double norm = sqrt(vx*vx + vy*vy);
    vx /= norm;
    vy /= norm;
    
    *angle = atan2(vy, vx);
    if (abs(*angle) > max_angle)
    {
        if (*angle >= 0)
        {
            *angle = max_angle;
        }
        else
        {
            *angle = -max_angle;
        }
    }
    
    *vel = vx * v_max;
    
    std::cout << "Angle: " << *angle << " Velocity: " << *vel << std::endl;
    
}


int main(int argc, char **argv)
{   
    // provjera da li je zadan model
    if (argc == 1)
    {
        std::cerr << "Not enough input arguments, specifiy model path!" << std::endl;
        return 1;
    }
    
    float weights[3][5] = {{0.1, 0.2, 0.2, 0.2, 0.1}, {0.2, 0.8, 0.8, 0.8, 0.2}, {0.4, 1, 1, 1, 0.4}};
    float deg_angles[3][5] = {{43.83, 25.64, 0, -25.64, -43.83}, {57.99, 38.66, 0, -38.66, -57.99}, {78.23, 67.38, 0, -67.38, -78.23}};
    
    float vx = 1.0;
    float vy = 0.0;
    
    std::vector<std::vector<std::pair<float, float>>> vectors;  
    for (int i = 0; i < 3; ++i)
    {
        std::vector<std::pair<float, float>> inner_vector;
        for (int j = 0; j < 5; ++j)
        {   
            float rad = deg_angles[i][j]/180.0 * PI;
            float c = cos(rad);
            float s = sin(rad);
            
            float x = c*vx - s*vy;
            float y = s*vx + c*vy;
            
            inner_vector.push_back(std::pair<float, float>(x, y));
        }
        vectors.push_back(inner_vector);
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
    float angle;
    nh.getParam("/ctrl1/angle", angle);
    angle = angle * PI;
    
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
    
    // vrijeme uzorkovanja za integraciju
    float T = 1.0/F;

    float vel_vec[2] = {1.0, 0.0};
    // rotiramo za početnu poziciju na cesti   
    float rotated_x = cos(0.25*PI + angle)*vel_vec[0] - sin(0.25*PI + angle)*vel_vec[1];
    vel_vec[1] = sin(0.25*PI + angle)*vel_vec[0] + cos(0.25*PI + angle)*vel_vec[1];
    vel_vec[0] = rotated_x;
    
    // trenutna brzina i kut
    float vel = 0;
    float theta = 0;
    
    // parametri upravljanja
    float alpha, beta, v_max, max_angle;
    nh.getParam("/ctrl1/alpha", alpha);
    nh.getParam("/ctrl1/beta", beta);
    nh.getParam("/ctrl1/v_max", v_max);
    nh.getParam("/ctrl1/max_angle", max_angle);
    max_angle = max_angle/180.0 * PI;
    
    // vrijednosti brzine i kuta iz algoritma
    float velocity;
    float turn_angle;
    
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
            
            calculate_ref(&velocity,
                          &turn_angle,
                          blob_output,
                          vectors,
                          weights[0],
                          3,
                          5,
                          256,
                          512,
                          0.5,
                          v_max,
                          max_angle);

            vel = (1.0 - alpha)*velocity + alpha*vel;
            theta = (1.0 - beta)*(turn_angle + theta) + beta*theta;
            
            pose.position.x += T*vel*(cos(theta)*vel_vec[0] - sin(theta)*vel_vec[1]);
            pose.position.y += T*vel*(sin(theta)*vel_vec[0] + cos(theta)*vel_vec[1]);
            pose.orientation.z = sin((angle + theta)/2.0);
            pose.orientation.w = cos((angle + theta)/2.0);
            pub.publish(pose);
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
