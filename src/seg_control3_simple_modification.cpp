#include <ros/ros.h>
#include <ros/package.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <inference_engine.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <stdlib.h>
#include "neural_net_control/AlgorithmOutput.h"
#include "geometry_msgs/PoseStamped.h"
#include "geometry_msgs/Pose.h"
#include "geometry_msgs/Vector3.h"
#include <tf/transform_datatypes.h>
#include "std_msgs/Float32.h"


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

// initializing weights and vector directions
void initialize_vectors(std::vector<std::vector<double>>& weights,
                        std::vector<std::vector<std::pair<double, double>>>& vectors,
                        int height,
                        int width,
                        double* max_angle)
{
    std::ifstream config_file;
    config_file.open(ros::package::getPath("neural_net_control") + "/config/weights.txt");
    if (!config_file.is_open())
    {
        std::cout << "Couldn't open!" << std::endl;
        exit(EXIT_FAILURE);
    }   
    std::string line;
    while (std::getline(config_file, line))
    {
        std::vector<double> vec;
        std::istringstream iss(line);
        std::string weight;
        while (iss >> weight)
        {
            vec.push_back(std::stof(weight));
        }
        weights.push_back(vec);
    }
    config_file.close();

    int cells_h = weights.size();
    int cells_w = weights[0].size();
    
    double vx = 1.0;
    double vy = 0.0;
    *max_angle = 0;
    // retci x koordinate, stupci y koordinate
    for (int i = 0; i < cells_h; ++i)
    {
        double h1 = i * ((double)height / cells_h);
        double h2 = (i + 1) * ((double)height / cells_h);
        int x_offset = std::round(h1 + (h2 - h1)/2);
        
        std::vector<std::pair<double, double>> inner_vector;
        for (int k = 0; k < cells_w; ++k) 
        {
            inner_vector.push_back(std::pair<double, double>(vx, vy));
        }

        for (int j = 0; j < cells_w/2; ++j)
        {
            double w1 = j * ((double)width / cells_w);
            double w2 = (j + 1) * ((double)width / cells_w);
            int y_offset = std::round(w1 + (w2 - w1)/2);   
            
            double angle = atan2(width/2 - y_offset, height - x_offset);
            
            if (i == cells_h - 1)
            {
                if (angle > *max_angle)
                {
                    *max_angle = angle;
                }
            }
            
            double c = cos(angle);
            double s = sin(angle); 
            double x_vec = c*vx - s*vy;
            double y_vec = s*vx + c*vy;
            
            inner_vector[j].first = x_vec;
            inner_vector[j].second = y_vec;
            
            c = cos(-angle);
            s = sin(-angle);
            x_vec = c*vx - s*vy;
            y_vec = s*vx + c*vy;
            
            inner_vector[cells_w - 1 - j].first = x_vec;
            inner_vector[cells_w - 1 - j].second = y_vec;
        }
        vectors.push_back(inner_vector);
    }
    std::cout << "Maximum allowed angle: " << *max_angle << std::endl;
}

void calculate_ref(double* vel,
                   double* angle,
                   float* net_output,
                   std::vector<std::vector<std::pair<double, double>>>& vectors,
                   std::vector<std::vector<double>>& weights,
                   int height,
                   int width,
                   double limit,
                   double v_max,
                   double max_angle,
                   double max_possible_angle)
{
    double vx = 0;
    double vy = 0;
    
    int row_squares = weights.size();
    int col_squares = weights[0].size();
    
    for (int i = 0; i < row_squares; ++i)
    {
        int start_row = i * (height / row_squares);
        int end_row = (i + 1) * (height / row_squares);
        
        for (int j = 0; j < col_squares; ++j)
        {
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
            //std::cout << factor << ' ';
            vx += vectors[i][j].first * weights[i][j] * factor;
            vy += vectors[i][j].second * weights[i][j] * factor; 
        }
        //std::cout << std::endl;   
    }
    
    double norm = sqrt(vx*vx + vy*vy);
    if (norm == 0.0)
    {
        *angle = 0.0;
        *vel = 0.0;
        return;
    }
    vx /= norm;
    vy /= norm;
    
    double ratio = atan2(vy, vx)/max_possible_angle;
    *angle = ratio * max_angle;   
    *vel = (1 - abs(ratio)) * v_max;  
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
    
    std::vector<std::vector<double>> weights;
    std::vector<std::vector<std::pair<double, double>>> vectors;
    double max_possible_angle;
    initialize_vectors(weights, vectors, 256, 512, &max_possible_angle);
       
    // init node
    ros::init(argc, argv, "uav_seg");
    ros::NodeHandle nh;
    
    // init subscriber na sliku
    image_transport::ImageTransport it(nh);
    image_transport::Subscriber image_sub = it.subscribe("/uav/camera1/image_raw", 1, image_callback);

    // init publisher pozicije
    ros::Publisher pub = nh.advertise<geometry_msgs::Vector3>("/uav/pos_ref", 1);
    ros::Publisher ao_pub = nh.advertise<neural_net_control::AlgorithmOutput>("/uav/algorithm_output", 1);
    ros::Publisher euler_pub = nh.advertise<geometry_msgs::Vector3>("/uav/euler_ref", 1);

    ros::Publisher vel_cnn_pub = nh.advertise<std_msgs::Float32>("/neural_net_control/debug/cnn/velocity", 1);
    ros::Publisher yaw_cnn_pub = nh.advertise<std_msgs::Float32>("/neural_net_control/debug/cnn/yaw", 1);

    ros::Publisher vel_pt1_pub = nh.advertise<std_msgs::Float32>("/neural_net_control/debug/pt1/velocity", 1);
    ros::Publisher yaw_pt1_pub = nh.advertise<std_msgs::Float32>("/neural_net_control/debug/pt1/yaw", 1);



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

    ROS_INFO("Neural_net_control - Setting initial position and orientation."); 
    
    // Set initial position and orientation
    geometry_msgs::Vector3 position;
    position.x = 250.0;
    position.y = -114.0;
    position.z = 16.0;
    pub.publish(position);

    geometry_msgs::Vector3 orientation;
    orientation.z = 2.0;
    euler_pub.publish(orientation);

    ros::spinOnce();

    ros::Duration(5.0).sleep();
    ros::spinOnce();
    
    // vrijeme uzorkovanja za integraciju
    double T = 1.0/F;

    double vel_vec[2] = {1.0, 0.0};
    // rotiramo za početnu poziciju na cesti   
    double rotated_x = cos(0.25*PI + angle)*vel_vec[0] - sin(0.25*PI + angle)*vel_vec[1];
    vel_vec[1] = sin(0.25*PI + angle)*vel_vec[0] + cos(0.25*PI + angle)*vel_vec[1];
    vel_vec[0] = rotated_x;
    
    // trenutna brzina i kut
    double velocity = 0;
    double theta = 0;
    
    // parametri upravljanja
    double alpha, beta, v_max, max_angle, value_limit;
    nh.getParam("/ctrl1/alpha", alpha);
    nh.getParam("/ctrl1/beta", beta);
    nh.getParam("/ctrl1/value_limit", value_limit);
    nh.getParam("/ctrl1/v_max", v_max);
    nh.getParam("/ctrl1/max_angle", max_angle);   
    max_angle = max_angle/180.0 * PI;
    
    // vrijednosti brzine i kuta iz algoritma
    double ref_vel;
    double ref_angle_offset;
    neural_net_control::AlgorithmOutput alg_out;
    
    bool first_frame = true;

    // msgs
    std_msgs::Float32 vel_msg, yaw_msg;

    ROS_INFO("Neural_net_control - starting visual servo.");
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
            
            calculate_ref(&ref_vel,
                          &ref_angle_offset,
                          blob_output,
                          vectors,
                          weights,
                          256,
                          512,
                          value_limit,
                          v_max,
                          max_angle,
                          max_possible_angle);

            vel_msg.data = ref_vel;
            yaw_msg.data = ref_angle_offset;

            vel_cnn_pub.publish(vel_msg);
            yaw_cnn_pub.publish(yaw_msg);

            velocity = alpha * velocity + (1 - alpha) * ref_vel;
            theta = beta * theta + (1 - beta) * (ref_angle_offset + theta);

            vel_msg.data = velocity;
            yaw_msg.data = theta;

            vel_pt1_pub.publish(vel_msg);
            yaw_pt1_pub.publish(yaw_msg);
            
            pose.position.x += T * velocity * (cos(theta) * vel_vec[0] - sin(theta) * vel_vec[1]);
            pose.position.y += T * velocity * (sin(theta) * vel_vec[0] + cos(theta) * vel_vec[1]);
            pose.orientation.z = sin((angle + theta)/2.0);
            pose.orientation.w = cos((angle + theta)/2.0);
            //pub.publish(pose);

            // transform quaternions to euler angles
            tf::Quaternion q(pose_msg.pose.orientation.x, pose_msg.pose.orientation.y, pose_msg.pose.orientation.z, pose_msg.pose.orientation.w);
            tf::Matrix3x3 m(q);
            double roll, pitch, yaw;
            m.getRPY(roll, pitch, yaw);

            geometry_msgs::Vector3 euler_msg;
            // Simple 'move forward' in direction of camera view.
            euler_msg.x = -0.01;
            euler_msg.y = 0.01;
            // Add yaw offset to current yaw value.
            euler_msg.z = yaw + ref_angle_offset;
            euler_pub.publish(euler_msg);
            
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
