#include <vector>
#include <algorithm> 
#include <cstring>
#include <cstdlib>
#include <assert.h>
#include <cstdio>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>
#include <map>
#include <cmath>
#include <random>
#include <cstdio>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

class MatF{
public:
    float* m_data;
    int m_rows, m_cols, m_channels;

public:
    MatF(int cols, int rows, int channels){
        m_rows = rows;
        m_cols = cols;
        m_channels = channels;
        int size = channels * rows * cols * sizeof(float);
        m_data = (float*)malloc(size);
        memset((void *)m_data, 0, size);
    }

    float *at(int channel, int row, int col){
        assert(m_data != NULL);
        assert(row < m_rows);
        assert(col < m_cols);
        assert(channel < m_channels);

        return m_data + (channel * m_rows * m_cols) + row * m_cols + col;
    }

    int getRows() {return m_rows;}
    int getCols() {return m_cols;}
    int getChannels() {return m_channels;}
};

class CupidShuffle{
    private:
        std::unique_ptr<tvm::runtime::Module> handle;
        std::unique_ptr<tvm::runtime::Module> pose_handle;

    public:
        std::string deploy_lib_path;
        std::string deploy_graph_path;
        std::string deploy_param_path;
        bool gpu = true;
        int device_id = 0;
        int dtype_code = kDLFloat;
        int dtype_bits = 32;
        int dtype_lanes = 1;
        int device_type = kDLGPU;
        int image_width;
        int image_width;
        int total_input;
        int in_ndim = 4;
        int out_dim = 1;
        int n_classes;

        /**
         * \brief the initialization function to start the Yolo from config path
         * \param[config_path] the string for the /location/of/config.json
         * \return None
         */  
        CupidShuffle(std::string config_path) {
            // read config with nlohmann-json
            json model_config;
            std::ifstream json_read(config_path);
            json_read >> model_config;
            // read detector variables
            std::string lib_path = model_config["deploy_lib_path"];
            std::string graph_path = model_config["deploy_graph_path"];
            std::string param_path = model_config["deploy_param_path"];

            device_id = model_config["device_id"];
            image_width = model_config["image_width"];
            image_width = model_config["image_width"];
            gpu = model_config["gpu"];
            n_classes = model_config["n_classes"];
            n_classes_dim[1] = n_classes;
            thresh = model_config["thresh"];
            total_input = 1 * 3 * image_width * image_width;

            std::string deploy_lib_path = lib_path;
            std::string deploy_graph_path = graph_path;
            std::string deploy_param_path = param_path;

            if (gpu){
                device_type = kDLGPU;
            } else {
                device_type = kDLCPU;
            }
            // DETECTOR READ
            // read deploy lib
            tvm::runtime::Module mod_syslib = tvm::runtime::Module::LoadFromFile(deploy_lib_path);
            // read deplpy json
            std::ifstream json_in(deploy_graph_path, std::ios::in);
            std::string json_data((std::istreambuf_iterator<char>(json_in)), std::istreambuf_iterator<char>());
            json_in.close();
            // get global function module for graph runtime
            tvm::runtime::Module mod = (*tvm::runtime::Registry::Get("tvm.graph_runtime.create"))(json_data, mod_syslib,
                                                                                                device_type, device_id);
            this->handle.reset(new tvm::runtime::Module(mod));
            // parameters in binary
            std::ifstream params_in(deploy_param_path, std::ios::binary);
            std::string params_data((std::istreambuf_iterator<char>(params_in)), std::istreambuf_iterator<char>());
            params_in.close();
            // parameters need to be TVMByteArray type to indicate the binary data
            TVMByteArray params_arr;
            params_arr.data = params_data.c_str();
            params_arr.size = params_data.length();
            tvm::runtime::PackedFunc load_params = mod.GetFunction("load_params");
            load_params(params_arr);
        }
    
        /**
         * \brief function to normalize an image before it's processed by the network
         * \param[in] the raw cv::mat image
         * \return the normalized version of the iamge.
         */  
        cv::Mat preprocess_image(cv::Mat frame){
            cv::Size new_size = cv::Size(width, height); // or is it height width????
            cv::Mat resized_image;
            cv::Mat rgb;
            // bgr to rgb
            cv::cvtColor(frame, rgb,  cv::COLOR_BGR2RGB);
            // resize to 512x512
            cv::resize(rgb, resized_image, new_size);
            cv::Mat resized_image_floats(new_size, CV_32FC3);
            // convert resized image to floats and normalize
            resized_image.convertTo(resized_image_floats, CV_32FC3, 1.0f/255.0f);
            //mimic mxnets 'to_tensor' function
            cv::Mat normalized_image(new_size, CV_32FC3);
            // mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            cv::Mat mean(new_size, CV_32FC3, cv::Scalar(0.485, 0.456, 0.406));
            cv::Mat theta(new_size, CV_32FC3, cv::Scalar(0.229, 0.224, 0.225));
            cv::Mat temp;
            temp = resized_image_floats - mean;
            normalized_image = temp / theta;
            return normalized_image;
        }

        /**
         * \brief function to normalize an image before it's processed by the network
         * \param[in] the raw cv::mat image
         * \return the normalized version of the iamge.
         */  
          // we can set it externally with dynamic reconfigure
          MatF forward(cv::Mat frame)
          {
              //std::cout << "starting function" << std::endl;
              // get height/width dynamically
              cv::Size image_size = frame.size();
              float img_height = static_cast<float>(image_size.height);
              float img_width = static_cast<float>(image_size.width);

              int64_t in_shape[4] = {1, 3, image_width, image_width};
              int total_input = 3 * image_width * image_width;

              DLTensor *model_output;
              DLTensor *input;
              float *data_x = (float *) malloc(total_input * sizeof(float));

              //std::cout << "about to allocate info" << std::endl;
              TVMArrayAlloc(in_shape, in_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &input);
              TVMArrayAlloc(n_classes_dim, out_dim, d1ype_code, dtype_bits, dtype_lanes, device_type, device_id, &model_output);

              //copy processed image to DLTensor
              cv::Mat processed_image = preprocess_image(frame, image_width, image_width, true);
              cv::Mat split_mat[3];
              cv::split(processed_image, split_mat);
              memcpy(data_x, split_mat[2].ptr<float>(), processed_image.cols * processed_image.rows * sizeof(float));
              memcpy(data_x + processed_image.cols * processed_image.rows, split_mat[1].ptr<float>(),
                  processed_image.cols * processed_image.rows * sizeof(float));
              memcpy(data_x + processed_image.cols * processed_image.rows * 2, split_mat[0].ptr<float>(),
                  processed_image.cols * processed_image.rows * sizeof(float));
              TVMArrayCopyFromBytes(input, data_x, total_input * sizeof(float));   

              // standard tvm module run
              // get the module, set the module-input, and run the function
              // this is symbolic it ISNT run until TVMSync is performed
              tvm::runtime::Module *mod = (tvm::runtime::Module *) handle.get();
              tvm::runtime::PackedFunc set_input = mod->GetFunction("set_input");
              set_input("data", input);
              tvm::runtime::PackedFunc run = mod->GetFunction("run");
              run();
              tvm::runtime::PackedFunc get_output = mod->GetFunction("get_output");

              // https://github.com/apache/incubator-tvm/issues/979?from=timeline
              //"This may give you some ideas to start with.
              //In general you want to use pinned memory and you want
              //to interleave computation with copying; so you want to
              // be upload the next thing while you are computing the
              //current thing while you are downloading the last thing."
              TVMSynchronize(device_type, device_id, nullptr);
              get_output(0, model_output);

              // copy to output
              MatF output(1, n_classes, 1); 
              TVMArrayCopyToBytes(model_output, output.m_data, 1 * n_classes * 1 * sizeof(float));
              
              // free all the goods for next round
              TVMArrayFree(input);
              TVMArrayFree(model_output);
              input = nullptr;
              free(data_x);
              data_x = nullptr;       
              return output;
          }
};