//
// Created by kris on 2021-11-07.
//

#ifndef JI_BISENETTRT_HPP
#define JI_BISENETTRT_HPP

#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "NvInferPlugin.h"
#include <cuda_runtime_api.h>
#include "NvInferRuntimeCommon.h"

#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <cstdlib>
#include <fstream>
#include <glog/logging.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/opencv.hpp>

#define STATUS int

#ifndef CUDA_CHECK
#define CUDA_CHECK(callstr)\
    {\
        cudaError_t error_code = callstr;\
        if (error_code != cudaSuccess) {\
            std::cerr << "CUDA error " << error_code << " at " << __FILE__ << ":" << __LINE__;\
            assert(0);\
        }\
    }
#endif  // CUDA_CHECK

using namespace nvinfer1;

class Logger: public ILogger {
    public:
        void log(Severity severity, const char* msg) noexcept override {
            if (severity == Severity::kINTERNAL_ERROR || severity == Severity::kERROR) {
                std::cout << msg << std::endl;
            }
        }
};

extern Logger gLogger;
/**
 * 使用BiSeNet实现的语义分割，并通过TensorRT进行推理。
 */

class BiSeNetTRT {

public:

    BiSeNetTRT();

    STATUS init(std::string serpth);

    /**
     * 反初始化函数
     */
    void unInit();

    /**
     * 对cv::Mat格式的图片进行分类，并输出预测分数前top排名的目标名称到mProcessResult
     * @param[in] image 输入图片
     * @param[out] detectResults 检测到的结果
     * @return 如果处理正常，则返回PROCESS_OK，否则返回`ERROR_*`
     */
    STATUS processImage(const cv::Mat &image, std::vector<int> &detectResults);


public:
    static const int ERROR_BASE = 0x0100;
    static const int ERROR_INVALID_INPUT = 0x0101;
    static const int ERROR_INIT = 0x0102;
    static const int ERROR_PROCESS = 0x0103;

    static const int SUCCESS_PROCESS = 0x1001;
    static const int SUCCESS_INIT = 0x1002;
    
    int inH, inW, outH, outW;

private:
    IRuntime* runtime {nullptr};
    ICudaEngine* engine {nullptr};
    IExecutionContext* context {nullptr};
    cudaStream_t stream;
    void* buffs[2];
};

#endif //JI_BISENETTRT_HPP