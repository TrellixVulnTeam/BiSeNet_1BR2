/**
 * 示例代码：实现ji.h定义的图像接口，开发者需要根据自己的实际需求对接口进行实现
 */

#include <cstdlib>
#include <cstring>
#include <fstream>

#include <opencv2/opencv.hpp>
#include <glog/logging.h>

#include "cJSON.h"
#include "ji.h"

#include "BiSeNetTRT.hpp"

char *jsonResult = nullptr; // 用于存储算法处理后输出到JI_EVENT的json字符串，根据ji.h的接口规范，接口实现需要负责释放该资源
cv::Mat outputFrame;

/**
 * 使用predictor对输入图像inFrame进行处理
 *
 * @param[in] predictor 算法句柄
 * @param[in] inFrame 输入图像
 * @param[in] args 处理当前输入图像所需要的输入参数，例如在目标检测中，通常需要输入ROI，由开发者自行定义和解析
 * @param[out] outFrame 输入图像，由内部填充结果，外部代码需要负责释放其内存空间
 * @param[out] event 以JI_EVENT封装的处理结果
 * @return 如果处理成功，返回JISDK_RET_SUCCEED
 */
int processMat(BiSeNetTRT *detector, const cv::Mat &inFrame, const char* args, cv::Mat &outFrame, JI_EVENT &event) {
    // 处理输入图像
    if (inFrame.empty()) {
        return JISDK_RET_FAILED;
    }
    cJSON * root = NULL;
    cJSON * item = NULL;//cjson对象
    
    root = cJSON_Parse(args);
    if (!root) {
        return JISDK_RET_FAILED;
    }
    else {
        item = cJSON_GetObjectItem(root, "mask_output_path");
        if (!item) {
            return JISDK_RET_FAILED;
        }
    }

    // 针对每个ROI进行算法处理
    std::vector<int> detectResults(detector->outW * detector->outH);
    int processRet = detector->processImage(inFrame, detectResults);
    if (processRet != BiSeNetTRT::SUCCESS_PROCESS) {
        return JISDK_RET_FAILED;
    }

    cv::Mat pred(cv::Size(detector->outW, detector->outH), CV_8UC1);
    int idx{0};
    for (int i{0}; i < detector->outH; ++i) {
        uint8_t *ptr = pred.ptr<uint8_t>(i);
        for (int j{0}; j < detector->outW; ++j) {
            ptr[j] = detectResults[idx]; // == 1 ? 255 : 0;
            ++idx;
        }
    }

    if ((inFrame.rows != detector->outH) || inFrame.cols != detector->outW) {
        cv::resize(pred, pred, cv::Size(inFrame.cols, inFrame.rows), cv::INTER_NEAREST);
    }
    
    cv::imwrite(item->valuestring, pred);
    
    // 将结果封装成json字符串
    cJSON *rootObj = cJSON_CreateObject();
    cJSON_AddItemToObject(rootObj, "mask", cJSON_CreateString(item->valuestring));

    char *jsonResultStr = cJSON_Print(rootObj);
    int jsonSize = strlen(jsonResultStr);
    if (jsonResult == nullptr) {
        jsonResult = new char[jsonSize + 1];
    } else if (strlen(jsonResult) < jsonSize) {
        free(jsonResult);   // 如果需要重新分配空间，需要释放资源
        jsonResult = new char[jsonSize + 1];
    }
    strcpy(jsonResult, jsonResultStr);

    event.json = jsonResult;
    event.code = JISDK_RET_SUCCEED;

    if (rootObj)
        cJSON_Delete(rootObj);
    if (jsonResultStr)
        free(jsonResultStr);

    return JISDK_RET_SUCCEED;
}

int ji_init(int argc, char **argv) {
    return JISDK_RET_SUCCEED;
}


void *ji_create_predictor(int pdtype) {
    auto *detector = new BiSeNetTRT();
    
    int iRet = 0;
    try {

        iRet = detector->init("/usr/local/ev_sdk/model/model.trt");

    }catch(...) {

        system("/usr/local/ev_sdk/3rd/BiSeNet/bin/segment compile /usr/local/ev_sdk/model/model_sim.onnx /usr/local/ev_sdk/model/model.trt");

        iRet = detector->init("/usr/local/ev_sdk/model/model.trt");
    }
    
    if (iRet != BiSeNetTRT::SUCCESS_INIT) {
        return nullptr;
    }
    LOG(INFO) << "BiSeNetTRT init OK.";

    return detector;
}

void ji_destroy_predictor(void *predictor) {
    if (predictor == NULL) return;

    auto *detector = reinterpret_cast<BiSeNetTRT *>(predictor);
    detector->unInit();
    delete detector;

    if (jsonResult) {
        free(jsonResult);
        jsonResult = nullptr;
    }
}

int ji_calc_frame(void *predictor, const JI_CV_FRAME *inFrame, const char *args,
                  JI_CV_FRAME *outFrame, JI_EVENT *event) {
    if (predictor == NULL || inFrame == NULL) {
        return JISDK_RET_INVALIDPARAMS;
    }

    auto *detector = reinterpret_cast<BiSeNetTRT *>(predictor);
    cv::Mat inMat(inFrame->rows, inFrame->cols, inFrame->type, inFrame->data, inFrame->step);
    if (inMat.empty()) {
        return JISDK_RET_FAILED;
    }
    int processRet = processMat(detector, inMat, args, outputFrame, *event);

    if (processRet == JISDK_RET_SUCCEED) {
        if ((event->code != JISDK_CODE_FAILED) && (!outputFrame.empty()) && (outFrame)) {
            outFrame->rows = outputFrame.rows;
            outFrame->cols = outputFrame.cols;
            outFrame->type = outputFrame.type();
            outFrame->data = outputFrame.data;
            outFrame->step = outputFrame.step;
        }
    }
    return processRet;
}

int ji_calc_buffer(void *predictor, const void *buffer, int length, const char *args, const char *outFile,
                   JI_EVENT *event) {
    return JISDK_RET_UNUSED;
}

int ji_calc_file(void *predictor, const char *inFile, const char *args, const char *outFile, JI_EVENT *event) {
    return JISDK_RET_UNUSED;
}

int ji_calc_video_file(void *predictor, const char *infile, const char* args,
                       const char *outfile, const char *jsonfile) {
    return JISDK_RET_UNUSED;
}

void ji_reinit() {}