//
// Created by kris on 2021-11-07.
//
#include "BiSeNetTRT.hpp"
Logger gLogger;

BiSeNetTRT::BiSeNetTRT() {
}

int BiSeNetTRT::init(std::string serpth) {
    if(serpth.empty()) {
        LOG(ERROR) << "Invalid init args";
        return BiSeNetTRT::ERROR_INVALID_INPUT;
    }
    
    std::ifstream ifile(serpth, std::ios::in | std::ios::binary);
    if (!ifile) {
        LOG(ERROR) << "read serialized file failed\n";
        throw -1;
    }

    ifile.seekg(0, std::ios::end);
    const int mdsize = ifile.tellg();
    ifile.clear();
    ifile.seekg(0, std::ios::beg);
    std::vector<char> buf(mdsize);
    ifile.read(&buf[0], mdsize);
    ifile.close();
    LOG(INFO) << "model size: " << mdsize;

    runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    
    engine = runtime->deserializeCudaEngine((void*)&buf[0], mdsize, nullptr);
    if(engine == nullptr) throw -1;
    
    context = engine->createExecutionContext();
    assert(context != nullptr);
    
    Dims3 i_dims = static_cast<Dims3&&>(
        engine->getBindingDimensions(engine->getBindingIndex("input_image")));
    Dims3 o_dims = static_cast<Dims3&&>(
        engine->getBindingDimensions(engine->getBindingIndex("preds")));

    inH = i_dims.d[2], inW = i_dims.d[3];
    
    outH = o_dims.d[1], outW = o_dims.d[2];
    
    CUDA_CHECK(cudaMalloc(&buffs[0], 3 * inH * inW * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buffs[1], outH * outW * sizeof(int)));
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    LOG(INFO) << "Init Done!";
    
    return BiSeNetTRT::SUCCESS_INIT;
}

void BiSeNetTRT::unInit() {

    cudaStreamDestroy(stream);
    CUDA_CHECK(cudaFree(buffs[0]));
    CUDA_CHECK(cudaFree(buffs[1]));
    
    context->destroy();
    engine->destroy();
    runtime->destroy();
}

STATUS BiSeNetTRT::processImage(const cv::Mat &image, std::vector<int> &detectResults) {
    // resize
    cv::Mat im(inW, inH, CV_8UC3);
    cv::resize(image, im, cv::Size(inW, inH), cv::INTER_CUBIC);
    
    // normalize and convert to rgb
    std::array<float, 3> mean{0.485f, 0.456f, 0.406f};
    std::array<float, 3> variance{0.229f, 0.224f, 0.225f};
    float scale = 1.f / 255.f;
    for (int i{0}; i < 3; ++ i) {
        variance[i] = 1.f / variance[i];
    }
    std::vector<float> data(inH * inW * 3);
    for (int h{0}; h < inH; ++h) {
        cv::Vec3b *p = im.ptr<cv::Vec3b>(h);
        for (int w{0}; w < inW; ++w) {
            for (int c{0}; c < 3; ++c) {
                int idx = (2 - c) * inH * inW + h * inW + w; // to rgb order
                data[idx] = (p[w][c] * scale - mean[c]) * variance[c];
            }
        }
    }
    
    // transfer data to gpu
    CUDA_CHECK(cudaMemcpyAsync(buffs[0], &data[0], 3 * inH * inW * sizeof(float), cudaMemcpyHostToDevice, stream));
    // infer
    context->enqueueV2(&buffs[0], stream, nullptr);
    // transfer data to cpu
    CUDA_CHECK(cudaMemcpyAsync(&detectResults[0], buffs[1], outH * outW * sizeof(int), cudaMemcpyDeviceToHost, stream));
    
    cudaStreamSynchronize(stream);

    return BiSeNetTRT::SUCCESS_PROCESS;
}






