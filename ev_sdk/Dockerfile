FROM 172.16.0.25/evtrain/ubuntu18.04-cuda10.1-cudnn7.5-opencv4.1-torch1.4-openvino2021r3:1.3

RUN rm -rf /usr/local/ev_sdk && mkdir -p /usr/local/ev_sdk
COPY ./ /usr/local/ev_sdk

RUN \
    cd /usr/local/ev_sdk && mkdir -p build && rm -rf build/* \
    && cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && make -j4 install && rm -rf ../build/*