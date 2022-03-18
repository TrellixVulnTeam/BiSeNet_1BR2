"""
An example that uses TensorRT's Python api to make inferences for BiSeNet.
"""
import os
import shutil
import random
import sys
import threading
import time
import cv2
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt

def get_img_path_batches(batch_size, img_dir):
    ret = []
    batch = []
    for root, dirs, files in os.walk(img_dir):
        for name in files:
            if len(batch) == batch_size:
                ret.append(batch)
                batch = []
            batch.append(os.path.join(root, name))
    if len(batch) > 0:
        ret.append(batch)
    return ret

class BiSeNet_TRT(object):
    """
    description: A BiSeNet class that warps TensorRT ops, preprocess and postprocess ops.
    """

    def __init__(self, engine_file_path):
        # Create a Context on this device,
        self.cfx = cuda.Device(0).make_context()
        stream = cuda.Stream()
        runtime = trt.Runtime(trt.Logger(trt.Logger.INFO))
        assert runtime
        
        # Deserialize the engine from file
        with open(engine_file_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()

        host_inputs = []
        cuda_inputs = []
        host_outputs = []
        cuda_outputs = []
        bindings = []

        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(cuda_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                self.input_w = engine.get_binding_shape(binding)[-1]
                self.input_h = engine.get_binding_shape(binding)[-2]
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
            else:
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)

        # Store
        self.stream = stream
        self.context = context
        self.engine = engine
        self.host_inputs = host_inputs
        self.cuda_inputs = cuda_inputs
        self.host_outputs = host_outputs
        self.cuda_outputs = cuda_outputs
        self.bindings = bindings
        self.batch_size = engine.max_batch_size
        
        self.img_mean = (0.46962251, 0.4464104,  0.40718787)
        self.img_std = (0.27469736, 0.27012361, 0.28515933)

    def infer(self, image_raw):
        threading.Thread.__init__(self)
        # Make self the active context, pushing it on top of the context stack.
        self.cfx.push()
        # Restore
        stream = self.stream
        context = self.context
        engine = self.engine
        host_inputs = self.host_inputs
        cuda_inputs = self.cuda_inputs
        host_outputs = self.host_outputs
        cuda_outputs = self.cuda_outputs
        bindings = self.bindings
        print('ori_shape: ', image_raw.shape)
        # if image_raw is constant, image_raw.shape[1] != self.input_w
        w_ori, h_ori = image_raw.shape[1], image_raw.shape[0]
        # Do image preprocess
        input_image = self.preprocess_image(image_raw)
        # Copy input image to host buffer
        np.copyto(host_inputs[0], input_image.ravel())
        # Transfer input data to the GPU.
        cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
        # Run inference.
        context.execute_async(bindings=bindings, stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
        # Synchronize the stream
        stream.synchronize()
        # Remove any context from the top of the context stack, deactivating it.
        self.cfx.pop()
        # Here we use the first row of output in that batch_size = 1
        output = host_outputs[0]
        
        # Do postprocess
        output = output.reshape(self.input_h, self.input_w).astype('uint8')
        output = cv2.resize(output, (w_ori, h_ori), interpolation=cv2.INTER_NEAREST)
        
        #palette = np.random.randint(0, 256, (256, 3), dtype=np.uint8)
        #output = palette[output]
        return output

    def destroy(self):
        # Remove any context from the top of the context stack, deactivating it.
        self.cfx.pop()

    def preprocess_image(self, image_raw):
        """
        description: Read an image from image path
        param:
            image_raw: numpy, raw image
        return:
            img_input:  the processed image
        """
        
        img_resize = cv2.resize(image_raw, (self.input_w, self.input_h))[:, :, ::-1] #bgr2rgb
        img_input = np.transpose(img_resize, [2, 0, 1]) # hwc2chw
        
        img_input = img_input.astype(np.float32) / 255.0
        img_mean = np.array(self.img_mean, dtype=np.float32).reshape(3, 1, 1)
        img_std = np.array(self.img_std, dtype=np.float32).reshape(3, 1, 1)
        img_input = (img_input - img_mean) / img_std
        # expand dims
        img_input = np.expand_dims(img_input, axis=0)
        
        return img_input

    def get_raw_image(self, image_path_batch):
        """
        description: Read an image from image path
        """
        for img_path in image_path_batch:
            return cv2.imread(img_path)
    
    def get_raw_image_zeros(self, image_path_batch=None):
        """
        description: Ready data for warmup
        """
        for _ in range(self.batch_size):
            return np.zeros([self.input_h, self.input_w, 3], dtype=np.uint8)


class inferThread(threading.Thread):
    def __init__(self, bisenet, image_path_batch):
        threading.Thread.__init__(self)
        self.bisenet = bisenet
        self.image_path_batch = image_path_batch

    def run(self):
        batch_image_raw = self.bisenet.infer(self.bisenet.get_raw_image(self.image_path_batch))

class warmUpThread(threading.Thread):
    def __init__(self, bisenet):
        threading.Thread.__init__(self)
        self.bisenet = bisenet

    def run(self):
        batch_image_raw = self.bisenet.infer(self.bisenet.get_raw_image_zeros())



if __name__ == "__main__":
    # load custom engine
    engine_file_path = "build/model.trt"  # the generated engine file
    
    if len(sys.argv) > 1:
        engine_file_path = sys.argv[1]

    if os.path.exists('output/'):
        shutil.rmtree('output/')
    os.makedirs('output/')
    # a hrnet instance
    bisenet = BiSeNet_TRT(engine_file_path)
    try:
        print('batch size is', bisenet.batch_size)  # batch size is set to 1!
        
        image_dir = "samples/"
        image_path_batches = get_img_path_batches(bisenet.batch_size, image_dir)

        for i in range(10):
            # create a new thread to do warm_up
            thread1 = warmUpThread(bisenet)
            thread1.start()
            thread1.join()
        for batch in image_path_batches:
            # create a new thread to do inference
            thread1 = inferThread(bisenet, batch)
            thread1.start()
            thread1.join()
    finally:
        # destroy the instance
        bisenet.destroy()