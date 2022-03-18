from trt import *
import json
import cv2
import numpy as np
import os

def init():

    """Initialize model
    Returns: model
    """
    onnx_file_path = "/project/train/models/model_sim.onnx"
    engine_file_path = "/usr/local/ev_sdk/model/model.trt"
    os.system("/usr/local/ev_sdk/3rd/BiSeNet/bin/segment compile " + onnx_file_path + " " + engine_file_path)
    bisenet = BiSeNet_TRT(engine_file_path)
    try:
        for i in range(10):
            # create a new thread to do warm_up
            thread1 = warmUpThread(bisenet)
            thread1.start()
            thread1.join()
    finally:
        # destroy the instance
        bisenet.destroy()
    return bisenet


def process_image(handle=None,input_image=None,args=None, **kwargs):

    """Do inference to analysis input_image and get output
    Attributes:
        handle: algorithm handle returned by init()
        input_image (numpy.ndarray): image to be process, format: (h, w, c), BGR
        args: string in JSON format, format: {
            "mask_output_path": "/path/to/output/mask.png"
        }
    Returns: process result
    """
    args =json.loads(args)
    mask_output_path =args['mask_output_path']
    output = handle.infer( input_image)
    cv2.imwrite(mask_output_path, output)
    return json.dumps({'mask': mask_output_path}, indent=4)

if __name__ == '__main__':
    # Test API
    img = cv2.imread('/project/ev_sdk/data/test.png')
    predictor = init()
    res = process_image(predictor, img, "{\"mask_output_path\":\"./out.jpg\"}")
    print(res)
    