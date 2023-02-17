import os
import urllib
import traceback
import time
import sys
import numpy as np
import cv2
from rknn.api import RKNN
import onnx
import onnxruntime
ONNX_MODEL = "model_data/onnx/20210204_unified_transformer_exp/onnx_model/ctc_48-64.onnx"
RKNN_MODEL = "model_data/onnx/20210204_unified_transformer_exp/onnx_model/ctc_48-64.rknn"
DATASET = "dataset.txt"

def convert_rknn():
    # Create RKNN object
    rknn = RKNN(verbose=True)
    # pre-process config
    print('--> Config model')
    rknn.config(mean_values=None, std_values=None,target_platform="rk3588",quantized_algorithm="normal",)
    print('done')

    # Load ONNX model
    print('--> Loading model')

    ret = rknn.load_onnx(model=ONNX_MODEL)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    QUANTIZE_ON = False

    ret = rknn.build(do_quantization=QUANTIZE_ON, dataset=DATASET)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # # Accuracy analysis
    # print('--> Accuracy analysis')
    # ret = rknn.accuracy_analysis(inputs=['./lane1.jpg'])
    # if ret != 0:
    #     print('Accuracy analysis failed!')
    #     exit(ret)
    # print('done')

    # Export RKNN model
    print('--> Export rknn model')

    ret = rknn.export_rknn(RKNN_MODEL)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    # Init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime()
    # ret = rknn.init_runtime('rk3566')
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')

    # # Set inputs
    # img = cv2.imread("lane1.jpg")
    # # img, ratio, (dw, dh) = letterbox(img, new_shape=(IMG_S
    # # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = cv2.resize(img, (512, 288))
    # # Inference
    # print('--> Running model')
    # outputs = rknn.inference(inputs=[img])
    # print(outputs[1].shape)
    # print(outputs[1])
    # # np.save('./onnx_yolov5_0.npy', outputs[0])
    # # np.save('./onnx_yolov5_1.npy', outputs[1])
    # print('done')
    # rknn.release()


if __name__ == '__main__':


    convert_rknn()
    # run_onnx()


    # # Init runtime environment
    # print('--> Init runtime environment')
    # ret = rknn.init_runtime()
    # # ret = rknn.init_runtime('rk3566')
    # if ret != 0:
    #     print('Init runtime environment failed!')
    #     exit(ret)
    # print('done')
    #
    #
    # # Set inputs
    # img = cv2.imread("lane1.jpg")
    # # img, ratio, (dw, dh) = letterbox(img, new
    # img = cv2.resize(img, (512, 288))
    #
    # # Inference
    # print('--> Running model')
    # outputs = rknn.inference(inputs=[img])
    #
    # np.save('./onnx_yolov5_0.npy', outputs[0])
    # np.save('./onnx_yolov5_1.npy', outputs[1])
    # print('done')