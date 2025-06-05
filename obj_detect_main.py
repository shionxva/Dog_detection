from Detector import *
import os

folder = r'C:\Users\ADMIN\Desktop\C files\realtime_obj'

def main():
    videoPath = 0 #path or 0 for camera
    configPath = r"C:\Users\ADMIN\Desktop\C files\realtime_obj\model_data\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
    modelPath = r"C:\Users\ADMIN\Desktop\C files\realtime_obj\model_data\frozen_inference_graph.pb"
    classesPath = r"C:\Users\ADMIN\Desktop\C files\realtime_obj\model_data\coco.names"

    detector = Detector(videoPath, configPath, modelPath, classesPath)
    detector.onVideo()

if __name__ =='__main__':
    main()