# RV1126_Yolov5_DeepSORT_rknn

****
# 改动: 

## 本仓库在原仓库的基础上:
1.去除了RV1126不支持的多核NPU等内容
2.部分API适配到rknpu1
3.将数据拷贝从零拷贝改为API拷贝
4.解决了在1126上使用egien库时，部分矩阵转置操作会报数据未对齐的问题
****

RV1126_Yolov5_DeepSORT是基于瑞芯微Rockchip Neural Network(RKNN)开发的目标跟踪部署仓库，Deepsort算法在rv1126上进行了测试。

下面是在1126上读取视频进行跟踪的结果

<div align="center">
  <img src="https://github.com/dogewu/rv1126_YolovV5_Deepsort//test_results4.gif" width="45%" />
  <br/>
  <font size=5>deepsort</font>
  <br/>
</div>

## 文档内容

- [文件目录结构描述](#文件目录结构描述)
- [安装及使用](#安装及使用)
- [性能测试](#性能测试)
- [参考仓库](#参考仓库)

## 文件目录结构描述

├── Readme.md                   // help
├── data						// 数据
├── model						// 模型
├── build
├── CMakeLists.txt			    // 编译Yolov5_DeepSORT
├── include						// 通用头文件
├── src
├── 3rdparty                    
│   ├── linrknn_api				// rknn   动态链接库
│   ├── rga		                // rga    动态链接库
│   ├── opencv		            // opencv 动态链接库(自行编译并在CmakeLists.txt中设置相应路径)
├── yolov5           			
│   └── include
│       └── decode.h            // 解码
│       └── detect.h            // 推理
│       └── videoio.h           // 视频IO
│   └── src
│       └── decode.cpp    
│       └── ...
├── deepsort
│   └── include
│       └── deepsort.h     		// class DeepSort
│       └── featuretensor.h     // Reid推理
│       └── ...
│   └── src
│       └── deepsort.cpp
│       └── ...
│   └── CMakeLists.txt			// 编译deepsort子模块


## 安装及使用

+ RKNN-Toolkit

  这个项目需要使用RKNN-Toolkit1(Rk1126)。可以在瑞芯微的官方仓库中获得相应的工具，可以先运行瑞芯微仓库中的Demo来测试。

+ opencv的编译安装

  我是直接使用交叉编译工具编译的opecv4，注意在编译opencv时一定要注意确保ffmpeg选项的状态为√，否则opencv无法读取视频。
  编译代码时需要根据自己的Opencv路径修改CMakeList文件中的内容，项目有两个CmakeList，都要进行修改

+ DeepSort选用的模型是TorchReID中的osnet_x0_25_market ，输入尺寸是128×256，输出为512
  模型可以在下面链接进行获取
  ```
  Torchreid
  https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO
  ```
  如果需要把对应的pytorch模型文件转为onnx，请关注该链接https://github.com/KaiyangZhou/deep-person-reid。

## 性能测试
综合来看在rv1126上进行检测加跟踪，处理一帧视频的平均时间为1s左右，无法做到实时处理。
当然，目前使用的模型都是没有经过量化和预编译处理的，根据之前使用yoloV5的经验，经过量化和预编译后，模型的处理速度会得到很大的提升。

## 参考仓库
1. https://github.com/ultralytics/yolov5
2. https://github.com/airockchip/rknn_model_zoo
3. https://github.com/airockchip/librga
4. https://github.com/RichardoMrMu/yolov5-deepsort-tensorrt
5. https://github.com/KaiyangZhou/deep-person-reid
6. https://github.com/Zhou-sx/yolov5_Deepsort_rknn

最后感谢Zhou-sx大佬的代码，我只是进行了一些改动
