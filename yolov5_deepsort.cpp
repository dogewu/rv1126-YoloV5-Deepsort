#include <unistd.h>
#include <stdio.h>
#include <string>
#include <string.h>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <unordered_set>

#include "common.h"
#include "detect.h"
#include "deepsort.h"
#include "mytime.h"
#include "videoio.h"


using namespace std;

bool add_head = false;
string PROJECT_DIR = "/userdata/yolov5_Deepsort_rknn";


string YOLO_MODEL_PATH = "./model/best.rknn";
string SORT_MODEL_PATH = "./model/osnet.rknn";

string VIDEO_PATH = "./data/test5.avi";
string VIDEO_SAVEPATH = "./data/test_results5.avi";

/*
string YOLO_MODEL_PATH = PROJECT_DIR + "/model/best_nofocus_relu.rknn";
string SORT_MODEL_PATH = PROJECT_DIR + "/model/osnet_x0_25_market.rknn";

string VIDEO_PATH = PROJECT_DIR + "/data/DJI_0001_S_cut.mp4";
string VIDEO_SAVEPATH = PROJECT_DIR + "/data/results.mp4";
*/


// 各任务进行状态序号
video_property video_probs; // 视频属性类
int idxInputImage = 0;  // image index of input video
int idxOutputImage = 0; // image index of output video
int idxTrackImage = 0;	  // 目标追踪下一帧要处理的对象
bool bReading = true;   // flag of input
bool bDetecting = true; // Detect是否完成
bool bTracking = true;  // Track是否完成
double start_time; // Video Detection开始时间
double end_time;   // Video Detection结束时间
int MAX_FRAMES_IN_POOL = 125;
bool video_read = true;

// 多线程控制相关
mutex mtxreadvideo;
mutex mtximagePool; //mutex of viedo cache
condition_variable cv_videoRead;
vector<cv::Mat> imagePool(MAX_FRAMES_IN_POOL);        // video cache
mutex mtxQueueInput;        	  // mutex of input queue
condition_variable cv_videoResize;
queue<input_image> queueInput;    // input queue 
mutex mtxQueueDetOut;
condition_variable cv_detect;
queue<imageout_idx> queueDetOut;  // output queue
mutex mtxQueueOutput;
condition_variable cv_deepsort;
queue<imageout_idx> queueOutput;  // output queue 目标追踪输出队列

int readIndex = 0;
int resizeIndex = 0;

unordered_set<int> low_id; //下行行人id集合
unordered_set<int> up_id; //上行行人id集合
unordered_set<int> total_region1; //在基准线上方的行人id存入该集合
unordered_set<int> total_region2; //在基准线下方的行人id存入该集合
int baseline = 500; //基准线

void videoReadandResize(const char *video_name, int cpuid);
void videoRead(const char *video_name, int cpuid);
void videoResize(int cpuid);
void videoWrite(const char* save_path,int cpuid);

int main() {

    // class Yolo detect1(YOLO_MODEL_PATH.c_str(), 4, RKNN_NPU_CORE_0, 1, 3);
    // class Yolo detect2(YOLO_MODEL_PATH.c_str(), 5, RKNN_NPU_CORE_1, 1, 3);
    // class DeepSort track(SORT_MODEL_PATH, 1, 512, 6, RKNN_NPU_CORE_2);
    class Yolo detect1(YOLO_MODEL_PATH.c_str(), 4, 1, 3);
    // class Yolo detect2(YOLO_MODEL_PATH.c_str(), 5, 1, 3);
    class DeepSort track(SORT_MODEL_PATH, 1, 512, 6);

    const int thread_num = 4;
    // std::array<thread, thread_num> threads;
    // videoRead(VIDEO_PATH.c_str(), 7);
    // videoResize(7);
    // used CPU: 0, 4, 5, 6, 7
    
    // threads = {   
    //               thread(videoRead, VIDEO_PATH.c_str(), 8),
    //               thread(&Yolo::detect_process, &detect1),  // 类成员函数特殊写法
    //               thread(&Yolo::detect_process, &detect2),
    //               thread(&DeepSort::track_process, &track),
    //               thread(videoResize, 7),
    //               thread(videoWrite, VIDEO_SAVEPATH.c_str(), 0),
    //           };
    thread threads[thread_num];
    threads[0] = thread(videoReadandResize, VIDEO_PATH.c_str(), 7);
    threads[1] = thread(&Yolo::detect_process, &detect1);
    // threads[2] = thread(&Yolo::detect_process, &detect2);
    threads[2] = thread(&DeepSort::track_process, &track);
    // threads[4] = thread(videoResize, 7);
    threads[3] = thread(videoWrite, VIDEO_SAVEPATH.c_str(), 0);
    for (int i = 0; i < thread_num; i++) threads[i].join();
    double cost_time = (end_time-start_time) / 1000 / video_probs.Frame_cnt;
    std::cout << "Video detection cost time(s):" << (end_time-start_time) / 1000 << std::endl;
    std::cout << "Video detection mean cost time(s):" << cost_time << std::endl;
    // printf("Video detection mean cost time(ms): %f\n", (end_time-start_time) / video_probs.Frame_cnt);
    return 0;
}
