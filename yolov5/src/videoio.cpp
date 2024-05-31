#include "videoio.h"
#include "resize.h"
#include "mytime.h"
#include "common.h"
#include <mutex>
#include <condition_variable>
#include <unistd.h>
#include <unordered_set>

using namespace std;

extern mutex mtxreadvideo;
extern video_property video_probs;
extern mutex mtximagePool;
extern condition_variable cv_videoRead;
extern vector<cv::Mat> imagePool;
extern mutex mtxQueueInput;
extern condition_variable cv_videoResize;
extern queue<input_image> queueInput;  // input queue client
extern mutex mtxQueueDetOut;
extern condition_variable cv_detect;
extern queue<imageout_idx> queueDetOut;        // Det output queue
extern mutex mtxQueueOutput;
extern condition_variable cv_deepsort;
extern queue<imageout_idx> queueOutput;  // 目标追踪输出队列

extern bool add_head;
extern bool bReading;      // flag of input
extern bool bDetecting;    // 目标检测进程状态
extern bool bTracking;
extern int idxInputImage;  // image index of input video
extern int MAX_FRAMES_IN_POOL;
extern bool video_read;

extern int readIndex;
extern int resizeIndex;

extern unordered_set<int> low_id; //下行行人id集合
extern unordered_set<int> up_id; //上行行人id集合
extern unordered_set<int> total_region1; //在基准线上方的行人id存入该集合
extern unordered_set<int> total_region2; //在基准线下方的行人id存入该集合
extern int baseline; //基准线

void videoReadandResize(const char *video_name, int cpuid)
{
	cpu_set_t mask;
	CPU_ZERO(&mask);
	CPU_SET(cpuid, &mask);

	if (pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask) < 0)
		cerr << "set thread affinity failed" << endl;
	printf("Bind videoReadandResizeClient process to CPU %d\n", cpuid);

	cv::VideoCapture video;
	if (!video.open(video_name)) {
		cout << "Fail to open " << video_name << endl;
		return;
	}

	video_probs.Frame_cnt = video.get(CV_CAP_PROP_FRAME_COUNT);
    video_probs.Fps = video.get(CV_CAP_PROP_FPS);
    video_probs.Video_width = video.get(CV_CAP_PROP_FRAME_WIDTH);
    video_probs.Video_height = video.get(CV_CAP_PROP_FRAME_HEIGHT);
    video_probs.Video_fourcc = video.get(CV_CAP_PROP_FOURCC);

	cout << "width" << video_probs.Video_width <<endl;
    cout << "height" << video_probs.Video_height <<endl;
    cout << "frame count" << video_probs.Frame_cnt <<endl;
    cout << "frame rate" << video_probs.Fps << endl;	

	PreResize pre_do(NET_INPUTHEIGHT, NET_INPUTWIDTH, NET_INPUTCHANNEL);
	bReading = true;//读写状态标记
	cout << "total length of video: " << video_probs.Frame_cnt << "\n";
	while(1)
	{
		cv::Mat img_src;
		video >> img_src;
		if (img_src.empty() || idxInputImage >= video_probs.Frame_cnt ) {
			// cout << "read and resize video stream failed! Maybe to the end!" << endl;
			cout << "read video end!" << endl;
			video.release();
			break;
		}
		if(idxInputImage % MAX_FRAMES_IN_POOL == 0 && idxInputImage != 0)
		{
			unique_lock<mutex> lock(mtxreadvideo);
			cout << "************image read full. wait." << endl;
			cv_videoRead.wait(lock, []{return queueInput.empty() && queueDetOut.empty();});
			cout << "************detect queue is empty, restart image read." << endl;
			lock.unlock();
			// double start = what_time_is_it_now();
			// cout << "queue is full. stop 180s" << endl;
			// sleep(180);
			// double end = what_time_is_it_now();
			// cout << "restart, has sleep " << (end - start) / 1000 << "s" << endl;
		}
			
		
		// imagePool[readIndex] = img_src.clone();
		

		// cv::Mat img_src = imagePool[resizeIndex];
		cv::Mat img = img_src.clone();

		cv::Mat img_pad;
		resize(img, img_pad, cv::Size(640,640), 0, 0, 1);

		mtxQueueInput.lock();
		queueInput.push(input_image(idxInputImage, img_src, img_pad));
		mtxQueueInput.unlock();
		readIndex++;
		idxInputImage++;
		resizeIndex++;
	}
	cout << "VideoRead is over." << endl;
	bReading = false;
	cout << "VideoResize is over." << endl;
	cout << "Resize Video Total Length: " << idxInputImage << "\n";
}

/*---------------------------------------------------------
	读视频 缓存在imagePool
	video_name: 视频路径
	cpuid:		绑定到某核
----------------------------------------------------------*/
void videoRead(const char *video_name, int cpuid) 
{
	// int initialization_finished = 1;
	cpu_set_t mask;

	CPU_ZERO(&mask);
	CPU_SET(cpuid, &mask);

	if (pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask) < 0)
		cerr << "set thread affinity failed" << endl;

	printf("Bind videoReadClient process to CPU %d\n", cpuid); 

	unique_lock<mutex> lock(mtxreadvideo);
	cv_videoRead.wait(lock, []{return video_read;});
	cv::VideoCapture video;
	if (!video.open(video_name)) {
		cout << "Fail to open " << video_name << endl;
		return;
	}

	video_probs.Frame_cnt = video.get(CV_CAP_PROP_FRAME_COUNT);
    video_probs.Fps = video.get(CV_CAP_PROP_FPS);
    video_probs.Video_width = video.get(CV_CAP_PROP_FRAME_WIDTH);
    video_probs.Video_height = video.get(CV_CAP_PROP_FRAME_HEIGHT);
    video_probs.Video_fourcc = video.get(CV_CAP_PROP_FOURCC);

	cout << "width" << video_probs.Video_width <<endl;
    cout << "height" << video_probs.Video_height <<endl;
    cout << "frame count" << video_probs.Frame_cnt <<endl;
    cout << "frame rate" << video_probs.Fps << endl;	
	video_read = false;
	lock.unlock();
	// cv_videoResize.notify_one();

	bReading = true;//读写状态标记
	// cv::Mat img_src;
	int i = 0;
	while (1) 
	{  
		cv::Mat img_src;
		// 如果读不到图片 或者 bReading 不在读取状态则跳出
		video >> img_src;
		// if (!video.read(img_src)) {
		if (img_src.empty() || i >= video_probs.Frame_cnt ) {
			cout << "read video stream failed! Maybe to the end!" << endl;
			video.release();
			break;
		}
		std::unique_lock<mutex> lock2(mtximagePool);
		cv_videoRead.wait(lock2, []{return imagePool.size() <= MAX_FRAMES_IN_POOL;});
		cout << "read nums: " << i << endl;
		// string s = "./image/" + to_string(i) + ".jpg";
		// cv::imwrite(s, img_src);
		// imagePool.emplace_back(img_src.clone());
		imagePool[readIndex] = img_src.clone();
		// imagePool[i] = img_src.clone();
		// std::cout << i << std::endl;
		i++;
		readIndex++;
		lock2.unlock();
		cv_videoResize.notify_one();
	}
	// for(int i = 0;i < imagePool.size() ;i++){
	// 	string s = "./imagePool/" + to_string(i) + ".jpg";
	// 	cv::imwrite(s, imagePool[i]);
	// }
	// cv::imwrite("origin2.jpg", imagePool[50]);
	cout << "VideoRead is over." << endl;
	// cout << "Video Total Length: " << imagePool.size() << "\n";
	cout << "Video Total Length: " << i << "\n";
}

/*---------------------------------------------------------
	调整视频尺寸
	cpuid:		绑定到某核
----------------------------------------------------------*/
void videoResize(int cpuid){
	// int initialization_finished = 1;
	// rga_buffer_t src;
	// rga_buffer_t dst;
	// im_rect src_rect;
	// im_rect dst_rect;
	// memset(&src_rect, 0, sizeof(src_rect));
	// memset(&dst_rect, 0, sizeof(dst_rect));
	// memset(&src, 0, sizeof(src));
	// memset(&dst, 0, sizeof(dst));

	cpu_set_t mask;

	CPU_ZERO(&mask);
	CPU_SET(cpuid, &mask);

	if (pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask) < 0)
		cerr << "set thread affinity failed" << endl;

	printf("Bind videoTransClient process to CPU %d\n", cpuid);
	
	unique_lock<mutex> lock(mtxreadvideo);
	cv_videoResize.wait(lock, []{return !video_read;});

	PreResize pre_do(NET_INPUTHEIGHT, NET_INPUTWIDTH, NET_INPUTCHANNEL);
	bReading = true;//读写状态标记
	cout << "total length of video: " << video_probs.Frame_cnt << "\n";
	lock.unlock();
	while (1) 
	{  
		// 如果读不到图片 或者 bReading 不在读取状态则跳出
		if (!bReading || idxInputImage >= video_probs.Frame_cnt) {
			break;
		}
		std::unique_lock<mutex> lock2(mtximagePool);
		// cv_videoResize.wait(lock2, []{return !imagePool.empty();});
		cv_videoResize.wait(lock2, []{return resizeIndex < readIndex;});
		cout << "resize num: " << idxInputImage <<endl;
		
		// cv::Mat img_src = imagePool[idxInputImage];
		cv::Mat img_src = imagePool[resizeIndex];
		cv::Mat img = img_src.clone();
		// std::cout << "image height" << img_src.rows << "image weight" << img_src.cols << std::endl;
		// cv::imwrite("origin.jpg", img_src);
		// cv::cvtColor(img_src, img_src, cv::COLOR_BGR2RGB);
		// cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
		// cv::imwrite("origin_src.jpg", img_src);
		// cv::imwrite("origin.jpg", img_src);
		cv::Mat img_pad;
		resize(img, img_pad, cv::Size(640,640), 0, 0, 1);
		// string s = "./resizeimg/" + to_string(idxInputImage) + ".jpg";
		// cv::imwrite(s, img_pad);
		// pre_do.resize(img, img_pad);
		// std::cout << "image height" << img_pad.rows << "image weight" << img_pad.cols << std::endl;
		// cv::imwrite("resize.jpg", img_pad);
		// if (add_head){
		// 	// adaptive head
		// }
		// else{
		// 	// rga resize
			
		// 	void *resize_buf = malloc(NET_INPUTHEIGHT * NET_INPUTWIDTH * NET_INPUTCHANNEL);
		// 	src = wrapbuffer_virtualaddr((void *)img.data, img.cols, img.rows, RK_FORMAT_RGB888);
		// 	dst = wrapbuffer_virtualaddr((void *)resize_buf, NET_INPUTWIDTH, NET_INPUTHEIGHT, RK_FORMAT_RGB888);
			
		// }

		mtxQueueInput.lock();
		queueInput.push(input_image(idxInputImage, img_src, img_pad));
		mtxQueueInput.unlock();
		idxInputImage++;
		resizeIndex++;
		
		lock2.unlock();
		cv_videoRead.notify_one();
	}
	bReading = false;
	cout << "VideoResize is over." << endl;
	// cout << "Resize Video Total Length: " << queueInput.size() << "\n";
	cout << "Resize Video Total Length: " << idxInputImage << "\n";
}

 /*
	预处理的缩放比例
	在不丢失原图比例的同时，尽可能的伸缩；同时为了保证检测效果，只允许缩放，不允许放大。
	fx = 1 沿x轴缩放
	fy = 1 沿y轴缩放
*/
void get_max_scale(int input_width, int input_height, int net_width, int net_height, double &fx, double &fy)
{
    double img_wh_ratio = (double)input_width / (double)input_height;
	double input_wh_ratio = (double)net_width / (double)net_height;
	if (img_wh_ratio >= input_wh_ratio){
		// 缩放相同倍数 w 先到达边界
		fx = (double)net_width / input_width;
		fy = (double)net_width / input_width;
	}
	else{
		fx = (double)net_height / input_height;
		fy = (double)net_height / input_height;
	}
	return;
}

// 写视频
void videoWrite(const char* save_path,int cpuid) 
{
	cpu_set_t mask;
	CPU_ZERO(&mask);
	CPU_SET(cpuid, &mask);

	if (pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask) < 0)
		cerr << "set thread affinity failed" << endl;

	printf("Bind videoWrite process to CPU %d\n", cpuid); 

	cv::VideoWriter vid_writer;
    while(1)
    {
		// cout << "checkpoint! " << queueInput.size() << "\n";
    	if(queueInput.size() > 0)
    	{
            // cout << video_probs.Video_width << " " << video_probs.Video_height << endl;
            vid_writer  = cv::VideoWriter(save_path, video_probs.Video_fourcc, video_probs.Fps, 
										  cv::Size(video_probs.Video_width, video_probs.Video_height));
            break;
    	}
    }

	while (1) 
	{  
		// if (queueOutput.size()) cout << "checkpoint in VideoWriter: " << queueOutput.size() << "\n";
		// queueOutput 就尝试写
		if (queueOutput.size() > 0) {
			mtxQueueOutput.lock();
 			imageout_idx res_pair = queueOutput.front();
			queueOutput.pop();
			mtxQueueOutput.unlock();
			draw_image(res_pair.img, res_pair.dets);
			vid_writer.write(res_pair.img); // Save-video
		}
		// 最后一帧检测/追踪结束 bWriting置为false 此时如果queueOutput仍存在元素 继续写
		else if(!bTracking){
			vid_writer.release();
			break;
		}
	}
	cout << "VideoWrite is over." << endl;
}

/*---------------------------------------------------------
	绘制预测框
----------------------------------------------------------*/
string labels[2]={"person", "vehicle"};
cv::Scalar colorArray[2]={
	cv::Scalar(139,0,0,255),
	cv::Scalar(139,0,139,255),
};

int draw_image(cv::Mat &img,detect_result_group_t detect_result_group)
{
	char text[256];
	char first_low[256];
	char first_up[256];
    for (auto det_result : detect_result_group.results)
    {
        // sprintf(text, "%s %.1f%%", det_result.name, det_result.confidence * 100);
		int id = (int)det_result.trackID;
		sprintf(text, "ID:%d", (int)det_result.trackID);
        int x1 = det_result.x1;
        int y1 = det_result.y1 / IMG_WIDTH * IMG_HEIGHT;
        int x2 = det_result.x2;
        int y2 = det_result.y2 / IMG_WIDTH * IMG_HEIGHT;
		int class_id = det_result.classID;

		// int cx = int((x1 + x2) / 2);
		// int cy = y2;

		// if(cy > baseline)
		// {
		// 	if(total_region2.find(id) != total_region2.end() && total_region1.find(id) == total_region1.end())
		// 	{
		// 		up_id.insert(id);
		// 		line(img, cv::Point(0, baseline), cv::Point(1920, baseline), cv::Scalar(255, 0, 0, 255), 3);
		// 	}
		// 	total_region1.insert(id);
		// }

		// if(cy < baseline)
		// {
		// 	if(total_region1.find(id) != total_region1.end() && total_region2.find(id) == total_region2.end())
		// 	{
		// 		low_id.insert(id);
		// 		line(img, cv::Point(0, baseline), cv::Point(1920, baseline), cv::Scalar(255, 0, 0, 255), 3);
		// 	}
		// 	total_region2.insert(id);
		// }
		
		// line(img, cv::Point(0, baseline), cv::Point(1920, baseline), cv::Scalar(0, 255, 0, 255), 3);
        rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), colorArray[class_id%10], 3);
        putText(img, text, cv::Point(x1, y1 - 12), 1, 2, cv::Scalar(0, 255, 0, 255));
    }
	// sprintf(first_low, "LOW:%d", low_id.size());
	// sprintf(first_low, "LOW:%d", 0);
	// sprintf(first_up, "UP:%d", up_id.size());
	// putText(img, first_low, cv::Point(0, 100), 1, 4, cv::Scalar(0, 255, 0, 255));
	// putText(img, first_up, cv::Point(1520, 100), 1, 4, cv::Scalar(0, 255, 0, 255));
	// imwrite("./display.jpg", img);
	return 0;
}