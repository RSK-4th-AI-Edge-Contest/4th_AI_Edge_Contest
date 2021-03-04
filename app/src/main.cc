#include <assert.h>
#include <algorithm>
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <atomic> //DB
#include <sys/stat.h>
#include <unistd.h> //DB
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <queue>
#include <mutex> 
#include <string>
#include <vector>
#include <thread> 
#include <opencv2/opencv.hpp>
#include <pthread.h>

// Header files for DNNDK APIs
#include <dnndk/dnndk.h>
#include <dputils.h>
using namespace std;
using namespace std::chrono;
using namespace cv;


const string baseImagePath = "./seg_test_images/";
vector<string> kinds, images; //DB

#define KERNEL_CONV "hardnet"
#define CONV_INPUT_NODE "conv2d_1_convolution"
#define CONV_OUTPUT_NODE "up_sampling2d_5_ResizeNearestNeighbor"

uint8_t colorB[] = {128, 232, 70, 156, 153, 153, 30,  0,   35, 152,
                    180, 60,  0,  142, 70,  100, 100, 230, 32};
uint8_t colorG[] = {64, 35, 70, 102, 153, 153, 170, 220, 142, 251, 
                    130, 20, 0, 0, 0, 60, 80, 0, 11};
uint8_t colorR[] = {128, 244, 70,  102, 190, 153, 250, 220, 107, 152,
                    70,  220, 255, 0,   0,   0,   0,   0,   119};

int stop_count=0;

Mat mask; 



void normalize_image(const Mat& image, int8_t* data, float scale, float* mean) {
  for(int i = 0; i < 3; ++i) {
    for(int j = 0; j < image.rows; ++j) {
      for(int k = 0; k < image.cols; ++k) {
	      data[j*image.cols*3+k*3+i] = (float(image.at<Vec3b>(j,k)[i])/127.5 - 1.0)* scale;
      }
     }
   }
}
inline void set_input_image(DPUTask *task, const string& input_node, const cv::Mat& image, float* mean)
{
  DPUTensor* dpu_in = dpuGetInputTensor(task, CONV_INPUT_NODE);
  float scale       = dpuGetTensorScale(dpu_in);
  int width         = dpuGetTensorWidth(dpu_in);
  int height        = dpuGetTensorHeight(dpu_in);
  int size          = dpuGetTensorSize(dpu_in);
  int8_t* data      = dpuGetTensorAddress(dpu_in);

  normalize_image(image, data, scale, mean);
}
void ListImages(std::string const &path, std::vector<std::string> &images) {
  images.clear();
  struct dirent *entry;

  struct stat s;
  lstat(path.c_str(), &s);
  if (!S_ISDIR(s.st_mode)) {
    fprintf(stderr, "Error: %s is not a valid directory!\n", path.c_str());
    exit(1);
  }

  DIR *dir = opendir(path.c_str());
  if (dir == nullptr) {
    fprintf(stderr, "Error: Open %s path failed.\n", path.c_str());
    exit(1);
  }

  while ((entry = readdir(dir)) != nullptr) {
    if (entry->d_type == DT_REG || entry->d_type == DT_UNKNOWN) {
      std::string name = entry->d_name;
      std::string ext = name.substr(name.find_last_of(".") + 1);
      if ((ext == "JPEG") || (ext == "jpeg") || (ext == "JPG") || (ext == "jpg") ||
          (ext == "bmp") ||  (ext == "BMP") || (ext == "PNG") || (ext == "png")) {
        images.push_back(name);
      }
    }
  }

  closedir(dir);
}
class Compare {
    public:
    bool operator()(const pair<int, Mat> &n1, const pair<int, Mat> &n2) const {
        return n1.first > n2.first;
    }
};

bool is_reading = true;
bool is_running_1 = true;
bool is_pp = true;
bool is_write = true;

queue<pair<string, Mat>> read_queue;
queue<pair<string, Mat>> pre_queue;
queue<pair<string, Mat>> pp_queue;
queue<pair<string, Mat>> write_queue;

mutex mtx_read_queue;
mutex mtx_pre_queue; 
mutex mtx_pp_queue;
mutex mtx_write_queue; 

int read_index = 0; 
int display_index = 0; 
int cut_h=100;

void runSegmentation(DPUTask *task, bool &is_running) {
    DPUTensor *conv_in_tensor = dpuGetInputTensor(task, CONV_INPUT_NODE);
    int inHeight = dpuGetTensorHeight(conv_in_tensor);
    int inWidth = dpuGetTensorWidth(conv_in_tensor);

    DPUTensor *conv_out_tensor = dpuGetOutputTensor(task, CONV_OUTPUT_NODE);
    int outHeight = dpuGetTensorHeight(conv_out_tensor);
    int outWidth = dpuGetTensorWidth(conv_out_tensor);
    int8_t *outTensorAddr = dpuGetTensorAddress(conv_out_tensor);
    string path;
    float mean[3] = {0.0f, 0.0f, 0.0f};
    std::chrono::system_clock::time_point  start, end;
    cout<<"DPU:Start"<<endl;

    int count=0;
    while (is_running) {
        count++;

        int index;
        Mat img;
        string path;
        mtx_pre_queue.lock();
        if (pre_queue.empty()) {
            mtx_pre_queue.unlock();
            usleep(200);
        } 
        else {
            start = std::chrono::system_clock::now();
            path = pre_queue.front().first;
            img = pre_queue.front().second;
            pre_queue.pop();
            mtx_pre_queue.unlock();

            set_input_image(task, CONV_INPUT_NODE, img, mean);

            dpuRunTask(task);

            int d_dpu_time = dpuGetTaskProfile(task) / 1000;


            Mat segMat(outHeight, outWidth, CV_8UC3);
            
            for (int row = 0; row < outHeight; row++) {
                for (int col = 0; col < outWidth; col++) {
                    int i = row * outWidth * 5 + col * 5;
                    auto max_ind = max_element(outTensorAddr + i, outTensorAddr + i + 5);
                    int posit = distance(outTensorAddr + i, max_ind);
                    segMat.at<Vec3b>(row, col) = Vec3b(colorB[posit], colorG[posit], colorR[posit]);
                }
            }

            mtx_pp_queue.lock();
            pp_queue.push(make_pair(path, segMat));
            mtx_pp_queue.unlock();
            end = std::chrono::system_clock::now();
            double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count(); 
            cout<<"DPU:"<<elapsed/1000.0<<endl;
            if(count==stop_count){
                is_running_1=0;
                break;
            }

        }
    }
}


void Pre(bool &is_reading) {
    cout<<"Pre:Start"<<endl;
    string path;
    Mat img;
    int count=0;
    std::chrono::system_clock::time_point  start, end;

    while(is_reading){
        start = std::chrono::system_clock::now();
        count++;
        if (read_queue.empty()) {
            is_reading=false;
            break;
        }
        mtx_read_queue.lock();
        path= read_queue.front().first;
        img= read_queue.front().second;
        read_queue.pop();
        mtx_read_queue.unlock();

        Mat cut_img(img,Rect(0,0,1936,1216-cut_h));
        img=cut_img;
        resize(img, img, Size(), 832.0/img.cols ,512.0/img.rows,INTER_NEAREST);
        cvtColor(img, img, CV_BGR2RGB);


        mtx_pre_queue.lock();
        pre_queue.push(make_pair(path, img));
        mtx_pre_queue.unlock();
        end = std::chrono::system_clock::now();
        double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count(); 
        cout << "pre count:"<<count<<" "<<path<<" "<<elapsed/1000.0<<endl;
    
    }
}


void pp(bool &is_pp) {

    int count=0;
    string path;
    Mat img;
    std::chrono::system_clock::time_point  start, end;
    while (is_pp) {
        mtx_pp_queue.lock();
        if (pp_queue.empty()) {
            mtx_pp_queue.unlock();
    	    usleep(20);
            if(count==649)
            {
                is_running_1=0;
                is_pp=0;
                break;
            }
            if(count==stop_count){
                is_running_1=0;
                is_pp=0;
                break;
            }

        }

        else {

            start = std::chrono::system_clock::now();
            count++;
            path = pp_queue.front().first;
            img = pp_queue.front().second;
            pp_queue.pop();
            mtx_pp_queue.unlock();

            resize(img, img, Size(), 1936.0/img.cols ,(1216.0-cut_h)/img.rows,INTER_NEAREST);

            Mat add_img(cut_h, 1936, CV_8UC3);
            add_img=Mat::zeros(cut_h, 1936, CV_8UC3);
            vconcat(img, add_img, img);

            //Mask
            Mat new_image = Mat(img.size(), CV_8UC3);;
            new_image.setTo(cv::Scalar(0,0,0) );
            img.copyTo(new_image,mask);

            path=path.substr(0,path.rfind('.'));

            mtx_write_queue.lock();
            write_queue.push(make_pair(path, new_image));

            mtx_write_queue.unlock();

            end = std::chrono::system_clock::now();
            double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count(); 
            cout << "pp count:"<<count<<" "<<path<<" "<<elapsed/1000.0<<endl;
        }
    }
}

int main(int argc, char **argv) {
    std::chrono::system_clock::time_point  start, end;
    start = std::chrono::system_clock::now();
    //Mask
    mask = imread("own_mask/mask_9.png", 1);

    bitwise_not(mask,mask);

    float total_time=0;
    array<thread, 2> threads;

    DPUKernel *kernel_conv;
    DPUTask *task_conv_1, *task_conv_2;

    dpuOpen();
    kernel_conv = dpuLoadKernel(KERNEL_CONV);
    task_conv_1 = dpuCreateTask(kernel_conv, 0);
    end = std::chrono::system_clock::now();
    total_time+=std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();

    ListImages(baseImagePath, images);
    int count=0;
    for(unsigned int ind = 0  ;ind < images.size();ind++)
    {
        count++;
        //50枚読み込み
        string path=images.at(ind);
        Mat img = imread(baseImagePath + path,1);
        cout<<count<<" "<<baseImagePath + path<<endl;
        read_queue.push(make_pair(path, img));
        if(count%50==0 || count==649){
            cout<<"Read Done"<<endl;
            start = std::chrono::system_clock::now();
            if(count == 649){
                stop_count=49;
            }
            else
            {
                stop_count=50;
            }
            // Run tasks for Segmentation
            is_reading = true;
            is_running_1 = true;
            is_pp = true;
            is_write = true;
            cout<<"start thread"<<endl;
            thread thread_pre1(Pre, ref(is_reading));
    	    thread thread_pre2(Pre, ref(is_reading));
            thread_pre1.join();
            thread_pre2.join();
            thread thread_pp1(pp, ref(is_pp));          
            runSegmentation(task_conv_1, is_running_1);
	        thread_pp1.join();
            end = std::chrono::system_clock::now();
            total_time+=std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();

            for(int i=0 ; i < stop_count;i++ ){
                path = write_queue.front().first;
                img = write_queue.front().second;
                imwrite("result/"+path+".png", img);
                cout<<"write "<<i<<endl;
                write_queue.pop();

            }
        }
    }
       cout<<"Time:"<<total_time/1000.0<<" "<<total_time/1000.0/649.0<<endl;
    dpuDestroyTask(task_conv_1);
    dpuDestroyKernel(kernel_conv);
    dpuClose();
    return 0;
}
