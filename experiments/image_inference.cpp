#include <iostream>
#include <onnxruntime/onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <algorithm>
#include <cmath>

using namespace std;
using namespace cv;
using namespace Ort;

static inline float sigmoid(float x){ return 1.f / (1.f + std::exp(-x)); }
static inline bool in01(float v){ return v >= 0.f && v <= 1.f; }

struct Detection { Rect box; int cls; float score; };

struct Letterbox {
    Mat img; float r; int dw; int dh;
};

static Letterbox letterbox(const Mat& im, int new_shape=640, int stride=32){
    float r = std::min((float)new_shape / im.rows, (float)new_shape / im.cols);
    int new_unpad_w = (int)std::round(im.cols * r);
    int new_unpad_h = (int)std::round(im.rows * r);
    int dw = new_shape - new_unpad_w;
    int dh = new_shape - new_unpad_h;
    // опционально выравниваем паддинг по stride (как Ultralytics)
    dw %= stride; dh %= stride;
    dw /= 2; dh /= 2;

    Mat resized; 
    if (im.cols != new_unpad_w || im.rows != new_unpad_h)
        resize(im, resized, Size(new_unpad_w, new_unpad_h), 0, 0, INTER_LINEAR);
    else
        resized = im;

    Mat out;
    copyMakeBorder(resized, out, dh, dh, dw, dw, BORDER_CONSTANT, Scalar(114,114,114));
    return {out, r, dw, dh};
}

static Rect from_xywh(float xc,float yc,float w,float h, bool normalized, int W, int H){
    if (normalized){ xc *= W; yc *= H; w *= W; h *= H; }
    int x = (int)std::round(xc - w*0.5f);
    int y = (int)std::round(yc - h*0.5f);
    return Rect(x,y,(int)std::round(w),(int)std::round(h));
}

static Rect deletterbox(const Rect& r, float scale, int dw, int dh, const Size& orig){
    Rect out;
    out.x = (int)std::round((r.x - dw) / scale);
    out.y = (int)std::round((r.y - dh) / scale);
    out.width  = (int)std::round(r.width  / scale);
    out.height = (int)std::round(r.height / scale);
    return out & Rect(0,0, orig.width, orig.height);
}

static vector<Detection> parse_yolov8(const float* out, const vector<int64_t>& shp, int W, int H, float thr=0.25f){
    vector<Detection> dets;

    // [1,N,6] end2end: x1,y1,x2,y2,score,cls
    if (shp.size()==3 && shp[2]==6){
        int N=(int)shp[1];
        for (int i=0;i<N;i++){
            const float* p=out+i*6;
            float x1=p[0],y1=p[1],x2=p[2],y2=p[3], sc=p[4]; int cls=(int)p[5];
            if (sc<thr) continue;
            dets.push_back({Rect((int)std::round(x1),(int)std::round(y1),
                                 (int)std::round(x2-x1),(int)std::round(y2-y1)), cls, sc});
        }
        return dets;
    }

    auto handle_attrs = [&](int N, int A, bool ch_first){
        int num_classes = A - 4;
        auto get = [&](int i,int a){ return ch_first? out[a*N + i] : out[i*A + a]; };
        for (int i=0;i<N;i++){
            float xc=get(i,0), yc=get(i,1), w=get(i,2), h=get(i,3);
            // классы: бывают уже в [0..1], а бывают логитами → сигмоида по необходимости
            int best=0; float bestv=-1e9f;
            for (int c=0;c<num_classes;c++){
                float v=get(i,4+c);
                float p = in01(v) ? v : sigmoid(v);
                if (p>bestv){ bestv=p; best=c; }
            }
            if (bestv < thr) continue;
            bool normalized = in01(xc) && in01(yc) && w <= 2.5f && h <= 2.5f;
            Rect box = from_xywh(xc,yc,w,h, normalized, W, H);
            dets.push_back({box, best, bestv});
        }
    };

    // [1,84,N] или [1,N,84] (A может быть !=84 для пользовательских классов)
    if (shp.size()==3){
        int A1=(int)shp[1], A2=(int)shp[2];
        if (A1>=6 && A2>=6){
            if (A1 < A2){ // [1, A, N]
                handle_attrs((int)shp[2], (int)shp[1], /*ch_first=*/true);
                return dets;
            } else {      // [1, N, A]
                handle_attrs((int)shp[1], (int)shp[2], /*ch_first=*/false);
                return dets;
            }
        }
    }

    return dets; // если формат неожиданный — вернём пусто
}

int main(){
    // ORT API v23
    Env env(ORT_LOGGING_LEVEL_WARNING, "YOLOv8");
    SessionOptions so;
    Session session(env, "./Resources/best.onnx", so);

    // исходное изображение
    Mat orig = imread("./Resources/mouth.jpg");
    if (orig.empty()){ std::cerr<<"Image not found\n"; return -1; }

    // letterbox -> 640x640
    auto lb = letterbox(orig, 640, 32); // как у Ultralytics
    Mat input = lb.img;

    // blob: RGB, NCHW, 0..1
    Mat blob = dnn::blobFromImage(input, 1.0/255.0, Size(), Scalar(), /*swapRB=*/true, /*crop=*/false);
    vector<float> input_data(blob.begin<float>(), blob.end<float>());
    vector<int64_t> in_shape = {1, 3, input.rows, input.cols};

    auto meminfo = MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    auto tensor   = Value::CreateTensor<float>(meminfo, input_data.data(), input_data.size(),
                                               in_shape.data(), in_shape.size());

    const char* in_names[]  = {"images"};
    const char* out_names[] = {"output0"};
    auto out_vec = session.Run(RunOptions{nullptr}, in_names, &tensor, 1, out_names, 1);

    // читаем выход
    auto shp = out_vec[0].GetTensorTypeAndShapeInfo().GetShape();
    const float* out_ptr = out_vec[0].GetTensorData<float>();
    size_t total=1; for (auto s:shp) total*= (size_t)s;
    vector<float> out_buf(out_ptr, out_ptr+total);

    // парсинг
    auto dets_lb = parse_yolov8(out_buf.data(), shp, input.cols, input.rows, 0.25f);

    // перенос боксов из letterbox в координаты оригинала
    vector<Detection> dets;
    dets.reserve(dets_lb.size());
    for (auto& d : dets_lb){
        Rect on_orig = deletterbox(d.box, lb.r, lb.dw, lb.dh, orig.size());
        dets.push_back({on_orig, d.cls, d.score});
    }

    // рисуем аккуратно (контур, без заливки)
    for (const auto& d : dets){
        if (d.score < 0.25f) continue;
        rectangle(orig, d.box, Scalar(255,0,255), 2);
        putText(orig, to_string(d.cls) + " " + to_string(int(d.score*100)) + "%",
                Point(d.box.x, max(0, d.box.y-5)), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255,0,255), 2);
    }

    imshow("result", orig);
    waitKey(0);
    return 0;
}
