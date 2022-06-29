#include <android/asset_manager_jni.h>
#include <android/bitmap.h>
#include <android/log.h>
#include <jni.h>
#include <string>

#include "layer.h"
#include "net.h"

#if defined(USE_NCNN_SIMPLEOCV)
#include "simpleocv.h"
#else
#endif

static ncnn::Net laneNet;
static ncnn::Net yolov5;

static ncnn::UnlockedPoolAllocator g_blob_pool_allocator;
static ncnn::PoolAllocator g_workspace_pool_allocator;

struct Object {
    float x;
    float y;
    float w;
    float h;
    int label;
    float prob;
};

struct Point {
    float x;
    float y;

    Point (float _x, float _y) : x(_x), y(_y) { }
};

static inline float intersection_area(const Object &a, const Object &b) {
    if (a.x > b.x + b.w || a.x + a.w < b.x || a.y > b.y + b.h || a.y + a.h < b.y) {
        // no intersection
        return 0.f;
    }

    float inter_width = std::min(a.x + a.w, b.x + b.w) - std::max(a.x, b.x);
    float inter_height = std::min(a.y + a.h, b.y + b.h) - std::max(a.y, b.y);

    return inter_width * inter_height;
}

static void qsort_descent_inplace(std::vector<Object> &faceobjects, int left, int right) {
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j) {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j) {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

#pragma omp parallel sections
    {
#pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
#pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<Object> &faceobjects) {
    if (faceobjects.empty())
        return;

    qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object> &faceobjects, std::vector<int> &picked,
                              float nms_threshold) {
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++) {
//        areas[i] = faceobjects[i].rect.area();
        areas[i] = faceobjects[i].w * faceobjects[i].h;
    }

    for (int i = 0; i < n; i++) {
        const Object &a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int) picked.size(); j++) {
            const Object &b = faceobjects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

static inline float sigmoid(float x) {
    return static_cast<float>(1.f / (1.f + exp(-x)));
}

static void generate_proposals(const ncnn::Mat &anchors, int stride, const ncnn::Mat &in_pad,
                               const ncnn::Mat &feat_blob, float prob_threshold,
                               std::vector<Object> &objects) {
    const int num_grid_x = feat_blob.w;
    const int num_grid_y = feat_blob.h;

    const int num_anchors = anchors.w / 2;

    const int num_class = feat_blob.c / num_anchors - 5;

    const int feat_offset = num_class + 5;

    for (int q = 0; q < num_anchors; q++) {
        const float anchor_w = anchors[q * 2];
        const float anchor_h = anchors[q * 2 + 1];

        for (int i = 0; i < num_grid_y; i++) {
            for (int j = 0; j < num_grid_x; j++) {
                // find class index with max class score
                int class_index = 0;
                float class_score = -FLT_MAX;
                for (int k = 0; k < num_class; k++) {
//                    float score = sigmoid(feat_blob.channel(q * feat_offset + 5 + k).row(i)[j]);
                    float score = feat_blob.channel(q * feat_offset + 5 + k).row(i)[j];
//                    __android_log_print(ANDROID_LOG_INFO, "ncnn:", "%s", (std::to_string(score).c_str()));

                    if (score > class_score) {
                        class_index = k;
                        class_score = score;
                    }
                }

                float box_score = feat_blob.channel(q * feat_offset + 4).row(i)[j];

                float confidence = sigmoid(box_score) * sigmoid(class_score);
//                float confidence = box_score * class_score;
                if (confidence >= prob_threshold) {
//                    __android_log_print(ANDROID_LOG_INFO, "ncnn:", "%s", (std::to_string(confidence).c_str()));
                    // yolov5/models/yolo.py Detect forward
                    // y = x[i].sigmoid()
                    // y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
                    // y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh

                    float dx = sigmoid(feat_blob.channel(q * feat_offset + 0).row(i)[j]);
                    float dy = sigmoid(feat_blob.channel(q * feat_offset + 1).row(i)[j]);
                    float dw = sigmoid(feat_blob.channel(q * feat_offset + 2).row(i)[j]);
                    float dh = sigmoid(feat_blob.channel(q * feat_offset + 3).row(i)[j]);

                    float pb_cx = (dx * 2.f - 0.5f + j) * stride;
                    float pb_cy = (dy * 2.f - 0.5f + i) * stride;

                    float pb_w = pow(dw * 2.f, 2) * anchor_w;
                    float pb_h = pow(dh * 2.f, 2) * anchor_h;

                    float x0 = pb_cx - pb_w * 0.5f;
                    float y0 = pb_cy - pb_h * 0.5f;
                    float x1 = pb_cx + pb_w * 0.5f;
                    float y1 = pb_cy + pb_h * 0.5f;

                    Object obj;
                    obj.x = x0;
                    obj.y = y0;
                    obj.w = x1 - x0;
                    obj.h = y1 - y0;
                    obj.label = class_index;
                    obj.prob = confidence;
//                    __android_log_print(ANDROID_LOG_INFO, "ncnn:", "%s", (std::to_string(obj.x).c_str()));


                    objects.push_back(obj);
                }
            }
        }
    }
}

extern "C"
JNIEXPORT jint JNI_OnLoad(JavaVM *vm, void *reserved) {
    __android_log_print(ANDROID_LOG_DEBUG, "Lane-Detection", "JNI_OnLoad");

    ncnn::create_gpu_instance();

    return JNI_VERSION_1_4;
}

extern "C"
JNIEXPORT void JNI_OnUnload(JavaVM *vm, void *reserved) {
    __android_log_print(ANDROID_LOG_DEBUG, "Lane-Detection", "JNI_OnUnload");

    ncnn::destroy_gpu_instance();
}

namespace yolo {

    static jclass objCls = NULL;
    static jmethodID constructortorId;
    static jfieldID xId;
    static jfieldID yId;
    static jfieldID wId;
    static jfieldID hId;
    static jfieldID labelId;
    static jfieldID probId;
    // public native boolean Init(AssetManager mgr);
    extern "C" JNIEXPORT jboolean JNICALL
    Java_com_tongji_lanedetection_YoloV5_init(JNIEnv *env, jobject thiz, jobject assetManager) {
        ncnn::Option opt;
        opt.lightmode = true;
        opt.num_threads = 4;
        opt.blob_allocator = &g_blob_pool_allocator;
        opt.workspace_allocator = &g_workspace_pool_allocator;
        opt.use_packing_layout = true;

        // use vulkan compute
        if (ncnn::get_gpu_count() != 0)
            opt.use_vulkan_compute = true;

        AAssetManager *mgr = AAssetManager_fromJava(env, assetManager);

        yolov5.opt = opt;

        // init param
        {
            int ret = yolov5.load_param(mgr, "yolov5n-lane.param");
            if (ret != 0) {
                __android_log_print(ANDROID_LOG_DEBUG, "YoloV5", "load_param failed");
                return JNI_FALSE;
            }
        }

        // init bin
        {
            int ret = yolov5.load_model(mgr, "yolov5n-lane.bin");
            if (ret != 0) {
                __android_log_print(ANDROID_LOG_DEBUG, "YoloV5", "load_model failed");
                return JNI_FALSE;
            }
        }

        // init jni glue
        jclass localObjCls = env->FindClass("com/tongji/lanedetection/YoloV5$Obj");
        objCls = reinterpret_cast<jclass>(env->NewGlobalRef(localObjCls));

        constructortorId = env->GetMethodID(objCls, "<init>", "(Lcom/tongji/lanedetection/YoloV5;)V");

        xId = env->GetFieldID(objCls, "x", "F");
        yId = env->GetFieldID(objCls, "y", "F");
        wId = env->GetFieldID(objCls, "w", "F");
        hId = env->GetFieldID(objCls, "h", "F");
        labelId = env->GetFieldID(objCls, "label", "I");
        probId = env->GetFieldID(objCls, "prob", "F");

        return JNI_TRUE;
    }

    // public native Obj[] Detect(Bitmap bitmap, boolean use_gpu);
    extern "C" JNIEXPORT jobjectArray JNICALL
    Java_com_tongji_lanedetection_YoloV5_detect(JNIEnv *env, jobject thiz, jobject bitmap,
                                                jboolean use_gpu) {
        if (use_gpu == JNI_TRUE && ncnn::get_gpu_count() == 0) {
            return NULL;
            //return env->NewStringUTF("no vulkan capable gpu");
        }

        AndroidBitmapInfo info;
        AndroidBitmap_getInfo(env, bitmap, &info);
        const int width = info.width;
        const int height = info.height;
        if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888)
            return NULL;

        // ncnn from bitmap
        const int target_size = 320;
        const float prob_threshold = 0.25f;
        const float nms_threshold = 0.45f;

        // letterbox pad to multiple of 32
        int w = width;
        int h = height;
        float scale = 1.f;
        if (w > h) {
            scale = (float) target_size / w;
            w = target_size;
            h = h * scale;
        } else {
            scale = (float) target_size / h;
            h = target_size;
            w = w * scale;
        }

        ncnn::Mat in = ncnn::Mat::from_android_bitmap_resize(env, bitmap, ncnn::Mat::PIXEL_RGB, w, h);

        // pad to target_size rectangle
        // yolov5/utils/datasets.py letterbox
        int wpad = (w + 31) / 32 * 32 - w;
        int hpad = (h + 31) / 32 * 32 - h;
        ncnn::Mat in_pad;
        ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2,
                               ncnn::BORDER_CONSTANT, 114.f);

        // yolov5
        std::vector<Object> objects;
        {

            const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
            in_pad.substract_mean_normalize(0, norm_vals);

            ncnn::Extractor ex = yolov5.create_extractor();

            ex.set_vulkan_compute(use_gpu);

            ex.input("in0", in_pad);

            std::vector<Object> proposals;

            // anchor setting from yolov5/models/yolov5s.yaml

            // stride 8
            {
                ncnn::Mat out;
                ex.extract("out0", out);

                /*ncnn::Mat anchors(6);
                anchors[0] = 10.f;
                anchors[1] = 13.f;
                anchors[2] = 16.f;
                anchors[3] = 30.f;
                anchors[4] = 33.f;
                anchors[5] = 23.f;*/

                ncnn::Mat anchors(6);
                anchors[0] = 2.81832f;
                anchors[1] = 3.24808f;
                anchors[2] = 5.06248f;
                anchors[3] = 3.92776f;
                anchors[4] = 3.59376f;
                anchors[5] = 7.11328f;

                std::vector<Object> objects8;
                generate_proposals(anchors, 8, in_pad, out, prob_threshold, objects8);

                proposals.insert(proposals.end(), objects8.begin(), objects8.end());
            }

            // stride 16
            {
                ncnn::Mat out;
                ex.extract("out1", out);

                /*ncnn::Mat anchors(6);
                anchors[0] = 30.f;
                anchors[1] = 61.f;
                anchors[2] = 62.f;
                anchors[3] = 45.f;
                anchors[4] = 59.f;
                anchors[5] = 119.f;*/

                ncnn::Mat anchors(6);
                anchors[0] = 7.49216f;
                anchors[1] = 6.38672f;
                anchors[2] = 6.78512f;
                anchors[3] = 14.88288f;
                anchors[4] = 12.69536f;
                anchors[5] = 9.7968f;

                std::vector<Object> objects16;
                generate_proposals(anchors, 16, in_pad, out, prob_threshold, objects16);

                proposals.insert(proposals.end(), objects16.begin(), objects16.end());
            }

            // stride 32
            {
                ncnn::Mat out;
                ex.extract("out2", out);

                /*ncnn::Mat anchors(6);
                anchors[0] = 116.f;
                anchors[1] = 90.f;
                anchors[2] = 156.f;
                anchors[3] = 198.f;
                anchors[4] = 373.f;
                anchors[5] = 326.f;*/

                ncnn::Mat anchors(6);
                anchors[0] = 22.5936f;
                anchors[1] = 17.18752f;
                anchors[2] = 43.37504f;
                anchors[3] = 33.78112f;
                anchors[4] = 84.24992f;
                anchors[5] = 68.75008f;

                std::vector<Object> objects32;
                generate_proposals(anchors, 32, in_pad, out, prob_threshold, objects32);

                proposals.insert(proposals.end(), objects32.begin(), objects32.end());
            }

            // sort all proposals by score from highest to lowest
            qsort_descent_inplace(proposals);

            // apply nms with nms_threshold
            std::vector<int> picked;
            nms_sorted_bboxes(proposals, picked, nms_threshold);

            int count = picked.size();

            objects.resize(count);
            for (int i = 0; i < count; i++) {
                objects[i] = proposals[picked[i]];

                // adjust offset to original unpadded
                float x0 = (objects[i].x - (wpad / 2)) / scale;
                float y0 = (objects[i].y - (hpad / 2)) / scale;
                float x1 = (objects[i].x + objects[i].w - (wpad / 2)) / scale;
                float y1 = (objects[i].y + objects[i].h - (hpad / 2)) / scale;

                // clip
                x0 = std::max(std::min(x0, (float) (width - 1)), 0.f);
                y0 = std::max(std::min(y0, (float) (height - 1)), 0.f);
                x1 = std::max(std::min(x1, (float) (width - 1)), 0.f);
                y1 = std::max(std::min(y1, (float) (height - 1)), 0.f);

                objects[i].x = x0;
                objects[i].y = y0;
                objects[i].w = x1 - x0;
                objects[i].h = y1 - y0;
            }
        }

        // objects to Obj[]
        jobjectArray jObjArray = env->NewObjectArray(objects.size(), objCls, NULL);
        for (size_t i = 0; i < objects.size(); i++) {
            jobject jObj = env->NewObject(objCls, constructortorId, thiz);

            env->SetFloatField(jObj, xId, objects[i].x);
            env->SetFloatField(jObj, yId, objects[i].y);
            env->SetFloatField(jObj, wId, objects[i].w);
            env->SetFloatField(jObj, hId, objects[i].h);
            env->SetIntField(jObj, labelId, objects[i].label);
            env->SetFloatField(jObj, probId, objects[i].prob);

            env->SetObjectArrayElement(jObjArray, i, jObj);
        }

        return jObjArray;
    }

}

namespace lane {

    // config
    const int target_w = 800;
    const int target_h = 288;
    const int griding_num = 200;
    const int max_lane_num = 4;
    const int cls_num_per_lane = 18;
    const int row_anchor[] = { 121, 131, 141, 150, 160, 170, 180, 189, 199, 209, 219, 228, 238, 248, 258, 267, 277, 287 };

    static jclass pointCls = NULL;
    static jmethodID constructortorId;
    static jfieldID xId;
    static jfieldID yId;

    // public native boolean Init(AssetManager mgr);
    extern "C" JNIEXPORT jboolean JNICALL
    Java_com_tongji_lanedetection_LaneNet_init(JNIEnv *env, jobject thiz, jobject assetManager) {
        ncnn::Option opt;
        opt.lightmode = true;
        opt.num_threads = 4;
        opt.blob_allocator = &g_blob_pool_allocator;
        opt.workspace_allocator = &g_workspace_pool_allocator;
        opt.use_packing_layout = true;

        // use vulkan compute
        if (ncnn::get_gpu_count() != 0)
            opt.use_vulkan_compute = true;
        AAssetManager *mgr = AAssetManager_fromJava(env, assetManager);
        yolov5.opt = opt;

        // init param
        {
            int ret = laneNet.load_param(mgr, "culane.param");
            if (ret != 0) {
                __android_log_print(ANDROID_LOG_DEBUG, "LaneNet", "load_param failed");
                return JNI_FALSE;
            }
        }

        // init bin
        {
            int ret = laneNet.load_model(mgr, "culane.bin");
            if (ret != 0) {
                __android_log_print(ANDROID_LOG_DEBUG, "LaneNet", "load_model failed");
                return JNI_FALSE;
            }
        }

        // init jni glue
        jclass localPointCls = env->FindClass("com/tongji/lanedetection/LaneNet$Point");
        pointCls = reinterpret_cast<jclass>(env->NewGlobalRef(localPointCls));
        constructortorId = env->GetMethodID(pointCls, "<init>", "(Lcom/tongji/lanedetection/LaneNet;)V");
        xId = env->GetFieldID(pointCls, "x", "F");
        yId = env->GetFieldID(pointCls, "y", "F");
        return JNI_TRUE;
    }

    // public native Obj[] Detect(Bitmap bitmap, boolean use_gpu);
    extern "C" JNIEXPORT jobjectArray JNICALL
    Java_com_tongji_lanedetection_LaneNet_detect(JNIEnv *env, jobject thiz, jobject bitmap,
                                                 jboolean use_gpu) {

        if (use_gpu == JNI_TRUE && ncnn::get_gpu_count() == 0) {
            return NULL;
            //return env->NewStringUTF("no vulkan capable gpu");
        }

        AndroidBitmapInfo info;
        AndroidBitmap_getInfo(env, bitmap, &info);
        const int width = info.width;
        const int height = info.height;
        if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888)
            return NULL;

        // input
        ncnn::Mat in = ncnn::Mat::from_android_bitmap_resize(env, bitmap, ncnn::Mat::PIXEL_RGB, target_w, target_h);

        // [0, 255] -> [0, 1]
        const float n[3] = { 1 / 255.f, 1 / 255.f, 1 / 255.f };
        in.substract_mean_normalize(0, n);

        // normalize
        const float mean[3] = { 0.485f, 0.456f, 0.406f };
        const float norm[3] = { 1 / 0.229f, 1 / 0.224f, 1 / 0.225f };
        in.substract_mean_normalize(mean, norm);

        // interface
        ncnn::Extractor ex = laneNet.create_extractor();
        ex.input("in0", in);

        ncnn::Mat out;
        ex.extract("out0", out);

        // process output
        ncnn::Mat sum_exp = ncnn::Mat(max_lane_num, cls_num_per_lane, sizeof(float));
        ncnn::Mat loc = ncnn::Mat(max_lane_num, cls_num_per_lane, sizeof(float));
        ncnn::Mat argmax = ncnn::Mat(max_lane_num, cls_num_per_lane, sizeof(float));
        ncnn::Mat argmax_val = ncnn::Mat(max_lane_num, cls_num_per_lane, sizeof(float));

        // sum for softmax and argmax
        for (int q = 0; q < griding_num + 1; q++) { // c - 1
            float* data_ptr = out.channel(q);
            float* sum_exp_ptr = sum_exp;
            float* argmax_ptr = argmax;
            float* argmax_val_ptr = argmax_val;

            for (int y = 0; y < cls_num_per_lane; y++) { // h

                for (int x = 0; x < max_lane_num; x++) { // w

                    // sum
                    if (q == 0) sum_exp_ptr[x] = exp(data_ptr[x]);
                    else if (q < griding_num) sum_exp_ptr[x] += exp(data_ptr[x]);

                    // argmax
                    if (q == 0 || data_ptr[x] > argmax_val_ptr[x]) {
                        argmax_val_ptr[x] = data_ptr[x];
                        argmax_ptr[x] = q;
                    }
                }

                argmax_ptr += max_lane_num;
                argmax_val_ptr += max_lane_num;
                data_ptr += max_lane_num;
                sum_exp_ptr += max_lane_num;
            }
        }

        // loc
        for (int q = 0; q < griding_num; q++) { // c - 1
            float* data_ptr = out.channel(q);
            float* sum_exp_ptr = sum_exp;
            float* loc_ptr = loc;
            float* argmax_ptr = argmax;

            for (int y = 0; y < cls_num_per_lane; y++) { // h
                for (int x = 0; x < max_lane_num; x++) { // w

                    // loc = np.sum(prob * idx, axis=0)
                    float prob = exp(data_ptr[x]) / sum_exp_ptr[x];
                    loc_ptr[x] += prob * (q + 1);

                    // loc[processed_output == cfg.griding_num] = 0
                    if (argmax_ptr[x] == griding_num) loc_ptr[x] = 0;
                }

                data_ptr += max_lane_num;
                loc_ptr += max_lane_num;
                sum_exp_ptr += max_lane_num;
                argmax_ptr += max_lane_num;
            }

        }

        // generate points
        std::vector<Point> points;
        const float col_sample_w = float(target_w - 1) / (griding_num - 1);

        float* loc_ptr = loc;
        for (int lane = 0; lane < max_lane_num; lane++) {

            int flag = 0;
            for (int point = 0; point < cls_num_per_lane; point++) {
                if (flag >= 2) break;
                if (loc_ptr[point * max_lane_num + lane] == 0.0f) flag++;
            }

            if (flag >= 2) {
                for (int point = 0; point < cls_num_per_lane; point++) {
                    float data = loc_ptr[point * max_lane_num + lane];
                    if (data > 0.0f) {
                        points.push_back(Point(
                                data * col_sample_w * width / target_w - 1,
                                row_anchor[point] * height / target_h - 1
                        ));
                    }
                }
            }
        }

        // points to Point[]
        jobjectArray jPointArray = env->NewObjectArray(points.size(), pointCls, NULL);
        for (size_t i = 0; i < points.size(); i++) {
            jobject jPoint = env->NewObject(pointCls, constructortorId, thiz);

            env->SetFloatField(jPoint, xId, points[i].x);
            env->SetFloatField(jPoint, yId, points[i].y);

            env->SetObjectArrayElement(jPointArray, i, jPoint);
        }

        return jPointArray;

    }
}