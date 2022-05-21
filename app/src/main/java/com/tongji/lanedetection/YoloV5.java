package com.tongji.lanedetection;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Color;

public class YoloV5 {

    static {
        System.loadLibrary("lanedetection");
    }

    static private String[] labels = {
            "traffic light", "traffic sign", "car", "person", "bus",
            "truck", "rider", "bike", "motor", "train"
    };
    static private int[] colors = {
            Color.parseColor("#6dd802"), Color.parseColor("#4b6bd0"),
            Color.parseColor("#029dd8"), Color.parseColor("#4102d8"),
            Color.parseColor("#c102d8"), Color.parseColor("#d80279"),
            Color.parseColor("#d80220"), Color.parseColor("#d85602"),
            Color.parseColor("#cbd802"), Color.parseColor("#217107")
    };

    public class Obj
    {
        public float x;
        public float y;
        public float w;
        public float h;
        public int label;
        public float prob;
    }

    static public String getLabel(int i) {
        return labels[i];
    }

    static public int getColor(int i) {
        return colors[i];
    }

    public native boolean init(AssetManager assetManager);
    public native Obj[] detect(Bitmap bitmap, boolean use_gpu);

}
