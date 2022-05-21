package com.tongji.lanedetection;

import android.content.res.AssetManager;
import android.graphics.Bitmap;

public class LaneNet {

    static {
        System.loadLibrary("lanedetection");
    }

    public  class Point {
        public float x;
        public float y;
    }

    public native boolean init(AssetManager assetManager);
    public native Point[] detect(Bitmap bitmap, boolean use_gpu);

}
