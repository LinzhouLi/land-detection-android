package com.tongji.lanedetection;

import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.view.PreviewView;

import android.graphics.Color;
import android.os.Bundle;
import android.view.View;
import android.view.Window;
import android.view.WindowInsets;
import android.view.WindowInsetsController;
import android.widget.ImageView;
import android.widget.Toast;


public class MainActivity extends AppCompatActivity {

    private final YoloV5 yolov5 = new YoloV5();
    private final LaneNet laneNet = new LaneNet();
    private final CameraProcess cameraProcess = new CameraProcess();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        supportRequestWindowFeature(Window.FEATURE_NO_TITLE); // 去掉标题栏
        setContentView(R.layout.activity_main);

        // 打开app的时候隐藏顶部状态栏
        WindowInsetsController ic = getWindow().getInsetsController();
        if (ic != null) {
            ic.hide(WindowInsets.Type.statusBars());
            ic.hide(WindowInsets.Type.navigationBars());
        }
        getWindow().setStatusBarColor(Color.TRANSPARENT);

        // 申请权限
        if (!cameraProcess.allPermissionsGranted(this)) {
            cameraProcess.requestPermissions(this);
        }

        // 模型初始化
        boolean yolo_init = yolov5.init(getAssets());
        if (yolo_init)
            Toast.makeText(this, "yolov5n初始化成功", Toast.LENGTH_LONG).show();
        else
            Toast.makeText(this, "yolov5n初始化失败", Toast.LENGTH_LONG).show();

        boolean laneNet_init = laneNet.init(getAssets());
        if (laneNet_init)
            Toast.makeText(this, "LaneNet初始化成功", Toast.LENGTH_LONG).show();
        else
            Toast.makeText(this, "LaneNet初始化失败", Toast.LENGTH_LONG).show();

        // 开始检测
        PreviewView cameraPreview = findViewById(R.id.camera_preview);
        ImageView canvas = findViewById(R.id.box_label_canvas);

        ImageAnalyzer imageAnalyzer = new ImageAnalyzer(
                MainActivity.this,
                cameraPreview,
                canvas,
                yolov5,
                laneNet
        );

        cameraProcess.startCamera(MainActivity.this, imageAnalyzer, cameraPreview);
    }
}