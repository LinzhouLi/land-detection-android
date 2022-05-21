package com.tongji.lanedetection;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.RectF;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.util.Log;
import android.widget.ImageView;

import androidx.annotation.NonNull;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.view.PreviewView;

import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;

import io.reactivex.rxjava3.android.schedulers.AndroidSchedulers;
import io.reactivex.rxjava3.core.Observable;
import io.reactivex.rxjava3.core.ObservableEmitter;
import io.reactivex.rxjava3.schedulers.Schedulers;

public class ImageAnalyzer implements ImageAnalysis.Analyzer {

    public static class Result{

        public Result(long costTime1, long costTime2, long costTime3, long costTimeTotal, Bitmap bitmap) {
            this.costTime1 = costTime1;
            this.costTime2 = costTime2;
            this.costTime3 = costTime3;
            this.costTimeTotal = costTimeTotal;
            this.bitmap = bitmap;
        }
        long costTime1, costTime2, costTime3, costTimeTotal;
        Bitmap bitmap;
    }

    private final ImageView boxLabelCanvas;
    private final PreviewView previewView;
    private final ImageProcess imageProcess;
    private final YoloV5 yolov5Detector;
    private final LaneNet laneNetDetector;
    private Matrix fullScreenTransform = null;

    public ImageAnalyzer (
            Context context,
            PreviewView previewView,
            ImageView boxLabelCanvas,
            YoloV5 yolov5Detector,
            LaneNet laneNetDetector
    ) {

        this.previewView = previewView;
        this.boxLabelCanvas = boxLabelCanvas;
        this.yolov5Detector = yolov5Detector;
        this.laneNetDetector = laneNetDetector;
        this.imageProcess = new ImageProcess();

    }

    private Bitmap convertImageProxyToBitmap(ImageProxy image) {
        ByteBuffer yBuffer = image.getPlanes()[0].getBuffer();
        ByteBuffer vuBuffer = image.getPlanes()[2].getBuffer();

        int ySize = yBuffer.remaining();
        int vuSize = vuBuffer.remaining();

        byte[] nv21 = new byte[ySize + vuSize];

        yBuffer.get(nv21, 0, ySize);
        vuBuffer.get(nv21, ySize, vuSize);

        YuvImage yuvImage = new YuvImage(nv21, ImageFormat.NV21, image.getWidth(), image.getHeight(), null);
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        yuvImage.compressToJpeg(new Rect(0, 0, yuvImage.getWidth(), yuvImage.getHeight()), 50, out);
        byte[] imageBytes = out.toByteArray();
        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.length);
    }

    @Override
    public void analyze(@NonNull ImageProxy image) {

        int previewHeight = previewView.getHeight();
        int previewWidth = previewView.getWidth();

        int imageHeight = image.getHeight();
        int imageWidth = image.getWidth();

        // 图片适应屏幕fill_start格式的bitmap
//        if (fullScreenTransform == null) {
//            double scale = Math.max(
//                    previewHeight / (double) (imageHeight),
//                    previewWidth / (double) (imageWidth)
//            );
//            fullScreenTransform = imageProcess.getTransformationMatrix(
//                    imageWidth, imageHeight,
//                    (int) (scale * imageWidth), (int) (scale * imageHeight),
//                    0, false
//            );
//        }

        Observable.create( (ObservableEmitter<Result> emitter) -> {

            fullScreenTransform = imageProcess.getTransformationMatrix(
                    imageWidth, imageHeight,
                    previewWidth, previewHeight,
                    0, false
            );

            long time1 = System.currentTimeMillis();

            // bitmap
            Bitmap imageBitmap = convertImageProxyToBitmap(image);
            Bitmap fullImageBitmap = Bitmap.createBitmap(imageBitmap, 0, 0, imageWidth, imageHeight, fullScreenTransform, false);

            long time2 = System.currentTimeMillis();

//            Log.i("imageBitmap", imageBitmap.getWidth() + "  " + imageBitmap.getHeight());
//            Log.i("fullImageBitmap", fullImageBitmap.getWidth() + "  " + fullImageBitmap.getHeight());

            // 调用yolov5预测接口
            YoloV5.Obj[] objects = yolov5Detector.detect(fullImageBitmap, true);

            long time3 = System.currentTimeMillis();

            // 调用laneNet预测接口
            LaneNet.Point[] points = laneNetDetector.detect(fullImageBitmap, true);

            long time4 = System.currentTimeMillis();

            // 画出预测结果
            Bitmap resultBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Bitmap.Config.ARGB_8888);
            Canvas canvas = new Canvas(resultBitmap);
            drawObjects(objects, canvas);
            drawPoints(points, canvas);

            long time5 = System.currentTimeMillis();

            image.close();
            emitter.onNext(new Result(
                    time2 - time1,
                    time3 - time2,
                    time4 - time3,
                    time5 - time1,
                    resultBitmap
            ));

        }).subscribeOn(Schedulers.io())
                .observeOn(AndroidSchedulers.mainThread())
                .subscribe( (Result result) -> {
                    Log.i("Bitmap", result.bitmap.getWidth() + "  "+ result.bitmap.getHeight());
                    boxLabelCanvas.setImageBitmap(result.bitmap);
                    Log.i("Analyzer", "bitmap: " + Long.toString(result.costTime1) + "ms");
                    Log.i("Analyzer", "yolo: " + Long.toString(result.costTime2) + "ms");
                    Log.i("Analyzer", "laneNet: " + Long.toString(result.costTime3) + "ms");
                    Log.i("Analyzer", "total: " + Long.toString(result.costTimeTotal) + "ms");
                });

    }

    private void drawPoints(LaneNet.Point[] points, Canvas canvas) {

        // 边框画笔
        Paint boxPaint = new Paint();
        boxPaint.setColor(Color.WHITE);
        boxPaint.setStyle(Paint.Style.FILL);
        final int width = 11;

        for (LaneNet.Point point : points) {
            RectF location = new RectF();
            location.left = point.x - width / 2;
            location.top = point.y - width / 2;
            location.right = point.x + width / 2 + 1;
            location.bottom = point.y + width / 2 + 1;

//            Log.i("Point", point.x + "  " +  point.y);

            canvas.drawRect(location, boxPaint);
        }

    }

    private void drawObjects(YoloV5.Obj[] objects, Canvas canvas) {

        // 边框画笔
        Paint boxPaint = new Paint();
        boxPaint.setStrokeWidth(5);
        boxPaint.setStyle(Paint.Style.STROKE);

        // 字体画笔
        Paint textPain = new Paint();
        textPain.setTextSize(50);
        textPain.setStyle(Paint.Style.FILL);

        for (YoloV5.Obj res : objects) {
            int label = res.label;
//            Log.i("Object", YoloV5.getLabel(label));
            textPain.setColor(YoloV5.getColor(label));
            boxPaint.setColor(YoloV5.getColor(label));

            RectF location = new RectF();
            location.left = res.x;
            location.top = res.y;
            location.right = res.x + res.w;
            location.bottom = res.y + res.h;
//            Log.i("Object", location.left + "  " + location.right + "  " + location.top + "  " + location.bottom);

            canvas.drawRect(location, boxPaint);
            canvas.drawText(YoloV5.getLabel(label) + ":" + String.format("%.2f", res.prob), location.left, location.top, textPain);
        }

    }

}