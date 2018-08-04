package com.nbs.nothotdog;

import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.support.annotation.NonNull;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.util.Pair;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.bumptech.glide.Glide;
import com.esafirm.imagepicker.features.ImagePicker;
import com.esafirm.imagepicker.model.Image;
import com.google.android.gms.tasks.Continuation;
import com.google.android.gms.tasks.Task;
import com.google.firebase.FirebaseException;
import com.google.firebase.ml.common.FirebaseMLException;
import com.google.firebase.ml.custom.FirebaseModelDataType;
import com.google.firebase.ml.custom.FirebaseModelInputOutputOptions;
import com.google.firebase.ml.custom.FirebaseModelInputs;
import com.google.firebase.ml.custom.FirebaseModelInterpreter;
import com.google.firebase.ml.custom.FirebaseModelManager;
import com.google.firebase.ml.custom.FirebaseModelOptions;
import com.google.firebase.ml.custom.FirebaseModelOutputs;
import com.google.firebase.ml.custom.model.FirebaseCloudModelSource;
import com.google.firebase.ml.custom.model.FirebaseLocalModelSource;
import com.google.firebase.ml.custom.model.FirebaseModelDownloadConditions;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;

public class MainActivity extends AppCompatActivity {

    private Button btnTakePicture;

    private ImageView imgPreview;

    private TextView tvResult;

    private static final String TAG = "MainActivity";

    public static final String DIR_ASSET = "asset";

    private Bitmap mSelectedImage;

    // Max width (portrait mode)
    private Integer mImageMaxWidth;
    // Max height (portrait mode)
    private Integer mImageMaxHeight;

    private static final String HOSTED_MODEL_NAME = "mobilenet_v1_224_quant";
    private static final String LOCAL_MODEL_ASSET = "mobilenet_v1.0_224_quant.tflite";
    /**
     * Name of the label file stored in Assets.
     */
    private static final String LABEL_PATH = "labels.txt";
    /**
     * Number of results to show in the UI.
     */
    private static final int RESULTS_TO_SHOW = 3;
    /**
     * Dimensions of inputs.
     */
    private static final int DIM_BATCH_SIZE = 1;
    private static final int DIM_PIXEL_SIZE = 3;
    private static final int DIM_IMG_SIZE_X = 224;
    private static final int DIM_IMG_SIZE_Y = 224;
    /**
     * Labels corresponding to the output of the vision model.
     */
    private List<String> mLabelList;

    private final PriorityQueue<Map.Entry<String, Float>> sortedLabels =
            new PriorityQueue<>(
                    RESULTS_TO_SHOW,
                    new Comparator<Map.Entry<String, Float>>() {
                        @Override
                        public int compare(Map.Entry<String, Float> o1, Map.Entry<String, Float>
                                o2) {
                            return (o1.getValue()).compareTo(o2.getValue());
                        }
                    });
    /* Preallocated buffers for storing image data. */
    private final int[] intValues = new int[DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y];

    private FirebaseModelInterpreter mInterpreter;

    private FirebaseModelInputOutputOptions mOptions;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        btnTakePicture = findViewById(R.id.btn_camera);

        imgPreview = findViewById(R.id.img_preview);

        tvResult = findViewById(R.id.tv_result);

        mLabelList = loadLabelList(this);

        int[] inputDims = {DIM_BATCH_SIZE, DIM_IMG_SIZE_X, DIM_IMG_SIZE_Y, DIM_PIXEL_SIZE};
        int[] outputDims = {DIM_BATCH_SIZE, mLabelList.size()};

        try{
            mOptions = new FirebaseModelInputOutputOptions.Builder()
                    .setInputFormat(0, FirebaseModelDataType.BYTE, inputDims)
                    .setOutputFormat(0, FirebaseModelDataType.BYTE, outputDims)
                    .build();

            FirebaseModelDownloadConditions downloadConditions = new FirebaseModelDownloadConditions.Builder()
                    .requireWifi()
                    .build();

            FirebaseLocalModelSource localModelSource = new FirebaseLocalModelSource.Builder("asset")
                    .setAssetFilePath(LOCAL_MODEL_ASSET)
                    .build();

            FirebaseCloudModelSource cloudModelSource = new FirebaseCloudModelSource.Builder(HOSTED_MODEL_NAME)
                    .enableModelUpdates(true)
                    .setInitialDownloadConditions(downloadConditions)
                    .setUpdatesDownloadConditions(downloadConditions)
                    .build();

            FirebaseModelManager firebaseModelManager = FirebaseModelManager.getInstance();
            firebaseModelManager.registerCloudModelSource(cloudModelSource);
            firebaseModelManager.registerLocalModelSource(localModelSource);

            FirebaseModelOptions modelOptions = new FirebaseModelOptions.Builder()
                    .setLocalModelName(DIR_ASSET)
                    .setCloudModelName(HOSTED_MODEL_NAME)
                    .build();

            mInterpreter = FirebaseModelInterpreter.getInstance(modelOptions);

        }catch (FirebaseException e){
            e.printStackTrace();
            showToast("Error while setting up the model");
        }

        btnTakePicture.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                tvResult.setVisibility(View.GONE);
                ImagePicker.cameraOnly().start(MainActivity.this);
            }
        });
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        if (ImagePicker.shouldHandle(requestCode, resultCode, data)){
            Image image = ImagePicker.getFirstImageOrNull(data);

            Glide.with(this).load(image.getPath()).into(imgPreview);

            mSelectedImage = BitmapFactory.decodeFile(image.getPath());

            runModelInference();
        }
        super.onActivityResult(requestCode, resultCode, data);
    }

    private void runModelInference() {
        if (mInterpreter == null){
            showToast("Image classifier has not been initialized; Skipped");
            return;
        }

        ByteBuffer imgData = convertBitmapToByteBuffer(mSelectedImage, mSelectedImage.getWidth(), mSelectedImage.getHeight());

        try{
            FirebaseModelInputs inputs = new FirebaseModelInputs.Builder().add(imgData).build();

            mInterpreter.run(inputs, mOptions)
                    .continueWith(new Continuation<FirebaseModelOutputs, List<String>>() {
                        @Override
                        public List<String> then(@NonNull Task<FirebaseModelOutputs> task) {
                            byte[][] labelProbArray = task.getResult().getOutput(0);

                            List<String> topLabels = getTopLabels(labelProbArray);

                            boolean isHotdog = false;

                            for (String s: topLabels){
                                if (s.contains("hotdog") ||
                                        s.contains("hot dog")){
                                    isHotdog = true;
                                    break;
                                }
                            }

                            int backgroundColor = 0;

                            String result;

                            if (isHotdog) {
                                backgroundColor = ContextCompat.getColor(MainActivity.this, R.color.colorGreen);
                                result = "This is hotdog!";
                            }else{
                                backgroundColor = ContextCompat.getColor(MainActivity.this, R.color.colorPrimary);
                                result = "This is not hotdog";
                            }

                            tvResult.setVisibility(View.VISIBLE);
                            tvResult.setText(result);
                            tvResult.setBackgroundColor(backgroundColor);

                            return topLabels;
                        }
                    });
        }catch (FirebaseMLException e){
            e.printStackTrace();
            showToast("Error running label inference");
        }
    }

    /**
     * Gets the top labels in the results.
     */
    private synchronized List<String> getTopLabels(byte[][] labelProbArray) {
        for (int i = 0; i < mLabelList.size(); ++i) {
            sortedLabels.add(
                    new AbstractMap.SimpleEntry<>(mLabelList.get(i), (labelProbArray[0][i] &
                            0xff) / 255.0f));
            if (sortedLabels.size() > RESULTS_TO_SHOW) {
                sortedLabels.poll();
            }
        }
        List<String> result = new ArrayList<>();
        final int size = sortedLabels.size();
        for (int i = 0; i < size; ++i) {
            Map.Entry<String, Float> label = sortedLabels.poll();
            result.add(label.getKey() + ":" + label.getValue());
        }
        Log.d(TAG, "labels: " + result.toString());
        return result;
    }

    /**
     * Reads label list from Assets.
     */
    private List<String> loadLabelList(Activity activity) {
        List<String> labelList = new ArrayList<>();
        try (BufferedReader reader =
                     new BufferedReader(new InputStreamReader(activity.getAssets().open
                             (LABEL_PATH)))) {
            String line;
            while ((line = reader.readLine()) != null) {
                labelList.add(line);
            }
        } catch (IOException e) {
            Log.e(TAG, "Failed to read label list.", e);
        }
        return labelList;
    }

    /**
     * Writes Image data into a {@code ByteBuffer}.
     */
    private synchronized ByteBuffer convertBitmapToByteBuffer(
            Bitmap bitmap, int width, int height) {
        ByteBuffer imgData =
                ByteBuffer.allocateDirect(
                        DIM_BATCH_SIZE * DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y * DIM_PIXEL_SIZE);
        imgData.order(ByteOrder.nativeOrder());
        Bitmap scaledBitmap = Bitmap.createScaledBitmap(bitmap, DIM_IMG_SIZE_X, DIM_IMG_SIZE_Y,
                true);
        imgData.rewind();
        scaledBitmap.getPixels(intValues, 0, scaledBitmap.getWidth(), 0, 0,
                scaledBitmap.getWidth(), scaledBitmap.getHeight());
        // Convert the image to int points.
        int pixel = 0;
        for (int i = 0; i < DIM_IMG_SIZE_X; ++i) {
            for (int j = 0; j < DIM_IMG_SIZE_Y; ++j) {
                final int val = intValues[pixel++];
                imgData.put((byte) ((val >> 16) & 0xFF));
                imgData.put((byte) ((val >> 8) & 0xFF));
                imgData.put((byte) (val & 0xFF));
            }
        }
        return imgData;
    }

    private void showToast(String message) {
        Toast.makeText(getApplicationContext(), message, Toast.LENGTH_SHORT).show();
    }

    // Utility functions for loading and resizing images from app asset folder.
    public static Bitmap getBitmapFromAsset(Context context, String filePath) {
        AssetManager assetManager = context.getAssets();

        InputStream is;
        Bitmap bitmap = null;
        try {
            is = assetManager.open(filePath);
            bitmap = BitmapFactory.decodeStream(is);
        } catch (IOException e) {
            e.printStackTrace();
        }

        return bitmap;
    }

    // Returns max image width, always for portrait mode. Caller needs to swap width / height for
    // landscape mode.
    private Integer getImageMaxWidth() {
        if (mImageMaxWidth == null) {
            // Calculate the max width in portrait mode. This is done lazily since we need to
            // wait for a UI layout pass to get the right values. So delay it to first time image
            // rendering time.
            mImageMaxWidth = imgPreview.getWidth();
        }

        return mImageMaxWidth;
    }

    // Returns max image height, always for portrait mode. Caller needs to swap width / height for
    // landscape mode.
    private Integer getImageMaxHeight() {
        if (mImageMaxHeight == null) {
            // Calculate the max width in portrait mode. This is done lazily since we need to
            // wait for a UI layout pass to get the right values. So delay it to first time image
            // rendering time.
            mImageMaxHeight =
                    imgPreview.getHeight();
        }

        return mImageMaxHeight;
    }

    // Gets the targeted width / height.
    private Pair<Integer, Integer> getTargetedWidthHeight() {
        int targetWidth;
        int targetHeight;
        int maxWidthForPortraitMode = getImageMaxWidth();
        int maxHeightForPortraitMode = getImageMaxHeight();
        targetWidth = maxWidthForPortraitMode;
        targetHeight = maxHeightForPortraitMode;
        return new Pair<>(targetWidth, targetHeight);
    }
}
