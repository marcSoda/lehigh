// Copyright (c) 2020 Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

package org.pytorch.demo.objectdetection;

import android.Manifest;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.graphics.Rect;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Base64;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.RadioButton;
import android.widget.Toast;

import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.android.volley.AuthFailureError;
import com.android.volley.Request;
import com.android.volley.RequestQueue;
import com.android.volley.Response;
import com.android.volley.VolleyError;
import com.android.volley.toolbox.StringRequest;
import com.android.volley.toolbox.Volley;

import org.apache.commons.io.IOUtils;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;
import org.pytorch.IValue;
import org.pytorch.LiteModuleLoader;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class MainActivity extends AppCompatActivity implements Runnable {
    private long startTime;
    private String URL = "http://192.168.4.191:1701/predict"; // Change this address to the address of your cloud server
    private int mImageIndex = 0;
    private String[] mTestImages = {"test1.png", "test2.jpg", "test3.png"};
    private ImageView mImageView;
    private ResultView mResultView;
    private Button mButtonDetect;
    private RadioButton mRadioMobile, mRadioCloud;
    private ProgressBar mProgressBar;
    private Bitmap mBitmap = null;
    private Module mModule = null;
    private float mImgScaleX, mImgScaleY, mIvScaleX, mIvScaleY, mStartX, mStartY;
    private boolean useCloud = false;

    public static String assetFilePath(Context context, String assetName) throws IOException {
        File file = new File(context.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }

        try (InputStream is = context.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.READ_EXTERNAL_STORAGE}, 1);
        }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, 1);
        }

        setContentView(R.layout.activity_main);

        try {
            mBitmap = BitmapFactory.decodeStream(getAssets().open(mTestImages[mImageIndex]));
        } catch (IOException e) {
            Log.e("Object Detection", "Error reading assets", e);
            finish();
        }

        mImageView = findViewById(R.id.imageView);
        mImageView.setImageBitmap(mBitmap);
        mResultView = findViewById(R.id.resultView);
        mResultView.setVisibility(View.INVISIBLE);

        final Button buttonTest = findViewById(R.id.testButton);
        buttonTest.setText(("Test Image 1/3"));
        buttonTest.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                mResultView.setVisibility(View.INVISIBLE);
                mImageIndex = (mImageIndex + 1) % mTestImages.length;
                buttonTest.setText(String.format("Test Image %d/%d", mImageIndex + 1, mTestImages.length));

                try {
                    mBitmap = BitmapFactory.decodeStream(getAssets().open(mTestImages[mImageIndex]));
                    mImageView.setImageBitmap(mBitmap);
                } catch (IOException e) {
                    Log.e("Object Detection", "Error reading assets", e);
                    finish();
                }
            }
        });


        final Button buttonSelect = findViewById(R.id.selectButton);
        buttonSelect.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                mResultView.setVisibility(View.INVISIBLE);

                final CharSequence[] options = { "Choose from Photos", "Take Picture", "Cancel" };
                AlertDialog.Builder builder = new AlertDialog.Builder(MainActivity.this);
                builder.setTitle("New Test Image");

                builder.setItems(options, new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialog, int item) {
                        if (options[item].equals("Take Picture")) {
                            Intent takePicture = new Intent(android.provider.MediaStore.ACTION_IMAGE_CAPTURE);
                            startActivityForResult(takePicture, 0);
                        }
                        else if (options[item].equals("Choose from Photos")) {
                            Intent pickPhoto = new Intent(Intent.ACTION_PICK, android.provider.MediaStore.Images.Media.INTERNAL_CONTENT_URI);
                            startActivityForResult(pickPhoto , 1);
                        }
                        else if (options[item].equals("Cancel")) {
                            dialog.dismiss();
                        }
                    }
                });
                builder.show();
            }
        });

        mButtonDetect = findViewById(R.id.detectButton);
        mRadioMobile = (RadioButton) findViewById(R.id.radioButtonMobile);
        mRadioCloud = (RadioButton) findViewById(R.id.radioButtonCloud);
        mProgressBar = (ProgressBar) findViewById(R.id.progressBar);
        mButtonDetect.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                startTime = System.currentTimeMillis();
                mButtonDetect.setEnabled(false);
                mProgressBar.setVisibility(ProgressBar.VISIBLE);
                if (mRadioMobile.isChecked()) {
                    mButtonDetect.setText(getString(R.string.inference_mobile));
                }
                else {
                    mButtonDetect.setText(getString(R.string.inference_cloud));
                }

                mImgScaleX = (float)mBitmap.getWidth() / PrePostProcessor.mInputWidth;
                mImgScaleY = (float)mBitmap.getHeight() / PrePostProcessor.mInputHeight;

                mIvScaleX = (mBitmap.getWidth() > mBitmap.getHeight() ? (float)mImageView.getWidth() / mBitmap.getWidth() : (float)mImageView.getHeight() / mBitmap.getHeight());
                mIvScaleY  = (mBitmap.getHeight() > mBitmap.getWidth() ? (float)mImageView.getHeight() / mBitmap.getHeight() : (float)mImageView.getWidth() / mBitmap.getWidth());

                mStartX = (mImageView.getWidth() - mIvScaleX * mBitmap.getWidth())/2;
                mStartY = (mImageView.getHeight() -  mIvScaleY * mBitmap.getHeight())/2;

                Thread thread = new Thread(MainActivity.this);
                thread.start();
            }
        });

        try {
            mModule = LiteModuleLoader.load(MainActivity.assetFilePath(getApplicationContext(), "yolov5l.torchscript.ptl"));
            BufferedReader br = new BufferedReader(new InputStreamReader(getAssets().open("classes.txt")));
            String line;
            List<String> classes = new ArrayList<>();
            while ((line = br.readLine()) != null) {
                classes.add(line);
            }
            PrePostProcessor.mClasses = new String[classes.size()];
            classes.toArray(PrePostProcessor.mClasses);
        } catch (IOException e) {
            Log.e("Object Detection", "Error reading assets", e);
            finish();
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode != RESULT_CANCELED) {
            switch (requestCode) {
                case 0:
                    if (resultCode == RESULT_OK && data != null) {
                        mBitmap = (Bitmap) data.getExtras().get("data");
                        Matrix matrix = new Matrix();
                        matrix.postRotate(90.0f);
                        mBitmap = Bitmap.createBitmap(mBitmap, 0, 0, mBitmap.getWidth(), mBitmap.getHeight(), matrix, true);
                        mImageView.setImageBitmap(mBitmap);
                    }
                    break;
                case 1:
                    if (resultCode == RESULT_OK && data != null) {
                        Uri selectedImage = data.getData();
                        String[] filePathColumn = {MediaStore.Images.Media.DATA};
                        if (selectedImage != null) {
                            Cursor cursor = getContentResolver().query(selectedImage,
                                    filePathColumn, null, null, null);
                            if (cursor != null) {
                                cursor.moveToFirst();
                                int columnIndex = cursor.getColumnIndex(filePathColumn[0]);
                                String picturePath = cursor.getString(columnIndex);
                                mBitmap = BitmapFactory.decodeFile(picturePath);
                                Matrix matrix = new Matrix();
                                matrix.postRotate(90.0f);
                                mBitmap = Bitmap.createBitmap(mBitmap, 0, 0, mBitmap.getWidth(), mBitmap.getHeight(), matrix, true);
                                mImageView.setImageBitmap(mBitmap);
                                cursor.close();
                            }
                        }
                    }
                    break;
            }
        }
    }

    @Override
    public void run() {
        if (mRadioMobile.isChecked()){
            Log.d("Inference Mode","Mobile inference");
            Bitmap resizedBitmap = Bitmap.createScaledBitmap(mBitmap, PrePostProcessor.mInputWidth, PrePostProcessor.mInputHeight, true);
            final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(resizedBitmap, PrePostProcessor.NO_MEAN_RGB, PrePostProcessor.NO_STD_RGB);
            IValue[] outputTuple = mModule.forward(IValue.from(inputTensor)).toTuple();
            final Tensor outputTensor = outputTuple[0].toTensor(); // (1, 25200, 85)
            float[] outputs = outputTensor.getDataAsFloatArray();
            final ArrayList<Result> results =  PrePostProcessor.outputsToNMSPredictions(outputs, mImgScaleX, mImgScaleY, mIvScaleX, mIvScaleY, mStartX, mStartY);
            runOnUiThread(() -> {
                mButtonDetect.setEnabled(true);
                mButtonDetect.setText(getString(R.string.detect));
                mProgressBar.setVisibility(ProgressBar.INVISIBLE);
                mResultView.setResults(results);
                mResultView.invalidate();
                mResultView.setVisibility(View.VISIBLE);
                long endTime = System.currentTimeMillis();
                long duration = endTime - startTime;
                Toast.makeText(getApplicationContext(), "Duration: " + duration + "ms",Toast.LENGTH_SHORT).show();
            });
        }
        else {
            Log.d("Inference Mode","Cloud inference");
            ArrayList<Result> results = new ArrayList<>();
            StringRequest request = new StringRequest(Request.Method.POST, URL, new Response.Listener<String>() {
                @Override
                public void onResponse(String response) { // Callback function that handles the response
                    Log.d("Cloud Inference", response.toString());
                    JSONObject json = null;
                    try {
                        json = new JSONObject(response);
                    } catch (JSONException e) {
                        e.printStackTrace();
                    }

                    JSONArray res = null;
                    try {
                        res = json.getJSONArray("results");
                        System.out.println(res);
                    } catch (JSONException e) {
                        e.printStackTrace();
                    }

                    float[][] fArr = null;
                    try {
                        fArr = new float[res.length()][6];
                        for (int i = 0; i < res.length(); i++) {
                            JSONArray row  = res.getJSONArray(i);
                            for (int j = 0; j < row.length(); j++) {
                                fArr[i][j] = Float.parseFloat(row.getString(j));
                            }
                        }
                    } catch (Exception e) {
                        e.printStackTrace();
                    }

                    for (int i = 0; i < fArr.length; i++) {
                        float x1 = fArr[i][0];
                        float y1 = fArr[i][1];
                        float x2 = fArr[i][2];
                        float y2 = fArr[i][3];
                        int cls = (int)fArr[i][5];
                        float score = fArr[i][4];
                        results.add(boxToResult(x1, y1, x2, y2, cls, score));
                    }

                    runOnUiThread(() -> {
                        mButtonDetect.setEnabled(true);
                        mButtonDetect.setText(getString(R.string.detect));
                        mProgressBar.setVisibility(ProgressBar.INVISIBLE);
                        mResultView.setResults(results);
                        mResultView.invalidate();
                        mResultView.setVisibility(View.VISIBLE);
                        long endTime = System.currentTimeMillis();
                        long duration = endTime - startTime;
                        Toast.makeText(getApplicationContext(), "Duration: " + duration + "ms",Toast.LENGTH_SHORT).show();
                    });
                }
            }, new Response.ErrorListener() {
                @Override
                public void onErrorResponse(VolleyError volleyError) {
                    Log.e("Cloud Inference", "Some error occurred -> " + volleyError);
                }
            }) {
                @Override
                protected Map<String, String> getParams() throws AuthFailureError { // Function that 
                    Map<String, String> parameters = new HashMap<String, String>();

                    String b64imgStr = null;
                    try {
                        InputStream img = getAssets().open(mTestImages[mImageIndex]);
                        byte[] imgBytes = IOUtils.toByteArray(img);
                        b64imgStr = Base64.encodeToString(imgBytes, Base64.DEFAULT);
                    } catch (Exception e) {
                        System.out.println(e);
                    }
                    parameters.put("data", b64imgStr);
                    return parameters;
                }
            };
            RequestQueue rQueue = Volley.newRequestQueue(MainActivity.this);
            rQueue.add(request);
            Log.d("Cloud Inference", "Final Processing");
        }
    }

    public Result boxToResult(float x1, float y1, float x2, float y2, int cls, float score) {
        Rect rect = new Rect((int)(mStartX+mIvScaleX*x1), (int)(mStartY+mIvScaleY*y1), (int)(mStartX+mIvScaleX*x2), (int)(mStartY+mIvScaleY*y2));
        Result result = new Result(cls, score, rect);
        return result;
    }

}
