package com.example.test_custom_tflite;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.media.ThumbnailUtils;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import com.example.test_custom_tflite.ml.Model;
import com.example.test_custom_tflite.ml.Model1;
import com.example.test_custom_tflite.ml.Resnet8;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;
import org.tensorflow.lite.Tensor;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class MainActivity extends AppCompatActivity {

    Button camera , gallery;
    ImageView imageView;
    TextView result;

    int image_size = 32;
    int image_width = 72;
    int image_hight = 128;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        camera = findViewById(R.id.button);
        gallery = findViewById(R.id.button2);

        result = findViewById(R.id.result);
        imageView = findViewById(R.id.imageView);

        // Đường dẫn đến tệp tin mô hình TensorFlow Lite
        String modelPath = "your_model.tflite";
        //Interpreter mInterpreter = new Interpreter(FileUtil.loadFileFromAssets(this, MODEL_PATH));

        camera.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED)
                {
                    Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                    startActivityForResult(cameraIntent,3);


                }
                else
                {
                    requestPermissions(new String[] {Manifest.permission.CAMERA}, 100);

                }
            }
        });

        gallery.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {

                    Intent cameraIntent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                    startActivityForResult(cameraIntent,1);
            }
        });
    }

//    public void classifyImage1(Bitmap bitmap)
//    {
//        bitmap = Bitmap.createScaledBitmap(bitmap, 72, 128, true);
//        ByteBuffer input = ByteBuffer.allocateDirect(72 * 128 * 3 * 4).order(ByteOrder.nativeOrder());
//        for (int y = 0; y < 72; y++) {
//            for (int x = 0; x < 128; x++) {
//                int px = bitmap.getPixel(x, y);
//
//                // Get channel values from the pixel value.
//                int r = Color.red(px);
//                int g = Color.green(px);
//                int b = Color.blue(px);
//
//                // Normalize channel values to [-1.0, 1.0]. This requirement depends
//                // on the model. For example, some models might require values to be
//                // normalized to the range [0.0, 1.0] instead.
//                float rf = (r - 127) / 255.0f;
//                float gf = (g - 127) / 255.0f;
//                float bf = (b - 127) / 255.0f;
//
//                input.putFloat(rf);
//                input.putFloat(gf);
//                input.putFloat(bf);
//            }
//        }
//
//        int bufferSize = 4 * java.lang.Float.SIZE / java.lang.Byte.SIZE;
//        ByteBuffer modelOutput = ByteBuffer.allocateDirect(bufferSize).order(ByteOrder.nativeOrder());
//        interpreter.run(input, modelOutput);
//    }

    public void classifyImage(Bitmap image)
    {
        try {
            //Model model = Model.newInstance(getApplicationContext());
            Resnet8 model = Resnet8.newInstance(getApplicationContext());
            //Model1 model = Model1.newInstance(getApplicationContext());
            // Creates inputs for reference.
            //TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 32, 32, 3}, DataType.FLOAT32);
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 3, 72, 128}, DataType.FLOAT32);
            //TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 128, 128, 3}, DataType.FLOAT32);

            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(3 * 72 * 128 * 4);
            byteBuffer.order(ByteOrder.nativeOrder());

            int[] intValues = new int[72 * 128];
            image.getPixels(intValues, 0, image.getWidth(), 0, 0, image.getWidth(), image.getHeight());
            int pixel = 0;
            //iterate over each pixel and extract R, G, and B values. Add those values individually to the byte buffer.
            for(int i = 0; i < 72; i ++){
                for(int j = 0; j < 128; j++){
                    int val = intValues[pixel++]; // RGB
                    byteBuffer.putFloat(((val >> 16) & 0xFF) * (1.f / 255));
                    byteBuffer.putFloat(((val >> 8) & 0xFF) * (1.f / 255));
                    byteBuffer.putFloat((val & 0xFF) * (1.f / 255));
                }
            }

            inputFeature0.loadBuffer(byteBuffer);

            // Runs model inference and gets result.
            //Model.Outputs outputs = model.process(inputFeature0);
            Resnet8.Outputs outputs = model.process(inputFeature0);
            //Model1.Outputs outputs = model.process(inputFeature0);

            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();


            float[] confidence = outputFeature0.getFloatArray();

            int maxPos = 0;
            float maxConfidence = 0;
            for (int i = 0; i < confidence.length; i++)
            {
                Log.d("Check" + i, String.valueOf(confidence[i]));
                if (confidence[i] >= maxConfidence)
                {
                    maxConfidence = confidence[i];
                    maxPos = i;
                }
            }

            //String[] classes = {"Apple", "Banana", "Orange","Pass"};
            String[] classes = {"up", "left", "right","idle","stand", "up1"};

            result.setText(classes[maxPos]);
            // Releases model resources if no longer used.
            model.close();
        } catch (IOException e) {
            // TODO Handle the exception
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        if (resultCode == RESULT_OK)
        {
            if (requestCode == 3)
            {
                Bitmap image = (Bitmap) data.getExtras().get("data");
                int dimension = Math.min (image.getWidth(), image.getHeight());
                image = ThumbnailUtils.extractThumbnail(image, dimension, dimension);
                imageView.setImageBitmap(image);

                image = Bitmap.createScaledBitmap(image, image_width,image_hight, false);
                classifyImage(image);
            }
            else
            {
                Uri dat = data.getData();
                Bitmap image = null;
                try {
                    image = MediaStore.Images.Media.getBitmap(this.getContentResolver(),dat);
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
                imageView.setImageBitmap(image);

                image = Bitmap.createScaledBitmap(image, image_width,image_hight, false);
                classifyImage(image);
            }
        }

        super.onActivityResult(requestCode, resultCode, data);
    }
}