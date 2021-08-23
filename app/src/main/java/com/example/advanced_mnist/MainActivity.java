package com.example.advanced_mnist;

import androidx.appcompat.app.AppCompatActivity;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

public class MainActivity extends AppCompatActivity {

    ImageView imageView;
    TextView resultsTextView;

    static {
        System.loadLibrary("tensorflow_inference");
    }

    private static final String MODEL_FILE="file:///android_asset/optimized_advanced_mnist.pb";
    private static final String INPUT_NODE= "x_input";
    private static final String KEEP_PROB= "keep_prob";
    private static final int[] INPUT_SHAPE= {1, 784};
    private static final int[] KEEP_PROB_SHAPE= {1};
    private static final String OUTPUT_NODE= "y_readout1";
    private TensorFlowInferenceInterface inferenceInterface;

    private int imageIndex = 9;
    private final int[] imageResourceIDs = {
            R.drawable.digit0,
            R.drawable.digit1,
            R.drawable.digit2,
            R.drawable.digit3,
            R.drawable.digit4,
            R.drawable.digit5,
            R.drawable.digit6,
            R.drawable.digit7,
            R.drawable.digit8,
            R.drawable.digit9
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        imageView = findViewById(R.id.image_view);
        resultsTextView = findViewById(R.id.results_text_view);

        inferenceInterface = new TensorFlowInferenceInterface();
        inferenceInterface.initializeTensorFlow(getAssets(), MODEL_FILE);

    }

    public void loadImageAction (View view){
        imageIndex= (imageIndex >= 9) ? 0 : imageIndex+1;
        imageView.setImageResource(imageResourceIDs[imageIndex]);
    }

    public void guessImageAction (View view) {
        float[] pixelBuffer = convertImage();
        float[] results = predictDigit(pixelBuffer);
        printResults(results);
    }

    private void printResults (float[] results) {
        float max = 0;
        float secondMax = 0;
        int maxIndex= 0;
        int secondMaxIndex= 0;
        for (int i=0; i<10; i++) {
            if (results[i] > max) {
                secondMax=max;
                secondMaxIndex=maxIndex;
                max=results[i];
                maxIndex=i;
            } else if (results[i]< max && results[i]> secondMax) {
                secondMax=results[i];
                secondMaxIndex=i;
            }
        }
        String output = "Model predicts: "+ String.valueOf(maxIndex) +
                ", second choice: " +String.valueOf(secondMaxIndex);
        resultsTextView.setText(output);
    }
    private float[] predictDigit (float[] pixelBuffer) {
        inferenceInterface.fillNodeFloat(INPUT_NODE, INPUT_SHAPE, pixelBuffer);
        inferenceInterface.fillNodeFloat(KEEP_PROB, KEEP_PROB_SHAPE, new float[] {0.5f});
        inferenceInterface.runInference(new String[] {OUTPUT_NODE});
        float[] outputs = new float[10];
        inferenceInterface.readNodeFloat(OUTPUT_NODE, outputs);
        return outputs;
    }
    private float[] convertImage()
    {
        Bitmap imageBitmap = BitmapFactory.decodeResource(getResources(), imageResourceIDs[imageIndex]);
        imageBitmap = Bitmap.createScaledBitmap(imageBitmap, 28, 28, true);
        imageView.setImageBitmap(imageBitmap);
        int[] imageAsIntArray = new int[784];
        float[] imageAsFloatArray = new float[784];
        imageBitmap.getPixels(imageAsIntArray, 0, 28, 0, 0, 28, 28);
        for (int i=0; i<784; i++) {
            //pixel values are a number between 0 and -16777216
            //divide to convert numbers to be between 0 and 1
            imageAsFloatArray[i]= imageAsIntArray[i]/ -16777216;
        }
        return imageAsFloatArray;
    }
}