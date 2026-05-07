/*
 * Copyright (C) Photon Vision.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

package org.photonvision.tflite;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.photonvision.jni.CombinedRuntimeLoader;
import org.photonvision.tflite.TFLiteJNI.TFLiteResult;
import org.photonvision.tflite.TFLiteJNI.TFLiteSource;

public class TFLiteTest {
    public void testModel(String modelPath, String imagePath, int modelVersion) {
        try {
            CombinedRuntimeLoader.loadLibraries(TFLiteTest.class, Core.NATIVE_LIBRARY_NAME);

            System.out.println(Core.getBuildInformation());
            System.out.println(Core.OpenCLApiCallError);

            System.out.println("Loading tflite_jni");
            Path localSo = Path.of("cmake_build", "lib", "libtflite_jni.so").toAbsolutePath();
            Assumptions.assumeTrue(
                    Files.exists(localSo),
                    "Native library not found at " + localSo + " (run the native build first)");
            System.load(localSo.toString());

            System.out.println("Loading image: " + imagePath);
            Mat img = Imgcodecs.imread(imagePath);

            if (img.empty()) {
                throw new RuntimeException("Failed to load image");
            }

            System.out.println("Image loaded: " + img.size() + " " + img.type());

            System.out.println("Creating TFLite detector");
            long ptr = TFLiteJNI.create(modelPath, modelVersion, TFLiteSource.CPU.value());

            if (ptr == 0) {
                throw new RuntimeException("Failed to create TFLite detector");
            }

            System.out.println("TFLite detector created: " + ptr);

            assertTrue(TFLiteJNI.isQuantized(ptr), "TFLite detector should be quantized");

            TFLiteResult[] ret = TFLiteJNI.detect(ptr, img.getNativeObjAddr(), 0.5f, 0.45f);

            System.out.println("Detection results: " + Arrays.toString(ret));

            System.out.println("Releasing TFLite detector");
            TFLiteJNI.destroy(ptr);

            for (TFLiteResult result : ret) {
                System.out.println("Result: " + result);

                Scalar color = new Scalar(0, 255, 0); // Green color is default for bounding box

                if (result.class_id == 0) {
                    color = new Scalar(255, 0, 0); // Blue for person
                } else if (result.class_id == 5) {
                    color = new Scalar(0, 0, 255); // Red for bus
                }

                // Draw bounding box on the image
                Point[] rectPoints = new Point[4];
                result.rect.points(rectPoints);
                for (int j = 0; j < 4; j++) {
                    Imgproc.line(img, rectPoints[j], rectPoints[(j + 1) % 4], color, 2, 8);
                }
            }

            String newImagePath =
                    imagePath.substring(0, imagePath.lastIndexOf('.')) + "_with_results.jpg";

            // Save the image with results
            Imgcodecs.imwrite(newImagePath, img);
            System.out.println("Results written to image and saved as " + newImagePath);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Test
    public void testYoloV8() {
        testModel(
                "src/test/resources/models/yolov8nCoco.tflite", "src/test/resources/images/bus.jpg", 1);
    }

    // Helper method to determine if the memory leak test should be enabled
    static boolean isIterationTestEnabled(String param) {
        String iterations = System.getProperty(param);
        if (iterations == null || iterations.trim().isEmpty()) {
            System.out.println(param + " property not set or empty; skipping memory leak test.");
            return false;
        }

        try {
            int numIterations = Integer.parseInt(iterations.trim());
            return numIterations > 0;
        } catch (NumberFormatException e) {
            return false;
        }
    }

    static boolean memLeakEnabled() {
        return isIterationTestEnabled("memLeakTestIterations");
    }

    /**
     * This test will create and destroy a TFLite detector repeatedly to try and cause memory leaks.
     * To find a memory leak, it's necessary to manually watch memory as this test runs, as the test
     * itself does not check memory. It can be enabled by setting the number of iterations, using the
     * system property "memLeakTestIterations".
     */
    @Test
    @org.junit.jupiter.api.condition.EnabledIf("memLeakEnabled")
    public void memLeakFinder() {
        try {
            CombinedRuntimeLoader.loadLibraries(TFLiteTest.class, Core.NATIVE_LIBRARY_NAME);
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }

        System.out.println(Core.getBuildInformation());
        System.out.println(Core.OpenCLApiCallError);

        System.out.println("Loading tflite_jni");
        System.load("/home/photon/tflite_jni/cmake_build/libtflite_jni.so");

        int numRuns = Integer.parseInt(System.getProperty("memLeakTestIterations"));

        System.out.println("Starting memory leak finder test; running for " + numRuns + " iterations");
        for (int i = 0; i < numRuns; i++) {
            if (i % 1000 == 0) {
                System.out.println("Iteration " + i);
            }

            // Create a TFLite detector instance
            long ptr =
                    TFLiteJNI.create(
                            "src/test/resources/models/yolov8nCoco.tflite", 0, TFLiteSource.CPU.value());

            if (ptr == 0) {
                throw new RuntimeException("Failed to create TFLite detector");
            }
            TFLiteJNI.destroy(ptr);
        }
    }

    static boolean benchmarkEnabled() {
        return isIterationTestEnabled("benchmarkIterations");
    }

    /**
     * This test will run the detect function repeatedly to benchmark performance. It can be enabled
     * by setting the number of iterations, using the system property "benchmarkIterations".
     */
    @Test
    @org.junit.jupiter.api.condition.EnabledIf("benchmarkEnabled")
    public void benchmark() {
        System.out.println("Running benchmark test");
        try {
            CombinedRuntimeLoader.loadLibraries(TFLiteTest.class, Core.NATIVE_LIBRARY_NAME);
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }

        System.out.println(Core.getBuildInformation());
        System.out.println(Core.OpenCLApiCallError);

        System.out.println("Loading tflite_jni");
        System.load("/home/photon/tflite_jni/cmake_build/libtflite_jni.so");

        System.out.println("Loading bus");
        Mat img = Imgcodecs.imread("src/test/resources/images/bus.jpg");

        if (img.empty()) {
            throw new RuntimeException("Failed to load image");
        }

        System.out.println("Image loaded: " + img.size() + " " + img.type());

        System.out.println("Creating TFLite detector");
        long ptr =
                TFLiteJNI.create(
                        "src/test/resources/models/yolov8nCoco.tflite", 1, TFLiteSource.CPU.value());

        if (ptr == 0) {
            throw new RuntimeException("Failed to create TFLite detector");
        }

        int numRuns = Integer.parseInt(System.getProperty("benchmarkIterations"));
        System.out.println("Starting benchmark; running for " + numRuns + " iterations");

        long startTime = System.nanoTime();

        for (int i = 0; i < numRuns; i++) {
            TFLiteJNI.detect(ptr, img.getNativeObjAddr(), 0.5f, 0.45f);
        }

        long endTime = System.nanoTime();
        long duration = endTime - startTime; // Duration in nanoseconds
        double avgDurationMs = (duration / 1_000_000.0) / numRuns; // Average duration in milliseconds

        System.out.printf(
                "Benchmark complete. Average detection time: %.2f ms over %d runs.%n",
                avgDurationMs, numRuns);

        System.out.println("Releasing TFLite detector");
        TFLiteJNI.destroy(ptr);
    }
}
