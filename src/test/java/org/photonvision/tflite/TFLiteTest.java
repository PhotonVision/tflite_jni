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

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeEach;
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
    private static int platform = TFLiteSource.CPU.value();

    @BeforeEach
    public void useRubik() {
        if (System.getProperty("useRubik") != null) {
            platform = TFLiteSource.RUBIK.value();
            return;
        }

        try (BufferedReader reader = Files.newBufferedReader(Paths.get("/proc/device-tree/model"))) {
            while (true) {
                String value = reader.readLine();
                if (value == null) {
                    return;
                } else if (value.contains("RUBIK")) {
                    platform = TFLiteSource.RUBIK.value();
                    return;
                }
            }
        } catch (IOException ex) {
            return;
        }
    }

    private TFLiteResult[] runDetection(
            String modelName, String imagePath, int modelVersion, double boxThresh, double nmsThreshold)
            throws IOException {
        CombinedRuntimeLoader.loadLibraries(TFLiteTest.class, Core.NATIVE_LIBRARY_NAME);

        String modelPath = "src/test/resources/models/" + modelName + ".tflite";

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
        System.out.println("Image loaded: " + img.size() + " " + img.type());

        System.out.println("Creating TFLite detector");
        long ptr = TFLiteJNI.create(modelPath, modelVersion, platform);
        System.out.println("TFLite detector created: " + ptr);

        TFLiteResult[] ret = TFLiteJNI.detect(ptr, img.getNativeObjAddr(), boxThresh, nmsThreshold);

        System.out.println("Releasing TFLite detector");
        TFLiteJNI.destroy(ptr);
        img.release();

        return ret;
    }

    public void testModel(
            String modelName, String imagePath, int modelVersion, TFLiteResult[] expectedResults) {
        try {
            TFLiteResult[] ret = runDetection(modelName, imagePath, modelVersion, 0.7f, 0.45f);

            System.out.println("Detection results: " + Arrays.toString(ret));

            System.out.println("Expected detection results: " + Arrays.toString(expectedResults));

            // We can't guarantee expected results will be in the same order on every platform so just
            // check the length is the same and check each result is there somewhere
            // See https://github.com/PhotonVision/tflite_jni/pull/35#issuecomment-4423522333 for why we
            // need to tolerance the results like this
            assertTrue(ret.length == expectedResults.length, "Results should be the same length");

            boolean[] matched = new boolean[ret.length];
            for (TFLiteResult expected : expectedResults) {
                boolean found = false;
                for (int i = 0; i < ret.length; i++) {
                    if (!matched[i] && resultMatches(ret[i], expected)) {
                        matched[i] = true;
                        found = true;
                        break;
                    }
                }
                assertTrue(found, "Expected result not found: " + expected);
            }

            Mat img = Imgcodecs.imread(imagePath);
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
                    imagePath.substring(0, imagePath.lastIndexOf('.')) + modelName + "_with_results.jpg";

            // Save the image with results
            Imgcodecs.imwrite(newImagePath, img);
            System.out.println("Results written to image and saved as " + newImagePath);
            img.release();
        } catch (IOException e) {
            org.junit.jupiter.api.Assertions.fail("Unexpected IOException while running model test", e);
        }
    }

    private boolean withinTolerance(double actual, double expected, double tolerance) {
        if (expected == 0) {
            return actual == 0;
        }
        return Math.abs(actual - expected) / Math.abs(expected) <= tolerance;
    }

    private boolean resultMatches(TFLiteResult actual, TFLiteResult expected) {
        if (actual.class_id != expected.class_id) {
            return false;
        }

        double tolerance = 0.15;

        if (!withinTolerance(actual.rect.center.x, expected.rect.center.x, tolerance)) {
            return false;
        }
        if (!withinTolerance(actual.rect.center.y, expected.rect.center.y, tolerance)) {
            return false;
        }
        if (!withinTolerance(actual.rect.size.width, expected.rect.size.width, tolerance)) {
            return false;
        }
        if (!withinTolerance(actual.rect.size.height, expected.rect.size.height, tolerance)) {
            return false;
        }
        if (!withinTolerance(actual.rect.angle, expected.rect.angle, tolerance)) {
            return false;
        }
        if (!withinTolerance(actual.conf, expected.conf, tolerance)) {
            return false;
        }

        return true;
    }

    private boolean isModelQuantized(String modelName, int modelVersion) throws IOException {
        CombinedRuntimeLoader.loadLibraries(TFLiteTest.class, Core.NATIVE_LIBRARY_NAME);

        String modelPath = "src/test/resources/models/" + modelName + ".tflite";

        Path localSo = Path.of("cmake_build", "lib", "libtflite_jni.so").toAbsolutePath();
        Assumptions.assumeTrue(
                Files.exists(localSo),
                "Native library not found at " + localSo + " (run the native build first)");
        System.load(localSo.toString());

        long ptr = TFLiteJNI.create(modelPath, modelVersion, platform);
        if (ptr == 0) {
            throw new RuntimeException("Failed to create TFLite detector");
        }

        boolean quantized = TFLiteJNI.isQuantized(ptr);
        TFLiteJNI.destroy(ptr);
        return quantized;
    }

    @Test
    public void testYoloV8IsQuantized() throws IOException {
        assertTrue(isModelQuantized("yolov8nCoco", 1), "yolov8nCoco should be quantized");
    }

    @Test
    public void testYoloV8NonQuantIsNotQuantized() throws IOException {
        assertTrue(
                !isModelQuantized("yolov8nCoco_nonquant", 1),
                "yolov8nCoco_nonquant should not be quantized");
    }

    @Test
    public void testYoloV11IsQuantized() throws IOException {
        assertTrue(isModelQuantized("yolov11nCoco", 2), "yolov11nCoco should be quantized");
    }

    @Test
    public void testYoloV11NonQuantIsNotQuantized() throws IOException {
        assertTrue(
                !isModelQuantized("yolov11nCoco_nonquant", 2),
                "yolov11nCoco_nonquant should not be quantized");
    }

    @Test
    public void testYoloV8() {
        TFLiteResult[] expectedResults = {
            new TFLiteResult(206, 235, 274, 509, 0.8782271f, 0, 0.0f),
            new TFLiteResult(95, 137, 545, 447, 0.84069604f, 5, 0.0f),
            new TFLiteResult(118, 232, 229, 532, 0.84069604f, 0, 0.0f),
            new TFLiteResult(483, 222, 562, 522, 0.84069604f, 0, 0.0f),
        };
        testModel("yolov8nCoco", "src/test/resources/images/bus.jpg", 1, expectedResults);
    }

    @Test
    public void testYoloV11() {
        TFLiteResult[] expectedResults = {
            new TFLiteResult(91, 145, 552, 435, 0.9453125f, 5, 0.0f),
            new TFLiteResult(104, 246, 214, 536, 0.8984375f, 0, 0.0f),
            new TFLiteResult(221, 233, 281, 504, 0.8515625f, 0, 0.0f),
            new TFLiteResult(482, 224, 561, 514, 0.8203125f, 0, 0.0f),
        };
        testModel("yolov11nCoco", "src/test/resources/images/bus.jpg", 2, expectedResults);
    }

    @Test
    public void testYoloV8ConfidenceThresholding() throws IOException {
        TFLiteResult[] highThreshResults =
                runDetection("yolov8nCoco", "src/test/resources/images/bus.jpg", 1, 0.99, 0.45);
        assertTrue(
                highThreshResults.length == 0,
                "High confidence threshold (0.99) should filter out all detections");

        TFLiteResult[] midThreshResults =
                runDetection("yolov8nCoco", "src/test/resources/images/bus.jpg", 1, 0.5, 0.45);
        assertTrue(
                midThreshResults.length > 0, "Mid confidence threshold (0.5) should return some results");

        TFLiteResult[] lowThreshResults =
                runDetection("yolov8nCoco", "src/test/resources/images/bus.jpg", 1, 0.1, 0.45);
        assertTrue(
                lowThreshResults.length > midThreshResults.length,
                "Low confidence threshold (0.1) should return more detections than default (0.5)");
    }

    @Test
    public void testYoloV11ConfidenceThresholding() throws IOException {
        TFLiteResult[] highThreshResults =
                runDetection("yolov11nCoco", "src/test/resources/images/bus.jpg", 2, 0.99, 0.45);
        assertTrue(
                highThreshResults.length == 0,
                "High confidence threshold (0.99) should filter out all detections");

        TFLiteResult[] midThreshResults =
                runDetection("yolov11nCoco", "src/test/resources/images/bus.jpg", 2, 0.5, 0.45);
        assertTrue(
                midThreshResults.length > 0, "Mid confidence threshold (0.5) should return detections");

        TFLiteResult[] lowThreshResults =
                runDetection("yolov11nCoco", "src/test/resources/images/bus.jpg", 2, 0.1, 0.45);
        assertTrue(
                lowThreshResults.length > midThreshResults.length,
                "Low confidence threshold (0.1) should return more detections than mid (0.5)");
    }

    @Test
    public void testYoloV8NmsThresholding() throws IOException {
        TFLiteResult[] highNmsResults =
                runDetection("yolov8nCoco", "src/test/resources/images/bus.jpg", 1, 0.1, 0.99);
        TFLiteResult[] midNmsResults =
                runDetection("yolov8nCoco", "src/test/resources/images/bus.jpg", 1, 0.1, 0.75);
        TFLiteResult[] lowNmsResults =
                runDetection("yolov8nCoco", "src/test/resources/images/bus.jpg", 1, 0.1, 0.45);

        assertTrue(
                withinTolerance(highNmsResults.length, 29, 0.15),
                "High NMS (0.99) should return ~29 detections");
        assertTrue(
                withinTolerance(midNmsResults.length, 9, 0.15),
                "Mid NMS (0.75) should return ~9 detections");
        assertTrue(
                withinTolerance(lowNmsResults.length, 6, 0.2),
                "Low NMS (0.45) should return ~6 detections");

        // All mid-NMS results must be present in high-NMS results
        for (TFLiteResult mid : midNmsResults) {
            boolean found = false;
            for (TFLiteResult high : highNmsResults) {
                if (resultMatches(mid, high)) {
                    found = true;
                    break;
                }
            }
            assertTrue(found, "Mid-NMS result should be present in high-NMS results: " + mid);
        }

        // All low-NMS results must be present in mid-NMS results
        for (TFLiteResult low : lowNmsResults) {
            boolean found = false;
            for (TFLiteResult mid : midNmsResults) {
                if (resultMatches(low, mid)) {
                    found = true;
                    break;
                }
            }
            assertTrue(found, "Low-NMS result should be present in mid-NMS results: " + low);
        }
    }

    @Test
    public void testYoloV11NmsThresholding() throws IOException {
        TFLiteResult[] highNmsResults =
                runDetection("yolov11nCoco", "src/test/resources/images/bus.jpg", 2, 0.1, 0.99);
        TFLiteResult[] midNmsResults =
                runDetection("yolov11nCoco", "src/test/resources/images/bus.jpg", 2, 0.1, 0.75);
        TFLiteResult[] lowNmsResults =
                runDetection("yolov11nCoco", "src/test/resources/images/bus.jpg", 2, 0.1, 0.45);

        assertTrue(
                withinTolerance(highNmsResults.length, 29, 0.15),
                "High NMS (0.99) should return ~25 detections");
        assertTrue(
                withinTolerance(midNmsResults.length, 8, 0.15),
                "Mid NMS (0.75) should return ~8 detections");
        assertTrue(
                withinTolerance(lowNmsResults.length, 5, 0.2),
                "Low NMS (0.45) should return ~6 detections");

        // All mid-NMS results must be present in high-NMS results
        for (TFLiteResult mid : midNmsResults) {
            boolean found = false;
            for (TFLiteResult high : highNmsResults) {
                if (resultMatches(mid, high)) {
                    found = true;
                    break;
                }
            }
            assertTrue(found, "Mid-NMS result should be present in high-NMS results: " + mid);
        }

        // All low-NMS results must be present in mid-NMS results
        for (TFLiteResult low : lowNmsResults) {
            boolean found = false;
            for (TFLiteResult mid : midNmsResults) {
                if (resultMatches(low, mid)) {
                    found = true;
                    break;
                }
            }
            assertTrue(found, "Low-NMS result should be present in mid-NMS results: " + low);
        }
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
        Path localSo = Path.of("cmake_build", "lib", "libtflite_jni.so").toAbsolutePath();
        Assumptions.assumeTrue(
                Files.exists(localSo),
                "Native library not found at " + localSo + " (run the native build first)");
        System.load(localSo.toString());

        int numRuns = Integer.parseInt(System.getProperty("memLeakTestIterations"));

        System.out.println("Starting memory leak finder test; running for " + numRuns + " iterations");
        for (int i = 0; i < numRuns; i++) {
            if (i % 1000 == 0) {
                System.out.println("Iteration " + i);
            }

            // Create a TFLite detector instance
            long ptr = TFLiteJNI.create("src/test/resources/models/yolov8nCoco.tflite", 1, platform);

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
        Path localSo = Path.of("cmake_build", "lib", "libtflite_jni.so").toAbsolutePath();
        Assumptions.assumeTrue(
                Files.exists(localSo),
                "Native library not found at " + localSo + " (run the native build first)");
        System.load(localSo.toString());

        System.out.println("Loading bus");
        Mat img = Imgcodecs.imread("src/test/resources/images/bus.jpg");

        if (img.empty()) {
            throw new RuntimeException("Failed to load image");
        }

        System.out.println("Image loaded: " + img.size() + " " + img.type());

        System.out.println("Creating TFLite detector");
        long ptr = TFLiteJNI.create("src/test/resources/models/yolov8nCoco.tflite", 1, platform);

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
