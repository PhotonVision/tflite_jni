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

import org.opencv.core.Point;
import org.opencv.core.RotatedRect;
import org.opencv.core.Size;

public class TFLiteJNI {
    public static enum TFLiteSource {
        QNN(1),
        CPU(2);

        private int value = 0;

        TFLiteSource(int value) {
            this.value = value;
        }

        public int value() {
            return value;
        }
    }

    /** A class representing the result of a detection. */
    public static class TFLiteResult {
        /**
         * Create a TFLiteResult with the specified bounding box coordinates, confidence, class ID, and
         * angle.
         *
         * @param x1 The x coordinate of a vertex of the bounding box.
         * @param y1 The y coordinate of a vertex of the bounding box.
         * @param x2 The x coordinate of the opposite vertex of the bounding box.
         * @param y2 The y coordinate of the opposite vertex of the bounding box.
         * @param conf The confidence score of the detection.
         * @param angle The angle of the detected object in degrees.
         * @param class_id The class ID of the detected object.
         */
        public TFLiteResult(int x1, int y1, int x2, int y2, float conf, int class_id, float angle) {
            this.conf = conf;
            this.class_id = class_id;

            // Calc size
            double width = x2 - x1;
            double height = y2 - y1;

            Point center = new Point((x1 + x2) / 2.0, (y1 + y2) / 2.0);

            this.rect = new RotatedRect(center, new Size(width, height), angle);
        }

        public final RotatedRect rect;
        public final float conf;
        public final int class_id;

        @Override
        public String toString() {
            return "TFLiteResult [rect=" + rect + ", conf=" + conf + ", class_id=" + class_id + "]";
        }

        @Override
        public int hashCode() {
            final int prime = 31;
            int result = 1;
            result = prime * result + ((rect == null) ? 0 : rect.hashCode());
            result = prime * result + Float.floatToIntBits(conf);
            result = prime * result + class_id;
            return result;
        }

        @Override
        public boolean equals(Object obj) {
            if (this == obj) return true;
            if (obj == null) return false;
            if (getClass() != obj.getClass()) return false;
            TFLiteResult other = (TFLiteResult) obj;
            if (rect == null) {
                if (other.rect != null) return false;
            } else if (!rect.equals(other.rect)) return false;
            if (Float.floatToIntBits(conf) != Float.floatToIntBits(other.conf)) return false;
            if (class_id != other.class_id) return false;
            return true;
        }
    }

    /**
     * Create a TFLiteJNI instance with the specified model path.
     *
     * @param modelPath Absolute path to the model file
     * @param version The version of the model. This should be the ordinal from the ModelVersion enum.
     *     If this enum changes, it should be changed in the JNI as well.
     * @param source The source TFLite should use to run models. If this enum changes, it should be
     *     changed in the JNI as well.
     * @return A pointer to a struct with the tflite detector instance.
     */
    public static native long create(String modelPath, int version, int source);

    /**
     * Destroy the TFLiteJNI instance.
     *
     * @param ptr The pointer to the tflite detector instance.
     */
    public static native void destroy(long ptr);

    /**
     * Detect in the given image
     *
     * @param detectorPtr The pointer to the tflite detector instance.
     * @param imagePtr The pointer to the image data.
     * @param boxThresh The threshold for the bounding box detection.
     * @param nmsThreshold The threshold for non-maximum suppression.
     * @return An array of {@link TFLiteJNI.TFLiteResult} objects containing the detection results.
     */
    public static native TFLiteResult[] detect(
            long detectorPtr, long imagePtr, double boxThresh, double nmsThreshold);

    /**
     * Check model quantization.
     *
     * @param detectorPtr The pointer to the tflite detector instance.
     * @return A boolean indicating whether the model is quantized.
     */
    public static native boolean isQuantized(long detectorPtr);
}
