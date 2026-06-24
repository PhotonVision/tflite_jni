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

#pragma once

#include <vector>

#include <tensorflow/lite/c/c_api.h>

#include "utils.hpp"

/**
 * Enumeration of model versions. This matches the enum in the Java code, and
 * when one is updated the other should be as well. Note that YOLOV5 is omitted
 * since it is not supported.
 */
enum ModelVersion { YOLOV8 = 1, YOLOV11 = 2, YOLOV8_OBB = 3 };

/**
 * Where should TFLite run inference.
 */
enum TFLiteSource { NONE = 0, QNN = 1, CPU = 2, NUM_SOURCES = 3 };

inline bool uses_external_delegate(TFLiteSource source) {
  return source != CPU;
}

/**
 * Structure representing a TFLite detector instance.
 *
 * interpreter: Pointer to the TensorFlow Lite interpreter.
 * delegate: Pointer to the TensorFlow Lite delegate (if any).
 * model: Pointer to the TensorFlow Lite model.
 * version: The version of the model being used.
 */
class TFLiteDetector {
 public:
  TfLiteInterpreter* interpreter;
  TfLiteDelegate* delegate;
  TfLiteModel* model;
  TFLiteSource source;
  ModelVersion version;

  /**
   * Performs model post-processing including box decoding and NMS.
   * @param boxThresh Confidence threshold for filtering boxes.
   * @param nmsThreshold IoU threshold for Non-Maximum Suppression.
   * @param input_img_width Width of the input image.
   * @param input_img_height Height of the input image.
   * @return A vector of DetectResult containing the final detections.
   */
  std::vector<DetectResult> post_proc(double boxThresh, double nmsThreshold,
                                      int input_img_width,
                                      int input_img_height);

 private:
  std::vector<DetectResult> yolo_post_proc(double boxThresh,
                                           int input_img_width,
                                           int input_img_height);

  std::vector<DetectResult> obb_post_proc(double boxThresh, int input_img_width,
                                          int input_img_height);

  /**
   * Performs Non-Maximum Suppression (NMS) on a list of detection results.
   *
   * @param candidates The list of detection candidates.
   * @param nmsThreshold The IoU threshold for suppression.
   * @return A vector of filtered detection results.
   */
  std::vector<DetectResult> optimized_nms(std::vector<DetectResult>& candidates,
                                          float nmsThreshold);
};
