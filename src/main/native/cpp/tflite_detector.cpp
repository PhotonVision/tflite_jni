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

#include "tflite_detector.hpp"

#include <algorithm>
#include <cstdio>
#include <vector>

#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <tensorflow/lite/c/c_api.h>
#include <tensorflow/lite/c/c_api_experimental.h>
#include <tensorflow/lite/delegates/external/external_delegate.h>
#include <tensorflow/lite/version.h>

#include "utils.hpp"

std::vector<DetectResult> TFLiteDetector::post_proc(double boxThresh,
                                                    double nmsThreshold,
                                                    int input_img_width,
                                                    int input_img_height) {
  std::vector<DetectResult> candidate_results;
  switch (version) {
    case YOLOV8:
    case YOLOV11:
      candidate_results =
          yolo_post_proc(boxThresh, input_img_width, input_img_height);
      break;
    case YOLOV8_OBB:
      candidate_results =
          obb_post_proc(boxThresh, input_img_width, input_img_height);
      break;
  }

  return optimized_nms(candidate_results, nmsThreshold);
}

std::vector<DetectResult> TFLiteDetector::yolo_post_proc(double boxThresh,
                                                         int input_img_width,
                                                         int input_img_height) {
  const TfLiteTensor* boxesTensor =
      TfLiteInterpreterGetOutputTensor(interpreter, 0);
  const TfLiteTensor* scoresTensor =
      TfLiteInterpreterGetOutputTensor(interpreter, 1);
  const TfLiteTensor* classesTensor =
      TfLiteInterpreterGetOutputTensor(interpreter, 2);

  const TfLiteQuantizationParams boxesParams =
      TfLiteTensorQuantizationParams(boxesTensor);
  const TfLiteQuantizationParams scoresParams =
      TfLiteTensorQuantizationParams(scoresTensor);

  const int numBoxes = TfLiteTensorDim(boxesTensor, 1);
  DEBUG_PRINT("INFO: Detected %d boxes\n", numBoxes);

  // Debug tensor shapes
  DEBUG_PRINT("DEBUG: Boxes tensor dimensions: ");
#ifndef NDEBUG
  for (int i = 0; i < TfLiteTensorNumDims(boxesTensor); i++) {
    std::printf("%d ", TfLiteTensorDim(boxesTensor, i));
  }
  std::printf("\n");
#endif

  DEBUG_PRINT("DEBUG: Scores tensor dimensions: ");
#ifndef NDEBUG
  for (int i = 0; i < TfLiteTensorNumDims(scoresTensor); i++) {
    std::printf("%d ", TfLiteTensorDim(scoresTensor, i));
  }
  std::printf("\n");
#endif

  DEBUG_PRINT("DEBUG: Classes tensor dimensions: ");
#ifndef NDEBUG
  for (int i = 0; i < TfLiteTensorNumDims(classesTensor); i++) {
    std::printf("%d ", TfLiteTensorDim(classesTensor, i));
  }
  std::printf("\n");
#endif

  if (TfLiteTensorType(boxesTensor) != kTfLiteUInt8 ||
      TfLiteTensorType(scoresTensor) != kTfLiteUInt8 ||
      TfLiteTensorType(classesTensor) != kTfLiteUInt8) {
    throw std::runtime_error("Expected tensor type to be uint8");
  }

  uint8_t* boxesData = static_cast<uint8_t*>(TfLiteTensorData(boxesTensor));
  uint8_t* scoresData = static_cast<uint8_t*>(TfLiteTensorData(scoresTensor));
  uint8_t* classesData = static_cast<uint8_t*>(TfLiteTensorData(classesTensor));

  DEBUG_PRINT("DEBUG: Quantization params - boxes: zp=%d, scale=%f\n",
              boxesParams.zero_point, boxesParams.scale);
  DEBUG_PRINT("DEBUG: Quantization params - scores: zp=%d, scale=%f\n",
              scoresParams.zero_point, scoresParams.scale);

  std::vector<DetectResult> results;

  DEBUG_PRINT("DEBUG: Image dimensions: %dx%d\n", input_img_width,
              input_img_height);

  for (int i = 0; i < numBoxes; ++i) {
    // Use proper dequantization for score
    float score =
        get_dequant_value(scoresData, kTfLiteUInt8, i, scoresParams.zero_point,
                          scoresParams.scale);
    if (score < boxThresh) {
      continue;
    }

    int classId = classesData[i];

    // For tensor shape [1, 8400, 4], use sequential indexing per detection
    uint8_t raw_x_1_u8 = boxesData[i * 4 + 0];
    uint8_t raw_y_1_u8 = boxesData[i * 4 + 1];
    uint8_t raw_x_2_u8 = boxesData[i * 4 + 2];
    uint8_t raw_y_2_u8 = boxesData[i * 4 + 3];

    // Use proper dequantization for bbox coordinates (like we do for scores)
    float x1 = get_dequant_value(&raw_x_1_u8, kTfLiteUInt8, 0,
                                 boxesParams.zero_point, boxesParams.scale);
    float y1 = get_dequant_value(&raw_y_1_u8, kTfLiteUInt8, 0,
                                 boxesParams.zero_point, boxesParams.scale);
    float x2 = get_dequant_value(&raw_x_2_u8, kTfLiteUInt8, 0,
                                 boxesParams.zero_point, boxesParams.scale);
    float y2 = get_dequant_value(&raw_y_2_u8, kTfLiteUInt8, 0,
                                 boxesParams.zero_point, boxesParams.scale);

    float clamped_x1 =
        std::max(0.0f, std::min(x1, static_cast<float>(input_img_width)));
    float clamped_y1 =
        std::max(0.0f, std::min(y1, static_cast<float>(input_img_height)));
    float clamped_x2 =
        std::max(0.0f, std::min(x2, static_cast<float>(input_img_width)));
    float clamped_y2 =
        std::max(0.0f, std::min(y2, static_cast<float>(input_img_height)));

    // Skip bad boxes
    if (clamped_x1 >= clamped_x2 || clamped_y1 >= clamped_y2) {
      continue;
    }

#ifndef NDEBUG
    if (results.size() < 5) {
      std::printf(" DEBUG: box %d - uint8 corners: (%d, %d) to (%d, %d)\n", i,
                  raw_x_1_u8, raw_y_1_u8, raw_x_2_u8, raw_y_2_u8);
      std::printf(
          "DEBUG: box %d - dequantized corners: (%.2f, %.2f) to (%.2f, %.2f)\n",
          i, x1, y1, x2, y2);
      std::printf(
          "DEBUG: box %d - clamped corners: (%.2f, %.2f) to (%.2f, %.2f), "
          "score=%.3f, class=%d\n",
          i, clamped_x1, clamped_y1, clamped_x2, clamped_y2, score, classId);
    }
#endif

    BoundingBox box{static_cast<int>(std::round(clamped_x1)),
                    static_cast<int>(std::round(clamped_y1)),
                    static_cast<int>(std::round(clamped_x2)),
                    static_cast<int>(std::round(clamped_y2)), 0.0};

    DetectResult det{classId, box, score};

    results.push_back(det);
  }

  return results;
}

std::vector<DetectResult> TFLiteDetector::obb_post_proc(double boxThresh,
                                                        int input_img_width,
                                                        int input_img_height) {
  const TfLiteTensor* outputTensor =
      TfLiteInterpreterGetOutputTensor(interpreter, 0);

  const TfLiteQuantizationParams outputParams =
      TfLiteTensorQuantizationParams(outputTensor);

  const int numBoxes = TfLiteTensorDim(outputTensor, 1);
  DEBUG_PRINT("INFO: Detected %d boxes\n", numBoxes);

  // Debug tensor shapes
  DEBUG_PRINT("DEBUG: Boxes tensor dimensions: ");
#ifndef NDEBUG
  for (int i = 0; i < TfLiteTensorNumDims(outputTensor); i++) {
    std::printf("%d ", TfLiteTensorDim(outputTensor, i));
  }
  std::printf("\n");
#endif

  if (TfLiteTensorType(outputTensor) != kTfLiteUInt8) {
    throw std::runtime_error("Expected uint8 tensor type");
  }

  uint8_t* outputData = static_cast<uint8_t*>(TfLiteTensorData(outputTensor));

  DEBUG_PRINT("DEBUG: Quantization params - output: zp=%d, scale=%f\n",
              outputParams.zero_point, outputParams.scale);

  std::vector<DetectResult> results;

  DEBUG_PRINT("DEBUG: Image dimensions: %dx%d\n", input_img_width,
              input_img_height);

  for (int i = 0; i < numBoxes; ++i) {
    // Use proper dequantization for score
    float score =
        get_dequant_value(outputData, kTfLiteUInt8, i * 7 + 4,
                          outputParams.zero_point, outputParams.scale);
    if (score < boxThresh) {
      continue;
    }

    int classId = outputData[i * 7 + 5];

    // For tensor shape [1, 8400, 4], use sequential indexing per detection
    uint8_t raw_x_1_u8 = outputData[i * 7 + 0];
    uint8_t raw_y_1_u8 = outputData[i * 7 + 1];
    uint8_t raw_x_2_u8 = outputData[i * 7 + 2];
    uint8_t raw_y_2_u8 = outputData[i * 7 + 3];

    uint8_t raw_angle_u8 = outputData[i * 7 + 6];

    // Use proper dequantization for bbox coordinates (like we do for scores)
    float x1 = get_dequant_value(&raw_x_1_u8, kTfLiteUInt8, 0,
                                 outputParams.zero_point, outputParams.scale);
    float y1 = get_dequant_value(&raw_y_1_u8, kTfLiteUInt8, 0,
                                 outputParams.zero_point, outputParams.scale);
    float x2 = get_dequant_value(&raw_x_2_u8, kTfLiteUInt8, 0,
                                 outputParams.zero_point, outputParams.scale);
    float y2 = get_dequant_value(&raw_y_2_u8, kTfLiteUInt8, 0,
                                 outputParams.zero_point, outputParams.scale);

    float angle =
        get_dequant_value(&raw_angle_u8, kTfLiteUInt8, 0,
                          outputParams.zero_point, outputParams.scale);

    float clamped_x1 =
        std::max(0.0f, std::min(x1, static_cast<float>(input_img_width)));
    float clamped_y1 =
        std::max(0.0f, std::min(y1, static_cast<float>(input_img_height)));
    float clamped_x2 =
        std::max(0.0f, std::min(x2, static_cast<float>(input_img_width)));
    float clamped_y2 =
        std::max(0.0f, std::min(y2, static_cast<float>(input_img_height)));

    float angle_degrees = angle * 180.0f / M_PI;

    // Skip bad boxes
    if (clamped_x1 >= clamped_x2 || clamped_y1 >= clamped_y2) {
      continue;
    }

#ifndef NDEBUG
    if (results.size() < 5) {
      std::printf(
          " DEBUG: box %d - uint8 corners: (%d, %d) to (%d, %d), angle=%d\n", i,
          raw_x_1_u8, raw_y_1_u8, raw_x_2_u8, raw_y_2_u8, raw_angle_u8);
      std::printf(
          "DEBUG: box %d - dequantized corners: (%.2f, %.2f) to (%.2f, "
          "%.2f), angle=%.2f\n",
          i, x1, y1, x2, y2, angle);
      std::printf(
          "DEBUG: box %d - clamped corners: (%.2f, %.2f) to (%.2f, %.2f), "
          "score=%.3f, class=%d, angle=%.2f\n",
          i, clamped_x1, clamped_y1, clamped_x2, clamped_y2, score, classId,
          angle_degrees);
    }
#endif

    BoundingBox box{static_cast<int>(std::round(clamped_x1)),
                    static_cast<int>(std::round(clamped_y1)),
                    static_cast<int>(std::round(clamped_x2)),
                    static_cast<int>(std::round(clamped_y2)), angle_degrees};

    DetectResult det{classId, box, score};

    results.push_back(det);
  }

  return results;
}

std::vector<DetectResult> TFLiteDetector::optimized_nms(
    std::vector<DetectResult>& candidates, float nmsThreshold) {
  if (candidates.empty()) return {};

  // Sort by confidence (descending) - single pass
  std::sort(candidates.begin(), candidates.end(),
            [](const DetectResult& a, const DetectResult& b) {
              return a.obj_conf > b.obj_conf;
            });

  std::vector<DetectResult> results;
  results.reserve(candidates.size() / 4);  // Reasonable initial capacity

  // Use bitset for faster suppression tracking
  std::vector<bool> suppressed(candidates.size(), false);

  for (size_t i = 0; i < candidates.size(); ++i) {
    if (suppressed[i]) continue;

    // Keep this detection
    results.push_back(candidates[i]);
    const auto& currentBox = candidates[i];

    // Suppress overlapping boxes of the SAME class only
    // Start from i+1 since array is sorted by confidence
    for (size_t j = i + 1; j < candidates.size(); ++j) {
      if (suppressed[j] || candidates[j].id != currentBox.id) continue;

      if (calculateIoU(currentBox.box, candidates[j].box) > nmsThreshold) {
        suppressed[j] = true;
      }
    }
  }

  return results;
}
