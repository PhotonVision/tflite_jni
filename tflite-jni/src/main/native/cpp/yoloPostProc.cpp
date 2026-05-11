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

#include "yoloPostProc.hpp"

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <tensorflow/lite/c/c_api.h>
#include <tensorflow/lite/c/c_api_experimental.h>
#include <tensorflow/lite/delegates/external/external_delegate.h>
#include <tensorflow/lite/version.h>

#include "utils.hpp"

std::vector<DetectResult> yoloPostProc(TfLiteInterpreter* interpreter,
                                       double boxThresh, double nmsThreshold,
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

  std::vector<DetectResult> candidateResults;

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
    if (candidateResults.size() < 5) {
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

    candidateResults.push_back(det);
  }

  // NMS
  std::vector<DetectResult> results =
      optimizedNMS(candidateResults, static_cast<float>(nmsThreshold));

  return results;
}
