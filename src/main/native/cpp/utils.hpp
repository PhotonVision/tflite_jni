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

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <tensorflow/lite/c/c_api.h>

/**
 * Structure representing a bounding box.
 *
 * x1, y1: Top-left corner coordinates.
 * x2, y2: Bottom-right corner coordinates.
 * angle: Rotation angle of the bounding box.
 */
struct BoundingBox {
  int x1;
  int y1;
  int x2;
  int y2;
  double angle;
};

/**
 * Structure representing a detection result.
 *
 * id: Class ID of the detected object.
 * box: Bounding box of the detected object.
 * obj_conf: Confidence score of the detected object.
 */
struct DetectResult {
  int id;
  BoundingBox box;
  float obj_conf;
};

#ifndef NDEBUG
#define DEBUG_PRINT(...) std::printf(__VA_ARGS__)
#else
#define DEBUG_PRINT(...) \
  do {                   \
  } while (0)
#endif

/**
 * Gets the dequantized value from tensor data.
 * @param data Pointer to the tensor data.
 * @param tensor_type The type of the tensor.
 * @param idx The index of the value to retrieve.
 * @param zero_point The zero point for dequantization.
 * @param scale The scale for dequantization.
 * @return The dequantized float value.
 */
float get_dequant_value(void* data, TfLiteType tensor_type, int idx,
                        float zero_point, float scale);

/**
 * Infers the width, height, and channels of a tensor as if it were an image.
 * @param tensor The TensorFlow Lite tensor.
 * @param w Pointer to store the width.
 * @param h Pointer to store the height.
 * @param c Pointer to store the number of channels.
 * @return True if the dimensions were successfully inferred, false otherwise.
 */
bool tensor_image_dims(const TfLiteTensor* tensor, int* w, int* h, int* c);

/**
 * Calculates the Intersection over Union (IoU) between two bounding boxes.
 * Supports both axis-aligned and oriented bounding boxes.
 * @param box1 The first bounding box.
 * @param box2 The second bounding box.
 * @return The IoU value between 0 and 1.
 */
inline float calculateIoU(const BoundingBox& box1, const BoundingBox& box2) {
  // Optimization: If both angles are effectively zero, use faster AABB
  // calculation
  if (std::abs(box1.angle) < 0.1 && std::abs(box2.angle) < 0.1) {
    const int x1 = std::max(box1.x1, box2.x1);
    const int y1 = std::max(box1.y1, box2.y1);
    const int x2 = std::min(box1.x2, box2.x2);
    const int y2 = std::min(box1.y2, box2.y2);

    // No intersection case
    if (x2 <= x1 || y2 <= y1) {
      return 0.0f;
    }

    const float intersectionArea = static_cast<float>((x2 - x1) * (y2 - y1));
    const float area1 =
        static_cast<float>((box1.x2 - box1.x1) * (box1.y2 - box1.y1));
    const float area2 =
        static_cast<float>((box2.x2 - box2.x1) * (box2.y2 - box2.y1));

    return intersectionArea / (area1 + area2 - intersectionArea);
  }

  // OBB IoU using OpenCV
  float w1 = static_cast<float>(box1.x2 - box1.x1);
  float h1 = static_cast<float>(box1.y2 - box1.y1);
  float cx1 = box1.x1 + w1 * 0.5f;
  float cy1 = box1.y1 + h1 * 0.5f;

  float w2 = static_cast<float>(box2.x2 - box2.x1);
  float h2 = static_cast<float>(box2.y2 - box2.y1);
  float cx2 = box2.x1 + w2 * 0.5f;
  float cy2 = box2.y1 + h2 * 0.5f;

  cv::RotatedRect r1(cv::Point2f(cx1, cy1), cv::Size2f(w1, h1),
                     static_cast<float>(box1.angle));
  cv::RotatedRect r2(cv::Point2f(cx2, cy2), cv::Size2f(w2, h2),
                     static_cast<float>(box2.angle));

  std::vector<cv::Point2f> intersectionPoints;
  int res = cv::rotatedRectangleIntersection(r1, r2, intersectionPoints);

  float intersectionArea = 0.0f;
  if (res != cv::INTERSECT_NONE && !intersectionPoints.empty()) {
    intersectionArea = static_cast<float>(cv::contourArea(intersectionPoints));
  }

  float area1 = w1 * h1;
  float area2 = w2 * h2;
  float unionArea = area1 + area2 - intersectionArea;

  if (unionArea <= 1e-5f) return 0.0f;

  return intersectionArea / unionArea;
}
