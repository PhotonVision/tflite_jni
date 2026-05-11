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

#include "utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <tensorflow/lite/c/c_api.h>

float get_dequant_value(void* data, TfLiteType tensor_type, int idx,
                        float zero_point, float scale) {
  switch (tensor_type) {
    case kTfLiteUInt8:
      return (static_cast<uint8_t*>(data)[idx] - zero_point) * scale;
    case kTfLiteFloat32:
      return static_cast<float*>(data)[idx];
    case kTfLiteInt8:
      return (static_cast<int8_t*>(data)[idx] - zero_point) * scale;
    default:
      break;
  }
  return 0.0f;
}

bool tensor_image_dims(const TfLiteTensor* tensor, int* w, int* h, int* c) {
  int n = TfLiteTensorNumDims(tensor);
  int cursor = 0;

  for (int i = 0; i < n; i++) {
    int dim = TfLiteTensorDim(tensor, i);
    if (dim == 0) return false;
    if (dim == 1) continue;

    switch (cursor++) {
      case 0:
        if (w) *w = dim;
        break;
      case 1:
        if (h) *h = dim;
        break;
      case 2:
        if (c) *c = dim;
        break;
      default:
        return false;
    }
  }

  // Ensure that we at least have the width and height.
  if (cursor < 2) return false;
  // If we don't have the number of channels, then assume there's only one.
  if (cursor == 2 && c) *c = 1;
  // Ensure we have no more than 4 image channels.
  if (*c > 4) return false;
  // The tensor dimension appears coherent.
  return true;
}

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

std::vector<DetectResult> optimizedNMS(std::vector<DetectResult>& candidates,
                                       float nmsThreshold) {
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
