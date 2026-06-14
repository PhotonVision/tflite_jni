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

#include <cmath>
#include <cstdio>

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
