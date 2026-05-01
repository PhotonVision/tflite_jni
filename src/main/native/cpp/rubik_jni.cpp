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

/*
 * NOTE INTELLISENSE WILL NOT WORK UNTIL THE PROJECT IS BUILT AT LEAST ONCE
 */

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include <jni.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <tensorflow/lite/c/c_api.h>
#include <tensorflow/lite/c/c_api_experimental.h>
#include <tensorflow/lite/delegates/external/external_delegate.h>
#include <tensorflow/lite/version.h>

#include "utils.hpp"
#include "yoloPostProc.hpp"

static jclass runtimeExceptionClass = nullptr;

// JNI class reference (this can be global since it's shared)
static jclass detectionResultClass = nullptr;

extern "C" {
JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void* reserved) {
  JNIEnv* env;
  if (vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) != JNI_OK) {
    return JNI_ERR;
  }

  // Cache exception classes
  jclass localRuntimeClass = env->FindClass("java/lang/RuntimeException");
  if (localRuntimeClass) {
    runtimeExceptionClass = (jclass)env->NewGlobalRef(localRuntimeClass);
    env->DeleteLocalRef(localRuntimeClass);
  }

  // Find the detection result class
  jclass localClass =
      env->FindClass("org/photonvision/tflite/TFLiteJNI$TFLiteResult");
  if (!localClass) {
    std::printf(
        "Couldn't find class "
        "org/photonvision/tflite/TFLiteJNI$TFLiteResult!\n");
    return JNI_ERR;
  }

  // Create global reference
  detectionResultClass = (jclass)env->NewGlobalRef(localClass);
  env->DeleteLocalRef(localClass);

  if (!detectionResultClass) {
    std::printf("Couldn't create global reference to class!\n");
    return JNI_ERR;
  }

  return JNI_VERSION_1_6;
}

JNIEXPORT void JNICALL JNI_OnUnload(JavaVM* vm, void* reserved) {
  JNIEnv* env;
  if (vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) == JNI_OK) {
    if (detectionResultClass) {
      env->DeleteGlobalRef(detectionResultClass);
      detectionResultClass = nullptr;
    }
  }
}

static jobject MakeJObject(JNIEnv* env, const DetectResult& result) {
  if (!detectionResultClass) {
    std::printf("ERROR: detectionResultClass is null!\n");
    return nullptr;
  }

  jmethodID constructor =
      env->GetMethodID(detectionResultClass, "<init>", "(IIIIFIF)V");
  if (!constructor) {
    std::printf("ERROR: Could not find constructor for TFLiteResult!\n");
    return nullptr;
  }

  return env->NewObject(detectionResultClass, constructor, result.box.x1,
                        result.box.y1, result.box.x2, result.box.y2,
                        result.obj_conf, result.id, result.box.angle);
}

// Helper function to throw exceptions
void ThrowRuntimeException(JNIEnv* env, const char* message) {
  if (runtimeExceptionClass) {
    env->ThrowNew(runtimeExceptionClass, message);
  }
}

/*
 * Class:     org_photonvision_tflite_TFLiteJNI
 * Method:    create
 * Signature: (Ljava/lang/String;II)J
 */
JNIEXPORT jlong JNICALL
Java_org_photonvision_tflite_TFLiteJNI_create
  (JNIEnv* env, jobject obj, jstring modelPath, jint version, jint source)
{
  const char* model_name = env->GetStringUTFChars(modelPath, nullptr);
  if (model_name == nullptr) {
    ThrowRuntimeException(env, "Failed to retrieve model path");
    return 0;
  }

  // Validate model version
  int version_int = static_cast<int>(version);

  // This should be updated whenever a new model version is added
  if (version_int < ModelVersion::YOLOV8 ||
      version_int > ModelVersion::YOLOV11) {
    ThrowRuntimeException(env, "Invalid model version specified");
    env->ReleaseStringUTFChars(modelPath, model_name);
    return 0;
  }

  ModelVersion model_version = static_cast<ModelVersion>(version_int);

  // Validate source
  int source_int = static_cast<int>(source);

  // This should be updated whenever a new model version is added
  if (source_int < TFLiteSource::RUBIK || source_int > TFLiteSource::CPU) {
    ThrowRuntimeException(env, "Invalid TFLite source specified");
    env->ReleaseStringUTFChars(modelPath, model_name);
    return 0;
  }

  TFLiteSource tflite_source = static_cast<TFLiteSource>(source_int);

  // Load the model
  TfLiteModel* model = TfLiteModelCreateFromFile(model_name);
  if (!model) {
    ThrowRuntimeException(env, "Failed to load model file");
    env->ReleaseStringUTFChars(modelPath, model_name);
    return 0;
  }

  DEBUG_PRINT("INFO: Loaded model file '%s'\n", model_name);

  // Create interpreter options
  TfLiteInterpreterOptions* interpreterOpts = TfLiteInterpreterOptionsCreate();
  if (!interpreterOpts) {
    ThrowRuntimeException(env, "Failed to create interpreter options");
    env->ReleaseStringUTFChars(modelPath, model_name);
    return 0;
  }

  // Create TFLiteDetector object
  TFLiteDetector* detector = new TFLiteDetector;

  if (uses_delegate(tflite_source)) {
    TfLiteDelegate* delegate;
    // Create external delegate options
    // We just have to trust that this creates okay, but conveniently if it
    // fails the check when we insert options will catch it.
    TfLiteExternalDelegateOptions delegateOptsValue =
        TfLiteExternalDelegateOptionsDefault("libQnnTFLiteDelegate.so");

    TfLiteExternalDelegateOptions* delegateOpts = &delegateOptsValue;

    // See
    // https://docs.qualcomm.com/bundle/publicresource/topics/80-70014-54/external-delegate-options-for-qnn-delegate.html
    // for what the various delegate options are
    if (TfLiteExternalDelegateOptionsInsert(delegateOpts, "backend_type",
                                            "htp") != kTfLiteOk) {
      ThrowRuntimeException(env, "Failed to set backend type to htp");
      env->ReleaseStringUTFChars(modelPath, model_name);
      return 0;
    }

    if (TfLiteExternalDelegateOptionsInsert(delegateOpts, "htp_use_conv_hmx",
                                            "1") != kTfLiteOk) {
      ThrowRuntimeException(env, "Failed to enable convolutions");
      env->ReleaseStringUTFChars(modelPath, model_name);
      return 0;
    }

    if (TfLiteExternalDelegateOptionsInsert(
            delegateOpts, "htp_performance_mode", "2") != kTfLiteOk) {
      ThrowRuntimeException(env, "Failed to set htp performance mode");
      env->ReleaseStringUTFChars(modelPath, model_name);
      return 0;
    }

    // Create the delegate
    delegate = TfLiteExternalDelegateCreate(delegateOpts);

    if (!delegate) {
      ThrowRuntimeException(env, "Failed to create external delegate");
      env->ReleaseStringUTFChars(modelPath, model_name);
      return 0;
    } else {
      DEBUG_PRINT("INFO: Created external delegate\n");
    }

    DEBUG_PRINT("INFO: Loaded external delegate\n");

    TfLiteInterpreterOptionsAddDelegate(interpreterOpts, delegate);

    detector->delegate = delegate;
  }

  // Create the interpreter
  TfLiteInterpreter* interpreter =
      TfLiteInterpreterCreate(model, interpreterOpts);
  TfLiteInterpreterOptionsDelete(interpreterOpts);

  if (!interpreter) {
    ThrowRuntimeException(env, "Failed to create interpreter");
    if (uses_delegate(tflite_source)) {
      TfLiteExternalDelegateDelete(detector->delegate);
    }
    env->ReleaseStringUTFChars(modelPath, model_name);
    return 0;
  }

  if (uses_delegate(tflite_source)) {
    // Modify graph with delegate
    if (TfLiteInterpreterModifyGraphWithDelegate(
            interpreter, detector->delegate) != kTfLiteOk) {
      ThrowRuntimeException(env, "Failed to modify graph with delegate");
      TfLiteInterpreterDelete(interpreter);
      TfLiteExternalDelegateDelete(detector->delegate);
      env->ReleaseStringUTFChars(modelPath, model_name);
      return 0;
    } else {
      DEBUG_PRINT("INFO: Modified graph with external delegate\n");
    }
  }

  // Allocate tensors
  if (TfLiteInterpreterAllocateTensors(interpreter) != kTfLiteOk) {
    ThrowRuntimeException(env, "Failed to allocate tensors");
    TfLiteInterpreterDelete(interpreter);
    if (uses_delegate(tflite_source)) {
      TfLiteExternalDelegateDelete(detector->delegate);
    }
    env->ReleaseStringUTFChars(modelPath, model_name);
    return 0;
  }

  env->ReleaseStringUTFChars(modelPath, model_name);

  detector->interpreter = interpreter;
  detector->model = model;
  detector->version = model_version;

  // Convert TFLiteDetector pointer to jlong
  jlong ptr = reinterpret_cast<jlong>(detector);

  DEBUG_PRINT("INFO: TensorFlow Lite initialization completed successfully\n");

  return ptr;
}

/*
 * Class:     org_photonvision_tflite_TFLiteJNI
 * Method:    destroy
 * Signature: (J)V
 */
JNIEXPORT void JNICALL
Java_org_photonvision_tflite_TFLiteJNI_destroy
  (JNIEnv* env, jclass, jlong ptr)
{
  TFLiteDetector* detector = reinterpret_cast<TFLiteDetector*>(ptr);

  if (!detector) {
    ThrowRuntimeException(env, "Invalid TFLiteDetector pointer");
    return;
  }

  // Now safely use the pointers
  if (detector->interpreter) TfLiteInterpreterDelete(detector->interpreter);
  if (detector->delegate) TfLiteExternalDelegateDelete(detector->delegate);
  if (detector->model) TfLiteModelDelete(detector->model);
  // We don't need to delete the version since it's just an enum value not a
  // pointer

  // Delete the TFLiteDetector object
  delete detector;

  DEBUG_PRINT("INFO: Object Detection instance destroyed successfully\n");
}

/*
 * Class:     org_photonvision_tflite_TFLiteJNI
 * Method:    detect
 * Signature: (JJDD)[Ljava/lang/Object;
 */
JNIEXPORT jobjectArray JNICALL
Java_org_photonvision_tflite_TFLiteJNI_detect
  (JNIEnv* env, jobject obj, jlong ptr, jlong input_cvmat_ptr,
   jdouble boxThresh, jdouble nmsThreshold)
{
  TFLiteDetector* detector = reinterpret_cast<TFLiteDetector*>(ptr);

  if (!detector) {
    ThrowRuntimeException(env, "Invalid TFLiteDetector pointer");
    return nullptr;
  }

  if (!detector->interpreter) {
    ThrowRuntimeException(env, "Interpreter not initialized");
    return nullptr;
  }

  TfLiteInterpreter* interpreter = detector->interpreter;
  if (!interpreter) {
    ThrowRuntimeException(env, "Invalid interpreter handle");
    return nullptr;
  }

  TfLiteTensor* input = TfLiteInterpreterGetInputTensor(interpreter, 0);
  int in_w, in_h, in_c;
  if (!tensor_image_dims(input, &in_w, &in_h, &in_c)) {
    ThrowRuntimeException(env, "Invalid input tensor shape");
    return nullptr;
  }

  cv::Mat* input_img = reinterpret_cast<cv::Mat*>(input_cvmat_ptr);
  if (!input_img || input_img->empty() || input_img->cols != in_w ||
      input_img->rows != in_h) {
    ThrowRuntimeException(env, "Invalid input image or mismatched dimensions");
    return nullptr;
  }

  cv::Mat rgb;
  if (input_img->channels() == 3) {
    cv::cvtColor(*input_img, rgb, cv::COLOR_BGR2RGB);
  } else {
    ThrowRuntimeException(env, "Input image must be RGB");
    return nullptr;
  }

  std::memcpy(TfLiteTensorData(input), rgb.data, TfLiteTensorByteSize(input));

// Start timer for benchmark
#ifndef NDEBUG
  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC, &start);  // Start timing
#endif

  if (TfLiteInterpreterInvoke(interpreter) != kTfLiteOk) {
    ThrowRuntimeException(env, "Interpreter invocation failed");
    return nullptr;
  }

#ifndef NDEBUG
  clock_gettime(CLOCK_MONOTONIC, &end);  // End timing

  // Calculate elapsed time in milliseconds
  double elapsed_time = (end.tv_sec - start.tv_sec) * 1000.0 +
                        (end.tv_nsec - start.tv_nsec) / 1000000.0;

  DEBUG_PRINT("INFO: Model execution time: %.2f ms\n", elapsed_time);
#endif

  std::vector<DetectResult> results;

  try {
    switch (detector->version) {
      case ModelVersion::YOLOV8:
      case ModelVersion::YOLOV11:
        results = yoloPostProc(interpreter, boxThresh, nmsThreshold,
                               input_img->cols, input_img->rows);
        break;
      default:
        ThrowRuntimeException(env, "Unsupported YOLO version specified");
        return nullptr;
    }

    jobjectArray jResults =
        env->NewObjectArray(results.size(), detectionResultClass, nullptr);
    for (size_t i = 0; i < results.size(); ++i) {
      jobject jDet = MakeJObject(env, results[i]);
      env->SetObjectArrayElement(jResults, i, jDet);
    }

    return jResults;
  } catch (const std::runtime_error& e) {
    ThrowRuntimeException(env, e.what());
    return nullptr;
  }
}

/*
 * Class:     org_photonvision_tflite_TFLiteJNI
 * Method:    isQuantized
 * Signature: (J)Z
 */
JNIEXPORT jboolean JNICALL
Java_org_photonvision_tflite_TFLiteJNI_isQuantized
  (JNIEnv* env, jobject obj, jlong ptr)
{
  TFLiteDetector* detector = reinterpret_cast<TFLiteDetector*>(ptr);

  if (!detector) {
    ThrowRuntimeException(env, "Invalid TFLiteDetector pointer");
    return JNI_FALSE;
  }

  if (!detector->interpreter) {
    ThrowRuntimeException(env, "Interpreter not initialized");
    return JNI_FALSE;
  }

  TfLiteInterpreter* interpreter = detector->interpreter;

  if (!interpreter) {
    ThrowRuntimeException(env, "Invalid interpreter handle");
    return JNI_FALSE;
  }

  // Check if the input tensor is quantized
  TfLiteTensor* input = TfLiteInterpreterGetInputTensor(interpreter, 0);
  if (!input) {
    ThrowRuntimeException(env, "Failed to get input tensor");
    return JNI_FALSE;
  }

  // Check if the tensor type is kTfLiteUInt8
  TfLiteType tensorType = TfLiteTensorType(input);

  if (tensorType == kTfLiteUInt8) {
    DEBUG_PRINT("INFO: Input tensor is quantized\n");
    return JNI_TRUE;  // The model is quantized
  } else {
    DEBUG_PRINT("INFO: Input tensor is not quantized\n");
    return JNI_FALSE;  // The model is not quantized
  }
}
}  // extern "C"
