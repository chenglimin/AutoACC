// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#ifndef LAYER_CONVOLUTION_RISCV_H
#define LAYER_CONVOLUTION_RISCV_H

#include "convolution.h"

// Enable custom conv layer winograd43 optimization
#ifndef NCNN_CUSTOM_CONV_WINOGRAD43_OPT
#define NCNN_CUSTOM_CONV_WINOGRAD43_OPT 1
#endif

namespace ncnn {

class Convolution_riscv : public Convolution
{
public:
    Convolution_riscv();

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

protected:
#if NCNN_ZFH
    int create_pipeline_fp16s(const Option& opt);
    int forward_fp16s(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
    int forward_fp16sa(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
#endif


    // 分离的实现函数
    int forward_original(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
    int forward_yolov8_optimized(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

private:
    // 函数指针，在create_pipeline时设置，运行时零开销
    int (Convolution_riscv::*forward_impl)(const Mat&, Mat&, const Option&) const;

#if NCNN_CUSTOM_CONV_WINOGRAD43_OPT
    // Check if current layer should use forced winograd43 optimization
    bool should_use_forced_winograd43(int w, int h, int num_input, int num_output, int packn) const;
    
    // Flag to track if forced winograd43 was used during pipeline creation
    mutable bool use_forced_winograd43_flag;
#endif

public:
    Layer* activation;

    Mat weight_data_tm;
    Mat weight_winograd23_data;
    Mat weight_winograd43_data;
    Mat weight_winograd63_data;

    // fp16
    Mat bias_data_fp16;
};

} // namespace ncnn

#endif // LAYER_CONVOLUTION_RISCV_H