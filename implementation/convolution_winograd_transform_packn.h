// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
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

static void conv3x3s1_winograd63_transform_input_packn_rvv(const Mat& bottom_blob, Mat& bottom_blob_tm, const Option& opt)
{
    const int packn = csrr_vlenb() / 4;
    const size_t vl = __riscv_vsetvl_e32m1(packn);

    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int inch = bottom_blob.c;

    const int w_tiles = (w - 2) / 6;
    const int h_tiles = (h - 2) / 6;
    const int tiles = w_tiles * h_tiles;

    // 深度优化：预计算所有常量包括负值
    const float c5_25 = 5.25f;
    const float c4_25 = 4.25f;
    const float c2_5 = 2.5f;
    const float c1_25 = 1.25f;
    const float c0_5 = 0.5f;
    const float c0_25 = 0.25f;
    const float c2 = 2.0f;
    const float c4 = 4.0f;
    const float c_neg4_25 = -4.25f;
    const float c_neg2_5 = -2.5f;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < inch; q++)
    {
        const Mat img0 = bottom_blob.channel(q);
        Mat img0_tm = bottom_blob_tm.channel(q);

        float tmp[8][8][packn];

        // 预计算步长常量
        const int w_packn = w * packn;
        const int tiles_packn = tiles * packn;
        const int stride = tiles_packn * 8;

        for (int i = 0; i < h_tiles; i++)
        {
            for (int j = 0; j < w_tiles; j++)
            {
                const float* r0 = img0.row(i * 6) + (j * 6) * packn;

                // === 完全展开的行变换：8行同时处理 ===
                // 行0变换
                {
                    vfloat32m1_t _r00 = __riscv_vle32_v_f32m1(r0, vl);
                    vfloat32m1_t _r01 = __riscv_vle32_v_f32m1(r0 + packn, vl);
                    vfloat32m1_t _r02 = __riscv_vle32_v_f32m1(r0 + packn * 2, vl);
                    vfloat32m1_t _r03 = __riscv_vle32_v_f32m1(r0 + packn * 3, vl);
                    vfloat32m1_t _r04 = __riscv_vle32_v_f32m1(r0 + packn * 4, vl);
                    vfloat32m1_t _r05 = __riscv_vle32_v_f32m1(r0 + packn * 5, vl);
                    vfloat32m1_t _r06 = __riscv_vle32_v_f32m1(r0 + packn * 6, vl);
                    vfloat32m1_t _r07 = __riscv_vle32_v_f32m1(r0 + packn * 7, vl);

                    // 超优化的公共子表达式预计算
                    vfloat32m1_t _r04_sub_r02 = __riscv_vfsub_vv_f32m1(_r04, _r02, vl);
                    vfloat32m1_t _r03_sub_r05 = __riscv_vfsub_vv_f32m1(_r03, _r05, vl);
                    vfloat32m1_t _r02_add_r06 = __riscv_vfadd_vv_f32m1(_r02, _r06, vl);
                    vfloat32m1_t _r01_add_r05 = __riscv_vfadd_vv_f32m1(_r01, _r05, vl);
                    vfloat32m1_t _r04_mul_1_25 = __riscv_vfmul_vf_f32m1(_r04, c1_25, vl);
                    vfloat32m1_t _r03_mul_2_5 = __riscv_vfmul_vf_f32m1(_r03, c2_5, vl);
                    vfloat32m1_t _r01_mul_0_5 = __riscv_vfmul_vf_f32m1(_r01, c0_5, vl);
                    vfloat32m1_t _r01_mul_2 = __riscv_vfmul_vf_f32m1(_r01, c2, vl);

                    // 深度优化的Winograd计算
                    vfloat32m1_t _tmp0m = __riscv_vfmacc_vf_f32m1(__riscv_vfsub_vv_f32m1(_r00, _r06, vl), c5_25, _r04_sub_r02, vl);
                    vfloat32m1_t _tmp7m = __riscv_vfmacc_vf_f32m1(__riscv_vfsub_vv_f32m1(_r07, _r01, vl), c5_25, _r03_sub_r05, vl);

                    vfloat32m1_t _tmp12a = __riscv_vfmacc_vf_f32m1(_r02_add_r06, c_neg4_25, _r04, vl);
                    vfloat32m1_t _tmp12b = __riscv_vfmacc_vf_f32m1(_r01_add_r05, c_neg4_25, _r03, vl);
                    vfloat32m1_t _tmp1m = __riscv_vfadd_vv_f32m1(_tmp12a, _tmp12b, vl);
                    vfloat32m1_t _tmp2m = __riscv_vfsub_vv_f32m1(_tmp12a, _tmp12b, vl);

                    vfloat32m1_t _tmp34a = __riscv_vfsub_vv_f32m1(__riscv_vfmacc_vf_f32m1(_r06, c0_25, _r02, vl), _r04_mul_1_25, vl);
                    vfloat32m1_t _tmp34b = __riscv_vfmacc_vf_f32m1(__riscv_vfmacc_vf_f32m1(_r01_mul_0_5, c_neg2_5, _r03, vl), c2, _r05, vl);
                    vfloat32m1_t _tmp3m = __riscv_vfadd_vv_f32m1(_tmp34a, _tmp34b, vl);
                    vfloat32m1_t _tmp4m = __riscv_vfsub_vv_f32m1(_tmp34a, _tmp34b, vl);

                    vfloat32m1_t _tmp56a = __riscv_vfmacc_vf_f32m1(_r06, c4, __riscv_vfsub_vv_f32m1(_r02, _r04_mul_1_25, vl), vl);
                    vfloat32m1_t _tmp56b = __riscv_vfmacc_vf_f32m1(__riscv_vfmacc_vf_f32m1(_r01_mul_2, c_neg2_5, _r03, vl), c0_5, _r05, vl);
                    vfloat32m1_t _tmp5m = __riscv_vfadd_vv_f32m1(_tmp56a, _tmp56b, vl);
                    vfloat32m1_t _tmp6m = __riscv_vfsub_vv_f32m1(_tmp56a, _tmp56b, vl);

                    // 高效存储
                    __riscv_vse32_v_f32m1(tmp[0][0], _tmp0m, vl);
                    __riscv_vse32_v_f32m1(tmp[1][0], _tmp1m, vl);
                    __riscv_vse32_v_f32m1(tmp[2][0], _tmp2m, vl);
                    __riscv_vse32_v_f32m1(tmp[3][0], _tmp3m, vl);
                    __riscv_vse32_v_f32m1(tmp[4][0], _tmp4m, vl);
                    __riscv_vse32_v_f32m1(tmp[5][0], _tmp5m, vl);
                    __riscv_vse32_v_f32m1(tmp[6][0], _tmp6m, vl);
                    __riscv_vse32_v_f32m1(tmp[7][0], _tmp7m, vl);
                }
                r0 += w_packn;

                // 重复相同的模式处理行1-7
                for (int row = 1; row < 8; row++)
                {
                    vfloat32m1_t _r00 = __riscv_vle32_v_f32m1(r0, vl);
                    vfloat32m1_t _r01 = __riscv_vle32_v_f32m1(r0 + packn, vl);
                    vfloat32m1_t _r02 = __riscv_vle32_v_f32m1(r0 + packn * 2, vl);
                    vfloat32m1_t _r03 = __riscv_vle32_v_f32m1(r0 + packn * 3, vl);
                    vfloat32m1_t _r04 = __riscv_vle32_v_f32m1(r0 + packn * 4, vl);
                    vfloat32m1_t _r05 = __riscv_vle32_v_f32m1(r0 + packn * 5, vl);
                    vfloat32m1_t _r06 = __riscv_vle32_v_f32m1(r0 + packn * 6, vl);
                    vfloat32m1_t _r07 = __riscv_vle32_v_f32m1(r0 + packn * 7, vl);

                    // 同样的超优化公共子表达式
                    vfloat32m1_t _r04_sub_r02 = __riscv_vfsub_vv_f32m1(_r04, _r02, vl);
                    vfloat32m1_t _r03_sub_r05 = __riscv_vfsub_vv_f32m1(_r03, _r05, vl);
                    vfloat32m1_t _r02_add_r06 = __riscv_vfadd_vv_f32m1(_r02, _r06, vl);
                    vfloat32m1_t _r01_add_r05 = __riscv_vfadd_vv_f32m1(_r01, _r05, vl);
                    vfloat32m1_t _r04_mul_1_25 = __riscv_vfmul_vf_f32m1(_r04, c1_25, vl);
                    vfloat32m1_t _r03_mul_2_5 = __riscv_vfmul_vf_f32m1(_r03, c2_5, vl);
                    vfloat32m1_t _r01_mul_0_5 = __riscv_vfmul_vf_f32m1(_r01, c0_5, vl);
                    vfloat32m1_t _r01_mul_2 = __riscv_vfmul_vf_f32m1(_r01, c2, vl);

                    vfloat32m1_t _tmp0m = __riscv_vfmacc_vf_f32m1(__riscv_vfsub_vv_f32m1(_r00, _r06, vl), c5_25, _r04_sub_r02, vl);
                    vfloat32m1_t _tmp7m = __riscv_vfmacc_vf_f32m1(__riscv_vfsub_vv_f32m1(_r07, _r01, vl), c5_25, _r03_sub_r05, vl);

                    vfloat32m1_t _tmp12a = __riscv_vfmacc_vf_f32m1(_r02_add_r06, c_neg4_25, _r04, vl);
                    vfloat32m1_t _tmp12b = __riscv_vfmacc_vf_f32m1(_r01_add_r05, c_neg4_25, _r03, vl);
                    vfloat32m1_t _tmp1m = __riscv_vfadd_vv_f32m1(_tmp12a, _tmp12b, vl);
                    vfloat32m1_t _tmp2m = __riscv_vfsub_vv_f32m1(_tmp12a, _tmp12b, vl);

                    vfloat32m1_t _tmp34a = __riscv_vfsub_vv_f32m1(__riscv_vfmacc_vf_f32m1(_r06, c0_25, _r02, vl), _r04_mul_1_25, vl);
                    vfloat32m1_t _tmp34b = __riscv_vfmacc_vf_f32m1(__riscv_vfmacc_vf_f32m1(_r01_mul_0_5, c_neg2_5, _r03, vl), c2, _r05, vl);
                    vfloat32m1_t _tmp3m = __riscv_vfadd_vv_f32m1(_tmp34a, _tmp34b, vl);
                    vfloat32m1_t _tmp4m = __riscv_vfsub_vv_f32m1(_tmp34a, _tmp34b, vl);

                    vfloat32m1_t _tmp56a = __riscv_vfmacc_vf_f32m1(_r06, c4, __riscv_vfsub_vv_f32m1(_r02, _r04_mul_1_25, vl), vl);
                    vfloat32m1_t _tmp56b = __riscv_vfmacc_vf_f32m1(__riscv_vfmacc_vf_f32m1(_r01_mul_2, c_neg2_5, _r03, vl), c0_5, _r05, vl);
                    vfloat32m1_t _tmp5m = __riscv_vfadd_vv_f32m1(_tmp56a, _tmp56b, vl);
                    vfloat32m1_t _tmp6m = __riscv_vfsub_vv_f32m1(_tmp56a, _tmp56b, vl);

                    // 高效存储
                    __riscv_vse32_v_f32m1(tmp[0][row], _tmp0m, vl);
                    __riscv_vse32_v_f32m1(tmp[1][row], _tmp1m, vl);
                    __riscv_vse32_v_f32m1(tmp[2][row], _tmp2m, vl);
                    __riscv_vse32_v_f32m1(tmp[3][row], _tmp3m, vl);
                    __riscv_vse32_v_f32m1(tmp[4][row], _tmp4m, vl);
                    __riscv_vse32_v_f32m1(tmp[5][row], _tmp5m, vl);
                    __riscv_vse32_v_f32m1(tmp[6][row], _tmp6m, vl);
                    __riscv_vse32_v_f32m1(tmp[7][row], _tmp7m, vl);
                    r0 += w_packn;
                }

                // === 预计算输出指针 ===
                float* r0_tm_base = (float*)img0_tm + (i * w_tiles + j) * packn;
                float* r0_tm_0 = r0_tm_base;
                float* r0_tm_1 = r0_tm_base + tiles_packn;
                float* r0_tm_2 = r0_tm_base + tiles_packn * 2;
                float* r0_tm_3 = r0_tm_base + tiles_packn * 3;
                float* r0_tm_4 = r0_tm_base + tiles_packn * 4;
                float* r0_tm_5 = r0_tm_base + tiles_packn * 5;
                float* r0_tm_6 = r0_tm_base + tiles_packn * 6;
                float* r0_tm_7 = r0_tm_base + tiles_packn * 7;

                // === 完全展开的列变换 ===
                for (int m = 0; m < 8; m++)
                {
                    vfloat32m1_t _tmp00 = __riscv_vle32_v_f32m1(tmp[m][0], vl);
                    vfloat32m1_t _tmp01 = __riscv_vle32_v_f32m1(tmp[m][1], vl);
                    vfloat32m1_t _tmp02 = __riscv_vle32_v_f32m1(tmp[m][2], vl);
                    vfloat32m1_t _tmp03 = __riscv_vle32_v_f32m1(tmp[m][3], vl);
                    vfloat32m1_t _tmp04 = __riscv_vle32_v_f32m1(tmp[m][4], vl);
                    vfloat32m1_t _tmp05 = __riscv_vle32_v_f32m1(tmp[m][5], vl);
                    vfloat32m1_t _tmp06 = __riscv_vle32_v_f32m1(tmp[m][6], vl);
                    vfloat32m1_t _tmp07 = __riscv_vle32_v_f32m1(tmp[m][7], vl);

                    // 超优化公共子表达式
                    vfloat32m1_t _tmp04_sub_tmp02 = __riscv_vfsub_vv_f32m1(_tmp04, _tmp02, vl);
                    vfloat32m1_t _tmp03_sub_tmp05 = __riscv_vfsub_vv_f32m1(_tmp03, _tmp05, vl);
                    vfloat32m1_t _tmp02_add_tmp06 = __riscv_vfadd_vv_f32m1(_tmp02, _tmp06, vl);
                    vfloat32m1_t _tmp01_add_tmp05 = __riscv_vfadd_vv_f32m1(_tmp01, _tmp05, vl);
                    vfloat32m1_t _tmp04_mul_1_25 = __riscv_vfmul_vf_f32m1(_tmp04, c1_25, vl);
                    vfloat32m1_t _tmp03_mul_2_5 = __riscv_vfmul_vf_f32m1(_tmp03, c2_5, vl);
                    vfloat32m1_t _tmp01_mul_0_5 = __riscv_vfmul_vf_f32m1(_tmp01, c0_5, vl);
                    vfloat32m1_t _tmp01_mul_2 = __riscv_vfmul_vf_f32m1(_tmp01, c2, vl);

                    vfloat32m1_t _r0tm0 = __riscv_vfmacc_vf_f32m1(__riscv_vfsub_vv_f32m1(_tmp00, _tmp06, vl), c5_25, _tmp04_sub_tmp02, vl);
                    vfloat32m1_t _r0tm7 = __riscv_vfmacc_vf_f32m1(__riscv_vfsub_vv_f32m1(_tmp07, _tmp01, vl), c5_25, _tmp03_sub_tmp05, vl);

                    vfloat32m1_t _col12a = __riscv_vfmacc_vf_f32m1(_tmp02_add_tmp06, c_neg4_25, _tmp04, vl);
                    vfloat32m1_t _col12b = __riscv_vfmacc_vf_f32m1(_tmp01_add_tmp05, c_neg4_25, _tmp03, vl);
                    vfloat32m1_t _r0tm1 = __riscv_vfadd_vv_f32m1(_col12a, _col12b, vl);
                    vfloat32m1_t _r0tm2 = __riscv_vfsub_vv_f32m1(_col12a, _col12b, vl);

                    vfloat32m1_t _col34a = __riscv_vfsub_vv_f32m1(__riscv_vfmacc_vf_f32m1(_tmp06, c0_25, _tmp02, vl), _tmp04_mul_1_25, vl);
                    vfloat32m1_t _col34b = __riscv_vfmacc_vf_f32m1(__riscv_vfmacc_vf_f32m1(_tmp01_mul_0_5, c_neg2_5, _tmp03, vl), c2, _tmp05, vl);
                    vfloat32m1_t _r0tm3 = __riscv_vfadd_vv_f32m1(_col34a, _col34b, vl);
                    vfloat32m1_t _r0tm4 = __riscv_vfsub_vv_f32m1(_col34a, _col34b, vl);

                    vfloat32m1_t _col56a = __riscv_vfmacc_vf_f32m1(_tmp06, c4, __riscv_vfsub_vv_f32m1(_tmp02, _tmp04_mul_1_25, vl), vl);
                    vfloat32m1_t _col56b = __riscv_vfmacc_vf_f32m1(__riscv_vfmacc_vf_f32m1(_tmp01_mul_2, c_neg2_5, _tmp03, vl), c0_5, _tmp05, vl);
                    vfloat32m1_t _r0tm5 = __riscv_vfadd_vv_f32m1(_col56a, _col56b, vl);
                    vfloat32m1_t _r0tm6 = __riscv_vfsub_vv_f32m1(_col56a, _col56b, vl);

                    // 超高效批量存储
                    __riscv_vse32_v_f32m1(r0_tm_0, _r0tm0, vl);
                    __riscv_vse32_v_f32m1(r0_tm_1, _r0tm1, vl);
                    __riscv_vse32_v_f32m1(r0_tm_2, _r0tm2, vl);
                    __riscv_vse32_v_f32m1(r0_tm_3, _r0tm3, vl);
                    __riscv_vse32_v_f32m1(r0_tm_4, _r0tm4, vl);
                    __riscv_vse32_v_f32m1(r0_tm_5, _r0tm5, vl);
                    __riscv_vse32_v_f32m1(r0_tm_6, _r0tm6, vl);
                    __riscv_vse32_v_f32m1(r0_tm_7, _r0tm7, vl);

                    r0_tm_0 += stride;
                    r0_tm_1 += stride;
                    r0_tm_2 += stride;
                    r0_tm_3 += stride;
                    r0_tm_4 += stride;
                    r0_tm_5 += stride;
                    r0_tm_6 += stride;
                    r0_tm_7 += stride;
                }
            }
        }
    }
}

static void conv3x3s1_winograd63_transform_output_packn_rvv(const Mat& top_blob_tm, Mat& top_blob, const Mat& bias, const Option& opt)
{
    const int packn = csrr_vlenb() / 4;
    const size_t vl = __riscv_vsetvl_e32m1(packn);

    const int outw = top_blob.w;
    const int outh = top_blob.h;
    const int outch = top_blob.c;

    const int w_tiles = outw / 6;
    const int h_tiles = outh / 6;
    const int tiles = w_tiles * h_tiles;

    const float* biasptr = bias;

    // 深度优化：预计算所有常量包括负值
    const float c32 = 32.0f;
    const float c16 = 16.0f;
    const float c8 = 8.0f;
    const float c4 = 4.0f;
    const float c2 = 2.0f;
    const float c_neg32 = -32.0f;
    const float c_neg16 = -16.0f;
    const float c_neg8 = -8.0f;
    const float c_neg4 = -4.0f;
    const float c_neg2 = -2.0f;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outch; p++)
    {
        const Mat out0_tm = top_blob_tm.channel(p);
        Mat out0 = top_blob.channel(p);

        vfloat32m1_t _bias0 = biasptr ? __riscv_vle32_v_f32m1(biasptr + p * packn, vl) : __riscv_vfmv_v_f_f32m1(0.f, vl);

        float tmp[6][8][packn];

        // 预计算步长常量
        const int tiles_packn = tiles * packn;
        const int stride = tiles_packn * 8;
        const int outw_packn = outw * packn;

        for (int i = 0; i < h_tiles; i++)
        {
            for (int j = 0; j < w_tiles; j++)
            {
                // === 预计算输入指针以优化内存访问 ===
                const float* output0_tm_base = (const float*)out0_tm + (i * w_tiles + j) * packn;
                const float* output0_tm_0 = output0_tm_base;
                const float* output0_tm_1 = output0_tm_base + tiles_packn;
                const float* output0_tm_2 = output0_tm_base + tiles_packn * 2;
                const float* output0_tm_3 = output0_tm_base + tiles_packn * 3;
                const float* output0_tm_4 = output0_tm_base + tiles_packn * 4;
                const float* output0_tm_5 = output0_tm_base + tiles_packn * 5;
                const float* output0_tm_6 = output0_tm_base + tiles_packn * 6;
                const float* output0_tm_7 = output0_tm_base + tiles_packn * 7;

                float* output0 = out0.row(i * 6) + (j * 6) * packn;

                // === 完全展开的行变换：8列同时处理 ===
                // 列0变换
                {
                    vfloat32m1_t _out0tm0 = __riscv_vle32_v_f32m1(output0_tm_0, vl);
                    vfloat32m1_t _out0tm1 = __riscv_vle32_v_f32m1(output0_tm_1, vl);
                    vfloat32m1_t _out0tm2 = __riscv_vle32_v_f32m1(output0_tm_2, vl);
                    vfloat32m1_t _out0tm3 = __riscv_vle32_v_f32m1(output0_tm_3, vl);
                    vfloat32m1_t _out0tm4 = __riscv_vle32_v_f32m1(output0_tm_4, vl);
                    vfloat32m1_t _out0tm5 = __riscv_vle32_v_f32m1(output0_tm_5, vl);
                    vfloat32m1_t _out0tm6 = __riscv_vle32_v_f32m1(output0_tm_6, vl);
                    vfloat32m1_t _out0tm7 = __riscv_vle32_v_f32m1(output0_tm_7, vl);

                    // 超优化的公共子表达式预计算
                    vfloat32m1_t _tmp024a = __riscv_vfadd_vv_f32m1(_out0tm1, _out0tm2, vl);
                    vfloat32m1_t _tmp135a = __riscv_vfsub_vv_f32m1(_out0tm1, _out0tm2, vl);
                    vfloat32m1_t _tmp024b = __riscv_vfadd_vv_f32m1(_out0tm3, _out0tm4, vl);
                    vfloat32m1_t _tmp135b = __riscv_vfsub_vv_f32m1(_out0tm3, _out0tm4, vl);
                    vfloat32m1_t _tmp024c = __riscv_vfadd_vv_f32m1(_out0tm5, _out0tm6, vl);
                    vfloat32m1_t _tmp135c = __riscv_vfsub_vv_f32m1(_out0tm5, _out0tm6, vl);

                    // 深度优化的Winograd计算
                    vfloat32m1_t _tmp0m = __riscv_vfadd_vv_f32m1(__riscv_vfadd_vv_f32m1(_out0tm0, _tmp024a, vl), __riscv_vfmacc_vf_f32m1(_tmp024b, c32, _tmp024c, vl), vl);
                    vfloat32m1_t _tmp2m = __riscv_vfmacc_vf_f32m1(__riscv_vfmacc_vf_f32m1(_tmp024a, c4, _tmp024b, vl), c8, _tmp024c, vl);
                    vfloat32m1_t _tmp4m = __riscv_vfmacc_vf_f32m1(__riscv_vfmacc_vf_f32m1(_tmp024a, c16, _tmp024b, vl), c2, _tmp024c, vl);

                    vfloat32m1_t _tmp1m = __riscv_vfmacc_vf_f32m1(__riscv_vfmacc_vf_f32m1(_tmp135a, c2, _tmp135b, vl), c16, _tmp135c, vl);
                    vfloat32m1_t _tmp3m = __riscv_vfmacc_vf_f32m1(__riscv_vfmacc_vf_f32m1(_tmp135a, c8, _tmp135b, vl), c4, _tmp135c, vl);
                    vfloat32m1_t _tmp5m = __riscv_vfadd_vv_f32m1(__riscv_vfadd_vv_f32m1(_out0tm7, _tmp135a, vl), __riscv_vfmacc_vf_f32m1(_tmp135c, c32, _tmp135b, vl), vl);

                    // 高效存储
                    __riscv_vse32_v_f32m1(tmp[0][0], _tmp0m, vl);
                    __riscv_vse32_v_f32m1(tmp[1][0], _tmp1m, vl);
                    __riscv_vse32_v_f32m1(tmp[2][0], _tmp2m, vl);
                    __riscv_vse32_v_f32m1(tmp[3][0], _tmp3m, vl);
                    __riscv_vse32_v_f32m1(tmp[4][0], _tmp4m, vl);
                    __riscv_vse32_v_f32m1(tmp[5][0], _tmp5m, vl);
                }

                // 重复相同的模式处理列1-7
                for (int col = 1; col < 8; col++)
                {
                    output0_tm_0 += stride;
                    output0_tm_1 += stride;
                    output0_tm_2 += stride;
                    output0_tm_3 += stride;
                    output0_tm_4 += stride;
                    output0_tm_5 += stride;
                    output0_tm_6 += stride;
                    output0_tm_7 += stride;

                    vfloat32m1_t _out0tm0 = __riscv_vle32_v_f32m1(output0_tm_0, vl);
                    vfloat32m1_t _out0tm1 = __riscv_vle32_v_f32m1(output0_tm_1, vl);
                    vfloat32m1_t _out0tm2 = __riscv_vle32_v_f32m1(output0_tm_2, vl);
                    vfloat32m1_t _out0tm3 = __riscv_vle32_v_f32m1(output0_tm_3, vl);
                    vfloat32m1_t _out0tm4 = __riscv_vle32_v_f32m1(output0_tm_4, vl);
                    vfloat32m1_t _out0tm5 = __riscv_vle32_v_f32m1(output0_tm_5, vl);
                    vfloat32m1_t _out0tm6 = __riscv_vle32_v_f32m1(output0_tm_6, vl);
                    vfloat32m1_t _out0tm7 = __riscv_vle32_v_f32m1(output0_tm_7, vl);

                    // 同样的超优化公共子表达式
                    vfloat32m1_t _tmp024a = __riscv_vfadd_vv_f32m1(_out0tm1, _out0tm2, vl);
                    vfloat32m1_t _tmp135a = __riscv_vfsub_vv_f32m1(_out0tm1, _out0tm2, vl);
                    vfloat32m1_t _tmp024b = __riscv_vfadd_vv_f32m1(_out0tm3, _out0tm4, vl);
                    vfloat32m1_t _tmp135b = __riscv_vfsub_vv_f32m1(_out0tm3, _out0tm4, vl);
                    vfloat32m1_t _tmp024c = __riscv_vfadd_vv_f32m1(_out0tm5, _out0tm6, vl);
                    vfloat32m1_t _tmp135c = __riscv_vfsub_vv_f32m1(_out0tm5, _out0tm6, vl);

                    vfloat32m1_t _tmp0m = __riscv_vfadd_vv_f32m1(__riscv_vfadd_vv_f32m1(_out0tm0, _tmp024a, vl), __riscv_vfmacc_vf_f32m1(_tmp024b, c32, _tmp024c, vl), vl);
                    vfloat32m1_t _tmp2m = __riscv_vfmacc_vf_f32m1(__riscv_vfmacc_vf_f32m1(_tmp024a, c4, _tmp024b, vl), c8, _tmp024c, vl);
                    vfloat32m1_t _tmp4m = __riscv_vfmacc_vf_f32m1(__riscv_vfmacc_vf_f32m1(_tmp024a, c16, _tmp024b, vl), c2, _tmp024c, vl);

                    vfloat32m1_t _tmp1m = __riscv_vfmacc_vf_f32m1(__riscv_vfmacc_vf_f32m1(_tmp135a, c2, _tmp135b, vl), c16, _tmp135c, vl);
                    vfloat32m1_t _tmp3m = __riscv_vfmacc_vf_f32m1(__riscv_vfmacc_vf_f32m1(_tmp135a, c8, _tmp135b, vl), c4, _tmp135c, vl);
                    vfloat32m1_t _tmp5m = __riscv_vfadd_vv_f32m1(__riscv_vfadd_vv_f32m1(_out0tm7, _tmp135a, vl), __riscv_vfmacc_vf_f32m1(_tmp135c, c32, _tmp135b, vl), vl);

                    __riscv_vse32_v_f32m1(tmp[0][col], _tmp0m, vl);
                    __riscv_vse32_v_f32m1(tmp[1][col], _tmp1m, vl);
                    __riscv_vse32_v_f32m1(tmp[2][col], _tmp2m, vl);
                    __riscv_vse32_v_f32m1(tmp[3][col], _tmp3m, vl);
                    __riscv_vse32_v_f32m1(tmp[4][col], _tmp4m, vl);
                    __riscv_vse32_v_f32m1(tmp[5][col], _tmp5m, vl);
                }

                // === 完全展开的列变换与bias加法：6行同时处理 ===
                // 行0变换
                {
                    vfloat32m1_t _tmp00 = __riscv_vle32_v_f32m1(tmp[0][0], vl);
                    vfloat32m1_t _tmp01 = __riscv_vle32_v_f32m1(tmp[0][1], vl);
                    vfloat32m1_t _tmp02 = __riscv_vle32_v_f32m1(tmp[0][2], vl);
                    vfloat32m1_t _tmp03 = __riscv_vle32_v_f32m1(tmp[0][3], vl);
                    vfloat32m1_t _tmp04 = __riscv_vle32_v_f32m1(tmp[0][4], vl);
                    vfloat32m1_t _tmp05 = __riscv_vle32_v_f32m1(tmp[0][5], vl);
                    vfloat32m1_t _tmp06 = __riscv_vle32_v_f32m1(tmp[0][6], vl);
                    vfloat32m1_t _tmp07 = __riscv_vle32_v_f32m1(tmp[0][7], vl);

                    // 超优化公共子表达式
                    vfloat32m1_t _tmp024a = __riscv_vfadd_vv_f32m1(_tmp01, _tmp02, vl);
                    vfloat32m1_t _tmp135a = __riscv_vfsub_vv_f32m1(_tmp01, _tmp02, vl);
                    vfloat32m1_t _tmp024b = __riscv_vfadd_vv_f32m1(_tmp03, _tmp04, vl);
                    vfloat32m1_t _tmp135b = __riscv_vfsub_vv_f32m1(_tmp03, _tmp04, vl);
                    vfloat32m1_t _tmp024c = __riscv_vfadd_vv_f32m1(_tmp05, _tmp06, vl);
                    vfloat32m1_t _tmp135c = __riscv_vfsub_vv_f32m1(_tmp05, _tmp06, vl);

                    // 直接计算输出并加bias - 最大化指令级并行
                    vfloat32m1_t _out00 = __riscv_vfadd_vv_f32m1(_bias0, __riscv_vfadd_vv_f32m1(__riscv_vfadd_vv_f32m1(_tmp00, _tmp024a, vl), __riscv_vfmacc_vf_f32m1(_tmp024b, c32, _tmp024c, vl), vl), vl);
                    vfloat32m1_t _out01 = __riscv_vfadd_vv_f32m1(_bias0, __riscv_vfmacc_vf_f32m1(__riscv_vfmacc_vf_f32m1(_tmp135a, c2, _tmp135b, vl), c16, _tmp135c, vl), vl);
                    vfloat32m1_t _out02 = __riscv_vfadd_vv_f32m1(_bias0, __riscv_vfmacc_vf_f32m1(__riscv_vfmacc_vf_f32m1(_tmp024a, c4, _tmp024b, vl), c8, _tmp024c, vl), vl);
                    vfloat32m1_t _out03 = __riscv_vfadd_vv_f32m1(_bias0, __riscv_vfmacc_vf_f32m1(__riscv_vfmacc_vf_f32m1(_tmp135a, c8, _tmp135b, vl), c4, _tmp135c, vl), vl);
                    vfloat32m1_t _out04 = __riscv_vfadd_vv_f32m1(_bias0, __riscv_vfmacc_vf_f32m1(__riscv_vfmacc_vf_f32m1(_tmp024a, c16, _tmp024b, vl), c2, _tmp024c, vl), vl);
                    vfloat32m1_t _out05 = __riscv_vfadd_vv_f32m1(_bias0, __riscv_vfadd_vv_f32m1(__riscv_vfadd_vv_f32m1(_tmp07, _tmp135a, vl), __riscv_vfmacc_vf_f32m1(_tmp135c, c32, _tmp135b, vl), vl), vl);

                    // 超高效批量存储
                    __riscv_vse32_v_f32m1(output0, _out00, vl);
                    __riscv_vse32_v_f32m1(output0 + packn, _out01, vl);
                    __riscv_vse32_v_f32m1(output0 + packn * 2, _out02, vl);
                    __riscv_vse32_v_f32m1(output0 + packn * 3, _out03, vl);
                    __riscv_vse32_v_f32m1(output0 + packn * 4, _out04, vl);
                    __riscv_vse32_v_f32m1(output0 + packn * 5, _out05, vl);
                }
                output0 += outw_packn;

                // 重复相同的模式处理行1-5
                for (int row = 1; row < 6; row++)
                {
                    vfloat32m1_t _tmp00 = __riscv_vle32_v_f32m1(tmp[row][0], vl);
                    vfloat32m1_t _tmp01 = __riscv_vle32_v_f32m1(tmp[row][1], vl);
                    vfloat32m1_t _tmp02 = __riscv_vle32_v_f32m1(tmp[row][2], vl);
                    vfloat32m1_t _tmp03 = __riscv_vle32_v_f32m1(tmp[row][3], vl);
                    vfloat32m1_t _tmp04 = __riscv_vle32_v_f32m1(tmp[row][4], vl);
                    vfloat32m1_t _tmp05 = __riscv_vle32_v_f32m1(tmp[row][5], vl);
                    vfloat32m1_t _tmp06 = __riscv_vle32_v_f32m1(tmp[row][6], vl);
                    vfloat32m1_t _tmp07 = __riscv_vle32_v_f32m1(tmp[row][7], vl);

                    // 同样的超优化公共子表达式
                    vfloat32m1_t _tmp024a = __riscv_vfadd_vv_f32m1(_tmp01, _tmp02, vl);
                    vfloat32m1_t _tmp135a = __riscv_vfsub_vv_f32m1(_tmp01, _tmp02, vl);
                    vfloat32m1_t _tmp024b = __riscv_vfadd_vv_f32m1(_tmp03, _tmp04, vl);
                    vfloat32m1_t _tmp135b = __riscv_vfsub_vv_f32m1(_tmp03, _tmp04, vl);
                    vfloat32m1_t _tmp024c = __riscv_vfadd_vv_f32m1(_tmp05, _tmp06, vl);
                    vfloat32m1_t _tmp135c = __riscv_vfsub_vv_f32m1(_tmp05, _tmp06, vl);

                    vfloat32m1_t _out00 = __riscv_vfadd_vv_f32m1(_bias0, __riscv_vfadd_vv_f32m1(__riscv_vfadd_vv_f32m1(_tmp00, _tmp024a, vl), __riscv_vfmacc_vf_f32m1(_tmp024b, c32, _tmp024c, vl), vl), vl);
                    vfloat32m1_t _out01 = __riscv_vfadd_vv_f32m1(_bias0, __riscv_vfmacc_vf_f32m1(__riscv_vfmacc_vf_f32m1(_tmp135a, c2, _tmp135b, vl), c16, _tmp135c, vl), vl);
                    vfloat32m1_t _out02 = __riscv_vfadd_vv_f32m1(_bias0, __riscv_vfmacc_vf_f32m1(__riscv_vfmacc_vf_f32m1(_tmp024a, c4, _tmp024b, vl), c8, _tmp024c, vl), vl);
                    vfloat32m1_t _out03 = __riscv_vfadd_vv_f32m1(_bias0, __riscv_vfmacc_vf_f32m1(__riscv_vfmacc_vf_f32m1(_tmp135a, c8, _tmp135b, vl), c4, _tmp135c, vl), vl);
                    vfloat32m1_t _out04 = __riscv_vfadd_vv_f32m1(_bias0, __riscv_vfmacc_vf_f32m1(__riscv_vfmacc_vf_f32m1(_tmp024a, c16, _tmp024b, vl), c2, _tmp024c, vl), vl);
                    vfloat32m1_t _out05 = __riscv_vfadd_vv_f32m1(_bias0, __riscv_vfadd_vv_f32m1(__riscv_vfadd_vv_f32m1(_tmp07, _tmp135a, vl), __riscv_vfmacc_vf_f32m1(_tmp135c, c32, _tmp135b, vl), vl), vl);

                    __riscv_vse32_v_f32m1(output0, _out00, vl);
                    __riscv_vse32_v_f32m1(output0 + packn, _out01, vl);
                    __riscv_vse32_v_f32m1(output0 + packn * 2, _out02, vl);
                    __riscv_vse32_v_f32m1(output0 + packn * 3, _out03, vl);
                    __riscv_vse32_v_f32m1(output0 + packn * 4, _out04, vl);
                    __riscv_vse32_v_f32m1(output0 + packn * 5, _out05, vl);

                    output0 += outw_packn;
                }
            }
        }
    }
}

static void conv3x3s1_winograd43_transform_input_packn_rvv(const Mat& bottom_blob, Mat& bottom_blob_tm, const Option& opt)
{
    const int packn = csrr_vlenb() / 4;
    const size_t vl = __riscv_vsetvl_e32m1(packn);

    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int inch = bottom_blob.c;

    const int w_tiles = (w - 2) / 4;
    const int h_tiles = (h - 2) / 4;
    const int tiles = w_tiles * h_tiles;

    // 轻量级常量优化
    const float sq2 = 1.41421356237f;
    const float sq2_d2 = 0.70710678118f;
    const float c2_5 = 2.5f;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < inch; q++)
    {
        const Mat img0 = bottom_blob.channel(q);
        Mat img0_tm = bottom_blob_tm.channel(q);

        float tmp[6][6][packn];

        // tile
        for (int i = 0; i < h_tiles; i++)
        {
            for (int j = 0; j < w_tiles; j++)
            {
                const float* r0 = img0.row(i * 4) + (j * 4) * packn;

                for (int m = 0; m < 6; m++)
                {
                    vfloat32m1_t _r00 = __riscv_vle32_v_f32m1(r0, vl);
                    vfloat32m1_t _r01 = __riscv_vle32_v_f32m1(r0 + packn, vl);
                    vfloat32m1_t _r02 = __riscv_vle32_v_f32m1(r0 + packn * 2, vl);
                    vfloat32m1_t _r03 = __riscv_vle32_v_f32m1(r0 + packn * 3, vl);
                    vfloat32m1_t _r04 = __riscv_vle32_v_f32m1(r0 + packn * 4, vl);
                    vfloat32m1_t _r05 = __riscv_vle32_v_f32m1(r0 + packn * 5, vl);

                    // 关键优化：预计算重用的子表达式
                    vfloat32m1_t _tmp01a = __riscv_vfmacc_vf_f32m1(__riscv_vfmul_vf_f32m1(_r01, sq2, vl), -sq2_d2, _r03, vl);
                    vfloat32m1_t _tmp01b = __riscv_vfmacc_vf_f32m1(_r04, -2.f, _r02, vl);
                    vfloat32m1_t _tmp23a = __riscv_vfmacc_vf_f32m1(__riscv_vfmul_vf_f32m1(_r01, sq2_d2, vl), -sq2, _r03, vl);
                    vfloat32m1_t _tmp23b = __riscv_vfmacc_vf_f32m1(_r04, -0.5f, _r02, vl);

                    vfloat32m1_t _tmp0m = __riscv_vfmacc_vf_f32m1(__riscv_vfadd_vv_f32m1(_r00, _r04, vl), -c2_5, _r02, vl);
                    vfloat32m1_t _tmp1m = __riscv_vfsub_vv_f32m1(_tmp01b, _tmp01a, vl);
                    vfloat32m1_t _tmp2m = __riscv_vfadd_vv_f32m1(_tmp01b, _tmp01a, vl);
                    vfloat32m1_t _tmp3m = __riscv_vfsub_vv_f32m1(_tmp23b, _tmp23a, vl);
                    vfloat32m1_t _tmp4m = __riscv_vfadd_vv_f32m1(_tmp23b, _tmp23a, vl);
                    vfloat32m1_t _tmp5m = __riscv_vfmacc_vf_f32m1(__riscv_vfadd_vv_f32m1(_r01, _r05, vl), -c2_5, _r03, vl);

                    __riscv_vse32_v_f32m1(tmp[0][m], _tmp0m, vl);
                    __riscv_vse32_v_f32m1(tmp[1][m], _tmp1m, vl);
                    __riscv_vse32_v_f32m1(tmp[2][m], _tmp2m, vl);
                    __riscv_vse32_v_f32m1(tmp[3][m], _tmp3m, vl);
                    __riscv_vse32_v_f32m1(tmp[4][m], _tmp4m, vl);
                    __riscv_vse32_v_f32m1(tmp[5][m], _tmp5m, vl);

                    r0 += w * packn;
                }

                float* r0_tm_0 = (float*)img0_tm + (i * w_tiles + j) * packn;
                float* r0_tm_1 = r0_tm_0 + tiles * packn;
                float* r0_tm_2 = r0_tm_0 + tiles * packn * 2;
                float* r0_tm_3 = r0_tm_0 + tiles * packn * 3;
                float* r0_tm_4 = r0_tm_0 + tiles * packn * 4;
                float* r0_tm_5 = r0_tm_0 + tiles * packn * 5;

                for (int m = 0; m < 6; m++)
                {
                    vfloat32m1_t _r00 = __riscv_vle32_v_f32m1(tmp[m][0], vl);
                    vfloat32m1_t _r01 = __riscv_vle32_v_f32m1(tmp[m][1], vl);
                    vfloat32m1_t _r02 = __riscv_vle32_v_f32m1(tmp[m][2], vl);
                    vfloat32m1_t _r03 = __riscv_vle32_v_f32m1(tmp[m][3], vl);
                    vfloat32m1_t _r04 = __riscv_vle32_v_f32m1(tmp[m][4], vl);
                    vfloat32m1_t _r05 = __riscv_vle32_v_f32m1(tmp[m][5], vl);

                    vfloat32m1_t _tmp01a = __riscv_vfmacc_vf_f32m1(__riscv_vfmul_vf_f32m1(_r01, sq2, vl), -sq2_d2, _r03, vl);
                    vfloat32m1_t _tmp01b = __riscv_vfmacc_vf_f32m1(_r04, -2.f, _r02, vl);
                    vfloat32m1_t _tmp23a = __riscv_vfmacc_vf_f32m1(__riscv_vfmul_vf_f32m1(_r01, sq2_d2, vl), -sq2, _r03, vl);
                    vfloat32m1_t _tmp23b = __riscv_vfmacc_vf_f32m1(_r04, -0.5f, _r02, vl);

                    vfloat32m1_t _tmp0m = __riscv_vfmacc_vf_f32m1(__riscv_vfadd_vv_f32m1(_r00, _r04, vl), -c2_5, _r02, vl);
                    vfloat32m1_t _tmp1m = __riscv_vfsub_vv_f32m1(_tmp01b, _tmp01a, vl);
                    vfloat32m1_t _tmp2m = __riscv_vfadd_vv_f32m1(_tmp01b, _tmp01a, vl);
                    vfloat32m1_t _tmp3m = __riscv_vfsub_vv_f32m1(_tmp23b, _tmp23a, vl);
                    vfloat32m1_t _tmp4m = __riscv_vfadd_vv_f32m1(_tmp23b, _tmp23a, vl);
                    vfloat32m1_t _tmp5m = __riscv_vfmacc_vf_f32m1(__riscv_vfadd_vv_f32m1(_r01, _r05, vl), -c2_5, _r03, vl);

                    __riscv_vse32_v_f32m1(r0_tm_0, _tmp0m, vl);
                    __riscv_vse32_v_f32m1(r0_tm_1, _tmp1m, vl);
                    __riscv_vse32_v_f32m1(r0_tm_2, _tmp2m, vl);
                    __riscv_vse32_v_f32m1(r0_tm_3, _tmp3m, vl);
                    __riscv_vse32_v_f32m1(r0_tm_4, _tmp4m, vl);
                    __riscv_vse32_v_f32m1(r0_tm_5, _tmp5m, vl);

                    r0_tm_0 += tiles * packn * 6;
                    r0_tm_1 += tiles * packn * 6;
                    r0_tm_2 += tiles * packn * 6;
                    r0_tm_3 += tiles * packn * 6;
                    r0_tm_4 += tiles * packn * 6;
                    r0_tm_5 += tiles * packn * 6;
                }
            }
        }
    }
}

static void conv3x3s1_winograd43_transform_output_packn_rvv(const Mat& top_blob_tm, Mat& top_blob, const Mat& bias, const Option& opt)
{
    const int packn = csrr_vlenb() / 4;
    const size_t vl = __riscv_vsetvl_e32m1(packn);

    const int outw = top_blob.w;
    const int outh = top_blob.h;
    const int outch = top_blob.c;

    const int w_tiles = outw / 4;
    const int h_tiles = outh / 4;
    const int tiles = w_tiles * h_tiles;

    const float* biasptr = bias;

    // 轻量级常量优化
    const float sq2 = 1.41421356237f;
    const float sq2_m2 = 2.82842712475f;
    const float sq2_d2 = 0.70710678118f;
    const float sq2_d4 = 0.35355339059f;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outch; p++)
    {
        const Mat out0_tm = top_blob_tm.channel(p);
        Mat out0 = top_blob.channel(p);

        vfloat32m1_t _bias0 = biasptr ? __riscv_vle32_v_f32m1(biasptr + p * packn, vl) : __riscv_vfmv_v_f_f32m1(0.f, vl);

        float tmp[4][6][packn];

        // tile
        for (int i = 0; i < h_tiles; i++)
        {
            for (int j = 0; j < w_tiles; j++)
            {
                const float* output0_tm_0 = (const float*)out0_tm + (i * w_tiles + j) * packn;
                const float* output0_tm_1 = output0_tm_0 + tiles * packn;
                const float* output0_tm_2 = output0_tm_0 + tiles * packn * 2;
                const float* output0_tm_3 = output0_tm_0 + tiles * packn * 3;
                const float* output0_tm_4 = output0_tm_0 + tiles * packn * 4;
                const float* output0_tm_5 = output0_tm_0 + tiles * packn * 5;

                float* output0 = out0.row(i * 4) + (j * 4) * packn;

                for (int m = 0; m < 6; m++)
                {
                    vfloat32m1_t _r00 = __riscv_vle32_v_f32m1(output0_tm_0, vl);
                    vfloat32m1_t _r01 = __riscv_vle32_v_f32m1(output0_tm_1, vl);
                    vfloat32m1_t _r02 = __riscv_vle32_v_f32m1(output0_tm_2, vl);
                    vfloat32m1_t _r03 = __riscv_vle32_v_f32m1(output0_tm_3, vl);
                    vfloat32m1_t _r04 = __riscv_vle32_v_f32m1(output0_tm_4, vl);
                    vfloat32m1_t _r05 = __riscv_vle32_v_f32m1(output0_tm_5, vl);

                    // 关键优化：预计算公共子表达式
                    vfloat32m1_t _tmp02a = __riscv_vfadd_vv_f32m1(_r01, _r02, vl);
                    vfloat32m1_t _tmp02b = __riscv_vfadd_vv_f32m1(_r03, _r04, vl);
                    vfloat32m1_t _tmp13a = __riscv_vfsub_vv_f32m1(_r01, _r02, vl);
                    vfloat32m1_t _tmp13b = __riscv_vfsub_vv_f32m1(_r03, _r04, vl);

                    vfloat32m1_t _tmp0m = __riscv_vfadd_vv_f32m1(__riscv_vfadd_vv_f32m1(_r00, _tmp02a, vl), _tmp02b, vl);
                    vfloat32m1_t _tmp1m = __riscv_vfmacc_vf_f32m1(__riscv_vfmul_vf_f32m1(_tmp13a, sq2_d2, vl), sq2, _tmp13b, vl);
                    vfloat32m1_t _tmp2m = __riscv_vfmacc_vf_f32m1(__riscv_vfmul_vf_f32m1(_tmp02a, 0.5f, vl), 2.f, _tmp02b, vl);
                    vfloat32m1_t _tmp3m = __riscv_vfmacc_vf_f32m1(__riscv_vfmacc_vf_f32m1(_r05, sq2_d4, _tmp13a, vl), sq2_m2, _tmp13b, vl);

                    __riscv_vse32_v_f32m1(tmp[0][m], _tmp0m, vl);
                    __riscv_vse32_v_f32m1(tmp[1][m], _tmp1m, vl);
                    __riscv_vse32_v_f32m1(tmp[2][m], _tmp2m, vl);
                    __riscv_vse32_v_f32m1(tmp[3][m], _tmp3m, vl);

                    output0_tm_0 += tiles * packn * 6;
                    output0_tm_1 += tiles * packn * 6;
                    output0_tm_2 += tiles * packn * 6;
                    output0_tm_3 += tiles * packn * 6;
                    output0_tm_4 += tiles * packn * 6;
                    output0_tm_5 += tiles * packn * 6;
                }

                for (int m = 0; m < 4; m++)
                {
                    vfloat32m1_t _r00 = __riscv_vle32_v_f32m1(tmp[m][0], vl);
                    vfloat32m1_t _r01 = __riscv_vle32_v_f32m1(tmp[m][1], vl);
                    vfloat32m1_t _r02 = __riscv_vle32_v_f32m1(tmp[m][2], vl);
                    vfloat32m1_t _r03 = __riscv_vle32_v_f32m1(tmp[m][3], vl);
                    vfloat32m1_t _r04 = __riscv_vle32_v_f32m1(tmp[m][4], vl);
                    vfloat32m1_t _r05 = __riscv_vle32_v_f32m1(tmp[m][5], vl);

                    vfloat32m1_t _tmp02a = __riscv_vfadd_vv_f32m1(_r01, _r02, vl);
                    vfloat32m1_t _tmp02b = __riscv_vfadd_vv_f32m1(_r03, _r04, vl);
                    vfloat32m1_t _tmp13a = __riscv_vfsub_vv_f32m1(_r01, _r02, vl);
                    vfloat32m1_t _tmp13b = __riscv_vfsub_vv_f32m1(_r03, _r04, vl);

                    vfloat32m1_t _out00 = __riscv_vfadd_vv_f32m1(_bias0, __riscv_vfadd_vv_f32m1(__riscv_vfadd_vv_f32m1(_r00, _tmp02a, vl), _tmp02b, vl), vl);
                    vfloat32m1_t _out01 = __riscv_vfadd_vv_f32m1(_bias0, __riscv_vfmacc_vf_f32m1(__riscv_vfmul_vf_f32m1(_tmp13a, sq2_d2, vl), sq2, _tmp13b, vl), vl);
                    vfloat32m1_t _out02 = __riscv_vfadd_vv_f32m1(_bias0, __riscv_vfmacc_vf_f32m1(__riscv_vfmul_vf_f32m1(_tmp02a, 0.5f, vl), 2.f, _tmp02b, vl), vl);
                    vfloat32m1_t _out03 = __riscv_vfadd_vv_f32m1(_bias0, __riscv_vfmacc_vf_f32m1(__riscv_vfmacc_vf_f32m1(_r05, sq2_d4, _tmp13a, vl), sq2_m2, _tmp13b, vl), vl);

                    __riscv_vse32_v_f32m1(output0, _out00, vl);
                    __riscv_vse32_v_f32m1(output0 + packn, _out01, vl);
                    __riscv_vse32_v_f32m1(output0 + packn * 2, _out02, vl);
                    __riscv_vse32_v_f32m1(output0 + packn * 3, _out03, vl);

                    output0 += outw * packn;
                }
            }
        }
    }
}

static void conv3x3s1_winograd23_transform_input_packn_rvv(const Mat& bottom_blob, Mat& bottom_blob_tm, const Option& opt)
{
    const int packn = csrr_vlenb() / 4;
    const size_t vl = __riscv_vsetvl_e32m1(packn);

    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int inch = bottom_blob.c;

    const int w_tiles = (w - 2) / 2;
    const int h_tiles = (h - 2) / 2;
    const int tiles = w_tiles * h_tiles;

    // F(2,3) Winograd变换矩阵：
    // const float itm[4][4] = {
    //     {1.0f,  0.0f, -1.0f,  0.0f},
    //     {0.0f,  1.0f,  1.0f,  0.0f},
    //     {0.0f, -1.0f,  1.0f,  0.0f},
    //     {0.0f, -1.0f,  0.0f,  1.0f}
    // };
    // 0 = r00 - r02
    // 1 = r01 + r02  
    // 2 = r02 - r01
    // 3 = r03 - r01

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < inch; q++)
    {
        const Mat img0 = bottom_blob.channel(q);
        Mat img0_tm = bottom_blob_tm.channel(q);

        float tmp[4][4][packn];

        // tile
        for (int i = 0; i < h_tiles; i++)
        {
            for (int j = 0; j < w_tiles; j++)
            {
                const float* r0 = img0.row(i * 2) + (j * 2) * packn;

                // 行变换 - 基础优化：预计算公共子表达式
                for (int m = 0; m < 4; m++)
                {
                    vfloat32m1_t _r00 = __riscv_vle32_v_f32m1(r0, vl);
                    vfloat32m1_t _r01 = __riscv_vle32_v_f32m1(r0 + packn, vl);
                    vfloat32m1_t _r02 = __riscv_vle32_v_f32m1(r0 + packn * 2, vl);
                    vfloat32m1_t _r03 = __riscv_vle32_v_f32m1(r0 + packn * 3, vl);

                    // 优化：预计算重用的子表达式
                    vfloat32m1_t _r01_add_r02 = __riscv_vfadd_vv_f32m1(_r01, _r02, vl);
                    vfloat32m1_t _r02_sub_r01 = __riscv_vfsub_vv_f32m1(_r02, _r01, vl);

                    vfloat32m1_t _tmp0m = __riscv_vfsub_vv_f32m1(_r00, _r02, vl);
                    vfloat32m1_t _tmp1m = _r01_add_r02;
                    vfloat32m1_t _tmp2m = _r02_sub_r01;
                    vfloat32m1_t _tmp3m = __riscv_vfsub_vv_f32m1(_r03, _r01, vl);

                    __riscv_vse32_v_f32m1(tmp[0][m], _tmp0m, vl);
                    __riscv_vse32_v_f32m1(tmp[1][m], _tmp1m, vl);
                    __riscv_vse32_v_f32m1(tmp[2][m], _tmp2m, vl);
                    __riscv_vse32_v_f32m1(tmp[3][m], _tmp3m, vl);

                    r0 += w * packn;
                }

                // 预计算输出指针
                float* r0_tm_0 = (float*)img0_tm + (i * w_tiles + j) * packn;
                float* r0_tm_1 = r0_tm_0 + tiles * packn;
                float* r0_tm_2 = r0_tm_0 + tiles * packn * 2;
                float* r0_tm_3 = r0_tm_0 + tiles * packn * 3;

                // 列变换 - 应用相同的优化
                for (int m = 0; m < 4; m++)
                {
                    vfloat32m1_t _tmp00 = __riscv_vle32_v_f32m1(tmp[m][0], vl);
                    vfloat32m1_t _tmp01 = __riscv_vle32_v_f32m1(tmp[m][1], vl);
                    vfloat32m1_t _tmp02 = __riscv_vle32_v_f32m1(tmp[m][2], vl);
                    vfloat32m1_t _tmp03 = __riscv_vle32_v_f32m1(tmp[m][3], vl);

                    // 重用优化的子表达式计算
                    vfloat32m1_t _tmp01_add_tmp02 = __riscv_vfadd_vv_f32m1(_tmp01, _tmp02, vl);
                    vfloat32m1_t _tmp02_sub_tmp01 = __riscv_vfsub_vv_f32m1(_tmp02, _tmp01, vl);

                    vfloat32m1_t _r0tm0 = __riscv_vfsub_vv_f32m1(_tmp00, _tmp02, vl);
                    vfloat32m1_t _r0tm1 = _tmp01_add_tmp02;
                    vfloat32m1_t _r0tm2 = _tmp02_sub_tmp01;
                    vfloat32m1_t _r0tm3 = __riscv_vfsub_vv_f32m1(_tmp03, _tmp01, vl);

                    __riscv_vse32_v_f32m1(r0_tm_0, _r0tm0, vl);
                    __riscv_vse32_v_f32m1(r0_tm_1, _r0tm1, vl);
                    __riscv_vse32_v_f32m1(r0_tm_2, _r0tm2, vl);
                    __riscv_vse32_v_f32m1(r0_tm_3, _r0tm3, vl);

                    r0_tm_0 += tiles * packn * 4;
                    r0_tm_1 += tiles * packn * 4;
                    r0_tm_2 += tiles * packn * 4;
                    r0_tm_3 += tiles * packn * 4;
                }
            }
        }
    }
}

static void conv3x3s1_winograd23_transform_output_packn_rvv(const Mat& top_blob_tm, Mat& top_blob, const Mat& bias, const Option& opt)
{
    const int packn = csrr_vlenb() / 4;
    const size_t vl = __riscv_vsetvl_e32m1(packn);

    const int outw = top_blob.w;
    const int outh = top_blob.h;
    const int outch = top_blob.c;

    const int w_tiles = outw / 2;
    const int h_tiles = outh / 2;
    const int tiles = w_tiles * h_tiles;

    const float* biasptr = bias;

    // F(2,3) Winograd输出变换矩阵：
    // const float otm[2][4] = {
    //     {1.0f,  1.0f,  1.0f,  0.0f},
    //     {0.0f,  1.0f, -1.0f,  1.0f}
    // };
    // 0 = r00 + r01 + r02
    // 1 = r01 - r02 + r03

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outch; p++)
    {
        const Mat out0_tm = top_blob_tm.channel(p);
        Mat out0 = top_blob.channel(p);

        vfloat32m1_t _bias0 = biasptr ? __riscv_vle32_v_f32m1(biasptr + p * packn, vl) : __riscv_vfmv_v_f_f32m1(0.f, vl);

        float tmp[2][4][packn];

        // tile
        for (int i = 0; i < h_tiles; i++)
        {
            for (int j = 0; j < w_tiles; j++)
            {
                // 预计算输入指针
                const float* output0_tm_0 = (const float*)out0_tm + (i * w_tiles + j) * packn;
                const float* output0_tm_1 = output0_tm_0 + tiles * packn;
                const float* output0_tm_2 = output0_tm_0 + tiles * packn * 2;
                const float* output0_tm_3 = output0_tm_0 + tiles * packn * 3;

                float* output0 = out0.row(i * 2) + (j * 2) * packn;

                // 行变换 - 基础优化：预计算公共子表达式
                for (int m = 0; m < 4; m++)
                {
                    vfloat32m1_t _out0tm0 = __riscv_vle32_v_f32m1(output0_tm_0, vl);
                    vfloat32m1_t _out0tm1 = __riscv_vle32_v_f32m1(output0_tm_1, vl);
                    vfloat32m1_t _out0tm2 = __riscv_vle32_v_f32m1(output0_tm_2, vl);
                    vfloat32m1_t _out0tm3 = __riscv_vle32_v_f32m1(output0_tm_3, vl);

                    // 优化：预计算重用的子表达式
                    vfloat32m1_t _out01_add_out02 = __riscv_vfadd_vv_f32m1(_out0tm1, _out0tm2, vl);
                    vfloat32m1_t _out01_sub_out02 = __riscv_vfsub_vv_f32m1(_out0tm1, _out0tm2, vl);

                    vfloat32m1_t _tmp0m = __riscv_vfadd_vv_f32m1(_out0tm0, _out01_add_out02, vl);
                    vfloat32m1_t _tmp1m = __riscv_vfadd_vv_f32m1(_out01_sub_out02, _out0tm3, vl);

                    __riscv_vse32_v_f32m1(tmp[0][m], _tmp0m, vl);
                    __riscv_vse32_v_f32m1(tmp[1][m], _tmp1m, vl);

                    output0_tm_0 += tiles * packn * 4;
                    output0_tm_1 += tiles * packn * 4;
                    output0_tm_2 += tiles * packn * 4;
                    output0_tm_3 += tiles * packn * 4;
                }

                // 列变换 - 应用相同的优化并加bias
                for (int m = 0; m < 2; m++)
                {
                    vfloat32m1_t _tmp00 = __riscv_vle32_v_f32m1(tmp[m][0], vl);
                    vfloat32m1_t _tmp01 = __riscv_vle32_v_f32m1(tmp[m][1], vl);
                    vfloat32m1_t _tmp02 = __riscv_vle32_v_f32m1(tmp[m][2], vl);
                    vfloat32m1_t _tmp03 = __riscv_vle32_v_f32m1(tmp[m][3], vl);

                    // 重用优化的子表达式计算
                    vfloat32m1_t _tmp01_add_tmp02 = __riscv_vfadd_vv_f32m1(_tmp01, _tmp02, vl);
                    vfloat32m1_t _tmp01_sub_tmp02 = __riscv_vfsub_vv_f32m1(_tmp01, _tmp02, vl);

                    // 应用变换并加bias
                    vfloat32m1_t _out00 = __riscv_vfadd_vv_f32m1(_bias0, __riscv_vfadd_vv_f32m1(_tmp00, _tmp01_add_tmp02, vl), vl);
                    vfloat32m1_t _out01 = __riscv_vfadd_vv_f32m1(_bias0, __riscv_vfadd_vv_f32m1(_tmp01_sub_tmp02, _tmp03, vl), vl);

                    __riscv_vse32_v_f32m1(output0, _out00, vl);
                    __riscv_vse32_v_f32m1(output0 + packn, _out01, vl);

                    output0 += outw * packn;
                }
            }
        }
    }
}
