// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

// High-Performance SGEMM optimization for M8 shapes and musepi memory hierarchy
// L1 Cache: ~32KB, L2 Cache: ~256KB, L3 Cache: ~2MB (estimated)  
// Cache line: 64 bytes, VLEN: 256bit (8 float32), 32 vector registers
// Key optimizations: 8x unrolling for 16-element, 4x for 8-element, enhanced blocking

// Optimized block sizes for M8 shapes and musepi memory hierarchy
#define SGEMM_OUTCH_BLOCK 16     // Output channel blocking for L3 cache efficiency (increased for large channels)
#define SGEMM_SIZE_BLOCK 64      // Spatial size blocking for L2 cache efficiency  
#define SGEMM_K_BLOCK 256        // K dimension blocking for L1 cache efficiency (doubled for better cache utilization)

static void im2col_sgemm_packn_rvv(const Mat& bottom_im2col, Mat& top_blob, const Mat& kernel, const Mat& _bias, const Option& opt)
{
    const int packn = csrr_vlenb() / 4;
    const size_t vl = __riscv_vsetvl_e32m1(packn);

    // Mat bottom_im2col(size, maxk, inch, 4u * packn, packn, opt.workspace_allocator);

    const int size = bottom_im2col.w;
    const int maxk = bottom_im2col.h;
    const int inch = bottom_im2col.c;

    const int outch = top_blob.c;

    const float* bias = _bias;

    // Stable permute with better data layout - support 16/8/4/2/1 elements
    Mat tmp;
    if (size >= 16)
        tmp.create(16 * maxk, inch, size / 16 + (size % 16) / 8 + (size % 8) / 4 + (size % 4) / 2 + size % 2, 4u * packn, packn, opt.workspace_allocator);
    else if (size >= 8)
        tmp.create(8 * maxk, inch, size / 8 + (size % 8) / 4 + (size % 4) / 2 + size % 2, 4u * packn, packn, opt.workspace_allocator);
    else if (size >= 4)
        tmp.create(4 * maxk, inch, size / 4 + (size % 4) / 2 + size % 2, 4u * packn, packn, opt.workspace_allocator);
    else if (size >= 2)
        tmp.create(2 * maxk, inch, size / 2 + size % 2, 4u * packn, packn, opt.workspace_allocator);
    else
        tmp.create(maxk, inch, size, 4u * packn, packn, opt.workspace_allocator);
    {
        int remain_size_start = 0;
        
        // Process 16 elements at once for improved performance
        int nn_size = size >> 4;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            int i = remain_size_start + ii * 16;

            float* tmpptr = tmp.channel(i / 16);

            for (int q = 0; q < inch; q++)
            {
                const float* img0 = (const float*)bottom_im2col.channel(q) + i * packn;

                for (int k = 0; k < maxk; k++)
                {
#if C906
                    for (int l = 0; l < packn; l++)
                    {
                        for (int n = 0; n < 16; n++)
                        {
                            tmpptr[n] = img0[l + n * packn];
                        }
                        tmpptr += 16;
                    }
                    img0 += size * packn;
#else
                    // Stable vectorized transpose for 16 elements
                    for (int l = 0; l < packn; l++)
                    {
                        // Conservative prefetch
                        if (l == 0) __builtin_prefetch(img0 + size * packn, 0, 1);
                        
                        // Load 16 elements in order for correctness
                        float val0 = img0[l], val1 = img0[l + packn], val2 = img0[l + packn * 2], val3 = img0[l + packn * 3];
                        float val4 = img0[l + packn * 4], val5 = img0[l + packn * 5], val6 = img0[l + packn * 6], val7 = img0[l + packn * 7];
                        float val8 = img0[l + packn * 8], val9 = img0[l + packn * 9], val10 = img0[l + packn * 10], val11 = img0[l + packn * 11];
                        float val12 = img0[l + packn * 12], val13 = img0[l + packn * 13], val14 = img0[l + packn * 14], val15 = img0[l + packn * 15];
                        
                        // Store in correct order
                        tmpptr[0] = val0; tmpptr[1] = val1; tmpptr[2] = val2; tmpptr[3] = val3;
                        tmpptr[4] = val4; tmpptr[5] = val5; tmpptr[6] = val6; tmpptr[7] = val7;
                        tmpptr[8] = val8; tmpptr[9] = val9; tmpptr[10] = val10; tmpptr[11] = val11;
                        tmpptr[12] = val12; tmpptr[13] = val13; tmpptr[14] = val14; tmpptr[15] = val15;
                        tmpptr += 16;
                    }

                    img0 += size * packn;
#endif
                }
            }
        }

        remain_size_start += nn_size << 4;
        nn_size = (size - remain_size_start) >> 3;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            int i = remain_size_start + ii * 8;

            float* tmpptr = tmp.channel(i / 16 + (i % 16) / 8);

            for (int q = 0; q < inch; q++)
            {
                const float* img0 = (const float*)bottom_im2col.channel(q) + i * packn;

                for (int k = 0; k < maxk; k++)
                {
#if C906
                    for (int l = 0; l < packn; l++)
                    {
                        tmpptr[0] = img0[l];
                        tmpptr[1] = img0[l + packn];
                        tmpptr[2] = img0[l + packn * 2];
                        tmpptr[3] = img0[l + packn * 3];
                        tmpptr[4] = img0[l + packn * 4];
                        tmpptr[5] = img0[l + packn * 5];
                        tmpptr[6] = img0[l + packn * 6];
                        tmpptr[7] = img0[l + packn * 7];
                        tmpptr += 8;
                    }

                    img0 += size * packn;
#else
                    vfloat32m1_t _val0 = __riscv_vle32_v_f32m1(img0, vl);
                    vfloat32m1_t _val1 = __riscv_vle32_v_f32m1(img0 + packn, vl);
                    vfloat32m1_t _val2 = __riscv_vle32_v_f32m1(img0 + packn * 2, vl);
                    vfloat32m1_t _val3 = __riscv_vle32_v_f32m1(img0 + packn * 3, vl);
                    vfloat32m1_t _val4 = __riscv_vle32_v_f32m1(img0 + packn * 4, vl);
                    vfloat32m1_t _val5 = __riscv_vle32_v_f32m1(img0 + packn * 5, vl);
                    vfloat32m1_t _val6 = __riscv_vle32_v_f32m1(img0 + packn * 6, vl);
                    vfloat32m1_t _val7 = __riscv_vle32_v_f32m1(img0 + packn * 7, vl);
                    __riscv_vsseg8e32_v_f32m1x8(tmpptr, __riscv_vcreate_v_f32m1x8(_val0, _val1, _val2, _val3, _val4, _val5, _val6, _val7), vl);

                    img0 += size * packn;
                    tmpptr += packn * 8;
#endif
                }
            }
        }

        remain_size_start += nn_size << 3;
        nn_size = (size - remain_size_start) >> 2;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            int i = remain_size_start + ii * 4;

            float* tmpptr = tmp.channel(i / 16 + (i % 16) / 8 + (i % 8) / 4);

            for (int q = 0; q < inch; q++)
            {
                const float* img0 = (const float*)bottom_im2col.channel(q) + i * packn;

                for (int k = 0; k < maxk; k++)
                {
#if C906
                    for (int l = 0; l < packn; l++)
                    {
                        tmpptr[0] = img0[l];
                        tmpptr[1] = img0[l + packn];
                        tmpptr[2] = img0[l + packn * 2];
                        tmpptr[3] = img0[l + packn * 3];
                        tmpptr += 4;
                    }

                    img0 += size * packn;
#else
                    vfloat32m1_t _val0 = __riscv_vle32_v_f32m1(img0, vl);
                    vfloat32m1_t _val1 = __riscv_vle32_v_f32m1(img0 + packn, vl);
                    vfloat32m1_t _val2 = __riscv_vle32_v_f32m1(img0 + packn * 2, vl);
                    vfloat32m1_t _val3 = __riscv_vle32_v_f32m1(img0 + packn * 3, vl);
                    __riscv_vsseg4e32_v_f32m1x4(tmpptr, __riscv_vcreate_v_f32m1x4(_val0, _val1, _val2, _val3), vl);

                    img0 += size * packn;
                    tmpptr += packn * 4;
#endif
                }
            }
        }

        remain_size_start += nn_size << 2;
        nn_size = (size - remain_size_start) >> 1;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int ii = 0; ii < nn_size; ii++)
        {
            int i = remain_size_start + ii * 2;

            float* tmpptr = tmp.channel(i / 16 + (i % 16) / 8 + (i % 8) / 4 + (i % 4) / 2);

            for (int q = 0; q < inch; q++)
            {
                const float* img0 = (const float*)bottom_im2col.channel(q) + i * packn;

                for (int k = 0; k < maxk; k++)
                {
#if C906
                    for (int l = 0; l < packn; l++)
                    {
                        tmpptr[0] = img0[l];
                        tmpptr[1] = img0[l + packn];
                        tmpptr += 2;
                    }

                    img0 += size * packn;
#else
                    vfloat32m1_t _val0 = __riscv_vle32_v_f32m1(img0, vl);
                    vfloat32m1_t _val1 = __riscv_vle32_v_f32m1(img0 + packn, vl);
                    __riscv_vsseg2e32_v_f32m1x2(tmpptr, __riscv_vcreate_v_f32m1x2(_val0, _val1), vl);

                    img0 += size * packn;
                    tmpptr += packn * 2;
#endif
                }
            }
        }

        remain_size_start += nn_size << 1;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = remain_size_start; i < size; i++)
        {
            float* tmpptr = tmp.channel(i / 16 + (i % 16) / 8 + (i % 8) / 4 + (i % 4) / 2 + i % 2);

            for (int q = 0; q < inch; q++)
            {
                const float* img0 = (const float*)bottom_im2col.channel(q) + i * packn;

                for (int k = 0; k < maxk; k++)
                {
                    vfloat32m1_t _val = __riscv_vle32_v_f32m1(img0, vl);
                    __riscv_vse32_v_f32m1(tmpptr, _val, vl);

                    img0 += size * packn;
                    tmpptr += packn;
                }
            }
        }
    }

    // High-performance memory hierarchy-aware blocked SGEMM computation
    // Aggressive optimizations for M8 shapes while maintaining correctness
    const int nn_total = inch * maxk * packn;
    
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int p = 0; p < outch; p++)
    {
        float* outptr0 = top_blob.channel(p);

        int i = 0;
        // Process 16 elements with stable optimizations
        for (; i + 15 < size; i += 16)
        {
            const float* tmpptr = tmp.channel(i / 16);
            const float* kptr0 = kernel.channel(p);

            int nn = nn_total;

            vfloat32m1_t _sum0 = __riscv_vfmv_v_f_f32m1(0.f, vl);
            vfloat32m1_t _sum1 = __riscv_vfmv_v_f_f32m1(0.f, vl);
            vfloat32m1_t _sum2 = __riscv_vfmv_v_f_f32m1(0.f, vl);
            vfloat32m1_t _sum3 = __riscv_vfmv_v_f_f32m1(0.f, vl);
            vfloat32m1_t _sum4 = __riscv_vfmv_v_f_f32m1(0.f, vl);
            vfloat32m1_t _sum5 = __riscv_vfmv_v_f_f32m1(0.f, vl);
            vfloat32m1_t _sum6 = __riscv_vfmv_v_f_f32m1(0.f, vl);
            vfloat32m1_t _sum7 = __riscv_vfmv_v_f_f32m1(0.f, vl);
            vfloat32m1_t _sum8 = __riscv_vfmv_v_f_f32m1(0.f, vl);
            vfloat32m1_t _sum9 = __riscv_vfmv_v_f_f32m1(0.f, vl);
            vfloat32m1_t _sum10 = __riscv_vfmv_v_f_f32m1(0.f, vl);
            vfloat32m1_t _sum11 = __riscv_vfmv_v_f_f32m1(0.f, vl);
            vfloat32m1_t _sum12 = __riscv_vfmv_v_f_f32m1(0.f, vl);
            vfloat32m1_t _sum13 = __riscv_vfmv_v_f_f32m1(0.f, vl);
            vfloat32m1_t _sum14 = __riscv_vfmv_v_f_f32m1(0.f, vl);
            vfloat32m1_t _sum15 = __riscv_vfmv_v_f_f32m1(0.f, vl);

            if (bias)
            {
                vfloat32m1_t _bias = __riscv_vle32_v_f32m1(bias + p * packn, vl);
                _sum0 = _bias; _sum1 = _bias; _sum2 = _bias; _sum3 = _bias;
                _sum4 = _bias; _sum5 = _bias; _sum6 = _bias; _sum7 = _bias;
                _sum8 = _bias; _sum9 = _bias; _sum10 = _bias; _sum11 = _bias;
                _sum12 = _bias; _sum13 = _bias; _sum14 = _bias; _sum15 = _bias;
            }

            // Enhanced L1 cache blocking with larger K blocks for better utilization
            int k_block_start = 0;
            while (k_block_start < nn)
            {
                int k_block_size = (nn - k_block_start > SGEMM_K_BLOCK) ? SGEMM_K_BLOCK : (nn - k_block_start);
                int k_block_end = k_block_start + k_block_size;
                
                const float* tmpptr_block = tmpptr + k_block_start * 16;
                const float* kptr0_block = kptr0 + k_block_start * packn;
                
                // Aggressive 8x unrolling for high performance with M8 large shapes
                int j = k_block_start;
                for (; j + 7 < k_block_end; j += 8)
                {
                    // Multi-level prefetch strategy for better memory hierarchy utilization
                    __builtin_prefetch(tmpptr_block + 128, 0, 3);   // L1 cache
                    __builtin_prefetch(tmpptr_block + 256, 0, 2);   // L2 cache
                    __builtin_prefetch(kptr0_block + packn * 8, 0, 3);
                    __builtin_prefetch(kptr0_block + packn * 16, 0, 2);

                    // Load input values for 8 iterations - better instruction level parallelism
                    float val0_0 = tmpptr_block[0], val1_0 = tmpptr_block[1], val2_0 = tmpptr_block[2], val3_0 = tmpptr_block[3];
                    float val4_0 = tmpptr_block[4], val5_0 = tmpptr_block[5], val6_0 = tmpptr_block[6], val7_0 = tmpptr_block[7];
                    float val8_0 = tmpptr_block[8], val9_0 = tmpptr_block[9], val10_0 = tmpptr_block[10], val11_0 = tmpptr_block[11];
                    float val12_0 = tmpptr_block[12], val13_0 = tmpptr_block[13], val14_0 = tmpptr_block[14], val15_0 = tmpptr_block[15];

                    float val0_1 = tmpptr_block[16], val1_1 = tmpptr_block[17], val2_1 = tmpptr_block[18], val3_1 = tmpptr_block[19];
                    float val4_1 = tmpptr_block[20], val5_1 = tmpptr_block[21], val6_1 = tmpptr_block[22], val7_1 = tmpptr_block[23];
                    float val8_1 = tmpptr_block[24], val9_1 = tmpptr_block[25], val10_1 = tmpptr_block[26], val11_1 = tmpptr_block[27];
                    float val12_1 = tmpptr_block[28], val13_1 = tmpptr_block[29], val14_1 = tmpptr_block[30], val15_1 = tmpptr_block[31];

                    float val0_2 = tmpptr_block[32], val1_2 = tmpptr_block[33], val2_2 = tmpptr_block[34], val3_2 = tmpptr_block[35];
                    float val4_2 = tmpptr_block[36], val5_2 = tmpptr_block[37], val6_2 = tmpptr_block[38], val7_2 = tmpptr_block[39];
                    float val8_2 = tmpptr_block[40], val9_2 = tmpptr_block[41], val10_2 = tmpptr_block[42], val11_2 = tmpptr_block[43];
                    float val12_2 = tmpptr_block[44], val13_2 = tmpptr_block[45], val14_2 = tmpptr_block[46], val15_2 = tmpptr_block[47];

                    float val0_3 = tmpptr_block[48], val1_3 = tmpptr_block[49], val2_3 = tmpptr_block[50], val3_3 = tmpptr_block[51];
                    float val4_3 = tmpptr_block[52], val5_3 = tmpptr_block[53], val6_3 = tmpptr_block[54], val7_3 = tmpptr_block[55];
                    float val8_3 = tmpptr_block[56], val9_3 = tmpptr_block[57], val10_3 = tmpptr_block[58], val11_3 = tmpptr_block[59];
                    float val12_3 = tmpptr_block[60], val13_3 = tmpptr_block[61], val14_3 = tmpptr_block[62], val15_3 = tmpptr_block[63];

                    float val0_4 = tmpptr_block[64], val1_4 = tmpptr_block[65], val2_4 = tmpptr_block[66], val3_4 = tmpptr_block[67];
                    float val4_4 = tmpptr_block[68], val5_4 = tmpptr_block[69], val6_4 = tmpptr_block[70], val7_4 = tmpptr_block[71];
                    float val8_4 = tmpptr_block[72], val9_4 = tmpptr_block[73], val10_4 = tmpptr_block[74], val11_4 = tmpptr_block[75];
                    float val12_4 = tmpptr_block[76], val13_4 = tmpptr_block[77], val14_4 = tmpptr_block[78], val15_4 = tmpptr_block[79];

                    float val0_5 = tmpptr_block[80], val1_5 = tmpptr_block[81], val2_5 = tmpptr_block[82], val3_5 = tmpptr_block[83];
                    float val4_5 = tmpptr_block[84], val5_5 = tmpptr_block[85], val6_5 = tmpptr_block[86], val7_5 = tmpptr_block[87];
                    float val8_5 = tmpptr_block[88], val9_5 = tmpptr_block[89], val10_5 = tmpptr_block[90], val11_5 = tmpptr_block[91];
                    float val12_5 = tmpptr_block[92], val13_5 = tmpptr_block[93], val14_5 = tmpptr_block[94], val15_5 = tmpptr_block[95];

                    float val0_6 = tmpptr_block[96], val1_6 = tmpptr_block[97], val2_6 = tmpptr_block[98], val3_6 = tmpptr_block[99];
                    float val4_6 = tmpptr_block[100], val5_6 = tmpptr_block[101], val6_6 = tmpptr_block[102], val7_6 = tmpptr_block[103];
                    float val8_6 = tmpptr_block[104], val9_6 = tmpptr_block[105], val10_6 = tmpptr_block[106], val11_6 = tmpptr_block[107];
                    float val12_6 = tmpptr_block[108], val13_6 = tmpptr_block[109], val14_6 = tmpptr_block[110], val15_6 = tmpptr_block[111];

                    float val0_7 = tmpptr_block[112], val1_7 = tmpptr_block[113], val2_7 = tmpptr_block[114], val3_7 = tmpptr_block[115];
                    float val4_7 = tmpptr_block[116], val5_7 = tmpptr_block[117], val6_7 = tmpptr_block[118], val7_7 = tmpptr_block[119];
                    float val8_7 = tmpptr_block[120], val9_7 = tmpptr_block[121], val10_7 = tmpptr_block[122], val11_7 = tmpptr_block[123];
                    float val12_7 = tmpptr_block[124], val13_7 = tmpptr_block[125], val14_7 = tmpptr_block[126], val15_7 = tmpptr_block[127];
                    
                    // Load 8 weight vectors
                    vfloat32m1_t _w0 = __riscv_vle32_v_f32m1(kptr0_block, vl);
                    vfloat32m1_t _w1 = __riscv_vle32_v_f32m1(kptr0_block + packn, vl);
                    vfloat32m1_t _w2 = __riscv_vle32_v_f32m1(kptr0_block + packn * 2, vl);
                    vfloat32m1_t _w3 = __riscv_vle32_v_f32m1(kptr0_block + packn * 3, vl);
                    vfloat32m1_t _w4 = __riscv_vle32_v_f32m1(kptr0_block + packn * 4, vl);
                    vfloat32m1_t _w5 = __riscv_vle32_v_f32m1(kptr0_block + packn * 5, vl);
                    vfloat32m1_t _w6 = __riscv_vle32_v_f32m1(kptr0_block + packn * 6, vl);
                    vfloat32m1_t _w7 = __riscv_vle32_v_f32m1(kptr0_block + packn * 7, vl);

                    // FMA operations - all 8 weights (optimized instruction ordering for better ILP)
                    _sum0 = __riscv_vfmacc_vf_f32m1(_sum0, val0_0, _w0, vl);
                    _sum1 = __riscv_vfmacc_vf_f32m1(_sum1, val1_0, _w0, vl);
                    _sum2 = __riscv_vfmacc_vf_f32m1(_sum2, val2_0, _w0, vl);
                    _sum3 = __riscv_vfmacc_vf_f32m1(_sum3, val3_0, _w0, vl);
                    _sum4 = __riscv_vfmacc_vf_f32m1(_sum4, val4_0, _w0, vl);
                    _sum5 = __riscv_vfmacc_vf_f32m1(_sum5, val5_0, _w0, vl);
                    _sum6 = __riscv_vfmacc_vf_f32m1(_sum6, val6_0, _w0, vl);
                    _sum7 = __riscv_vfmacc_vf_f32m1(_sum7, val7_0, _w0, vl);
                    _sum8 = __riscv_vfmacc_vf_f32m1(_sum8, val8_0, _w0, vl);
                    _sum9 = __riscv_vfmacc_vf_f32m1(_sum9, val9_0, _w0, vl);
                    _sum10 = __riscv_vfmacc_vf_f32m1(_sum10, val10_0, _w0, vl);
                    _sum11 = __riscv_vfmacc_vf_f32m1(_sum11, val11_0, _w0, vl);
                    _sum12 = __riscv_vfmacc_vf_f32m1(_sum12, val12_0, _w0, vl);
                    _sum13 = __riscv_vfmacc_vf_f32m1(_sum13, val13_0, _w0, vl);
                    _sum14 = __riscv_vfmacc_vf_f32m1(_sum14, val14_0, _w0, vl);
                    _sum15 = __riscv_vfmacc_vf_f32m1(_sum15, val15_0, _w0, vl);

                    _sum0 = __riscv_vfmacc_vf_f32m1(_sum0, val0_1, _w1, vl);
                    _sum1 = __riscv_vfmacc_vf_f32m1(_sum1, val1_1, _w1, vl);
                    _sum2 = __riscv_vfmacc_vf_f32m1(_sum2, val2_1, _w1, vl);
                    _sum3 = __riscv_vfmacc_vf_f32m1(_sum3, val3_1, _w1, vl);
                    _sum4 = __riscv_vfmacc_vf_f32m1(_sum4, val4_1, _w1, vl);
                    _sum5 = __riscv_vfmacc_vf_f32m1(_sum5, val5_1, _w1, vl);
                    _sum6 = __riscv_vfmacc_vf_f32m1(_sum6, val6_1, _w1, vl);
                    _sum7 = __riscv_vfmacc_vf_f32m1(_sum7, val7_1, _w1, vl);
                    _sum8 = __riscv_vfmacc_vf_f32m1(_sum8, val8_1, _w1, vl);
                    _sum9 = __riscv_vfmacc_vf_f32m1(_sum9, val9_1, _w1, vl);
                    _sum10 = __riscv_vfmacc_vf_f32m1(_sum10, val10_1, _w1, vl);
                    _sum11 = __riscv_vfmacc_vf_f32m1(_sum11, val11_1, _w1, vl);
                    _sum12 = __riscv_vfmacc_vf_f32m1(_sum12, val12_1, _w1, vl);
                    _sum13 = __riscv_vfmacc_vf_f32m1(_sum13, val13_1, _w1, vl);
                    _sum14 = __riscv_vfmacc_vf_f32m1(_sum14, val14_1, _w1, vl);
                    _sum15 = __riscv_vfmacc_vf_f32m1(_sum15, val15_1, _w1, vl);

                    _sum0 = __riscv_vfmacc_vf_f32m1(_sum0, val0_2, _w2, vl);
                    _sum1 = __riscv_vfmacc_vf_f32m1(_sum1, val1_2, _w2, vl);
                    _sum2 = __riscv_vfmacc_vf_f32m1(_sum2, val2_2, _w2, vl);
                    _sum3 = __riscv_vfmacc_vf_f32m1(_sum3, val3_2, _w2, vl);
                    _sum4 = __riscv_vfmacc_vf_f32m1(_sum4, val4_2, _w2, vl);
                    _sum5 = __riscv_vfmacc_vf_f32m1(_sum5, val5_2, _w2, vl);
                    _sum6 = __riscv_vfmacc_vf_f32m1(_sum6, val6_2, _w2, vl);
                    _sum7 = __riscv_vfmacc_vf_f32m1(_sum7, val7_2, _w2, vl);
                    _sum8 = __riscv_vfmacc_vf_f32m1(_sum8, val8_2, _w2, vl);
                    _sum9 = __riscv_vfmacc_vf_f32m1(_sum9, val9_2, _w2, vl);
                    _sum10 = __riscv_vfmacc_vf_f32m1(_sum10, val10_2, _w2, vl);
                    _sum11 = __riscv_vfmacc_vf_f32m1(_sum11, val11_2, _w2, vl);
                    _sum12 = __riscv_vfmacc_vf_f32m1(_sum12, val12_2, _w2, vl);
                    _sum13 = __riscv_vfmacc_vf_f32m1(_sum13, val13_2, _w2, vl);
                    _sum14 = __riscv_vfmacc_vf_f32m1(_sum14, val14_2, _w2, vl);
                    _sum15 = __riscv_vfmacc_vf_f32m1(_sum15, val15_2, _w2, vl);

                    _sum0 = __riscv_vfmacc_vf_f32m1(_sum0, val0_3, _w3, vl);
                    _sum1 = __riscv_vfmacc_vf_f32m1(_sum1, val1_3, _w3, vl);
                    _sum2 = __riscv_vfmacc_vf_f32m1(_sum2, val2_3, _w3, vl);
                    _sum3 = __riscv_vfmacc_vf_f32m1(_sum3, val3_3, _w3, vl);
                    _sum4 = __riscv_vfmacc_vf_f32m1(_sum4, val4_3, _w3, vl);
                    _sum5 = __riscv_vfmacc_vf_f32m1(_sum5, val5_3, _w3, vl);
                    _sum6 = __riscv_vfmacc_vf_f32m1(_sum6, val6_3, _w3, vl);
                    _sum7 = __riscv_vfmacc_vf_f32m1(_sum7, val7_3, _w3, vl);
                    _sum8 = __riscv_vfmacc_vf_f32m1(_sum8, val8_3, _w3, vl);
                    _sum9 = __riscv_vfmacc_vf_f32m1(_sum9, val9_3, _w3, vl);
                    _sum10 = __riscv_vfmacc_vf_f32m1(_sum10, val10_3, _w3, vl);
                    _sum11 = __riscv_vfmacc_vf_f32m1(_sum11, val11_3, _w3, vl);
                    _sum12 = __riscv_vfmacc_vf_f32m1(_sum12, val12_3, _w3, vl);
                    _sum13 = __riscv_vfmacc_vf_f32m1(_sum13, val13_3, _w3, vl);
                    _sum14 = __riscv_vfmacc_vf_f32m1(_sum14, val14_3, _w3, vl);
                    _sum15 = __riscv_vfmacc_vf_f32m1(_sum15, val15_3, _w3, vl);

                    _sum0 = __riscv_vfmacc_vf_f32m1(_sum0, val0_4, _w4, vl);
                    _sum1 = __riscv_vfmacc_vf_f32m1(_sum1, val1_4, _w4, vl);
                    _sum2 = __riscv_vfmacc_vf_f32m1(_sum2, val2_4, _w4, vl);
                    _sum3 = __riscv_vfmacc_vf_f32m1(_sum3, val3_4, _w4, vl);
                    _sum4 = __riscv_vfmacc_vf_f32m1(_sum4, val4_4, _w4, vl);
                    _sum5 = __riscv_vfmacc_vf_f32m1(_sum5, val5_4, _w4, vl);
                    _sum6 = __riscv_vfmacc_vf_f32m1(_sum6, val6_4, _w4, vl);
                    _sum7 = __riscv_vfmacc_vf_f32m1(_sum7, val7_4, _w4, vl);
                    _sum8 = __riscv_vfmacc_vf_f32m1(_sum8, val8_4, _w4, vl);
                    _sum9 = __riscv_vfmacc_vf_f32m1(_sum9, val9_4, _w4, vl);
                    _sum10 = __riscv_vfmacc_vf_f32m1(_sum10, val10_4, _w4, vl);
                    _sum11 = __riscv_vfmacc_vf_f32m1(_sum11, val11_4, _w4, vl);
                    _sum12 = __riscv_vfmacc_vf_f32m1(_sum12, val12_4, _w4, vl);
                    _sum13 = __riscv_vfmacc_vf_f32m1(_sum13, val13_4, _w4, vl);
                    _sum14 = __riscv_vfmacc_vf_f32m1(_sum14, val14_4, _w4, vl);
                    _sum15 = __riscv_vfmacc_vf_f32m1(_sum15, val15_4, _w4, vl);

                    _sum0 = __riscv_vfmacc_vf_f32m1(_sum0, val0_5, _w5, vl);
                    _sum1 = __riscv_vfmacc_vf_f32m1(_sum1, val1_5, _w5, vl);
                    _sum2 = __riscv_vfmacc_vf_f32m1(_sum2, val2_5, _w5, vl);
                    _sum3 = __riscv_vfmacc_vf_f32m1(_sum3, val3_5, _w5, vl);
                    _sum4 = __riscv_vfmacc_vf_f32m1(_sum4, val4_5, _w5, vl);
                    _sum5 = __riscv_vfmacc_vf_f32m1(_sum5, val5_5, _w5, vl);
                    _sum6 = __riscv_vfmacc_vf_f32m1(_sum6, val6_5, _w5, vl);
                    _sum7 = __riscv_vfmacc_vf_f32m1(_sum7, val7_5, _w5, vl);
                    _sum8 = __riscv_vfmacc_vf_f32m1(_sum8, val8_5, _w5, vl);
                    _sum9 = __riscv_vfmacc_vf_f32m1(_sum9, val9_5, _w5, vl);
                    _sum10 = __riscv_vfmacc_vf_f32m1(_sum10, val10_5, _w5, vl);
                    _sum11 = __riscv_vfmacc_vf_f32m1(_sum11, val11_5, _w5, vl);
                    _sum12 = __riscv_vfmacc_vf_f32m1(_sum12, val12_5, _w5, vl);
                    _sum13 = __riscv_vfmacc_vf_f32m1(_sum13, val13_5, _w5, vl);
                    _sum14 = __riscv_vfmacc_vf_f32m1(_sum14, val14_5, _w5, vl);
                    _sum15 = __riscv_vfmacc_vf_f32m1(_sum15, val15_5, _w5, vl);

                    _sum0 = __riscv_vfmacc_vf_f32m1(_sum0, val0_6, _w6, vl);
                    _sum1 = __riscv_vfmacc_vf_f32m1(_sum1, val1_6, _w6, vl);
                    _sum2 = __riscv_vfmacc_vf_f32m1(_sum2, val2_6, _w6, vl);
                    _sum3 = __riscv_vfmacc_vf_f32m1(_sum3, val3_6, _w6, vl);
                    _sum4 = __riscv_vfmacc_vf_f32m1(_sum4, val4_6, _w6, vl);
                    _sum5 = __riscv_vfmacc_vf_f32m1(_sum5, val5_6, _w6, vl);
                    _sum6 = __riscv_vfmacc_vf_f32m1(_sum6, val6_6, _w6, vl);
                    _sum7 = __riscv_vfmacc_vf_f32m1(_sum7, val7_6, _w6, vl);
                    _sum8 = __riscv_vfmacc_vf_f32m1(_sum8, val8_6, _w6, vl);
                    _sum9 = __riscv_vfmacc_vf_f32m1(_sum9, val9_6, _w6, vl);
                    _sum10 = __riscv_vfmacc_vf_f32m1(_sum10, val10_6, _w6, vl);
                    _sum11 = __riscv_vfmacc_vf_f32m1(_sum11, val11_6, _w6, vl);
                    _sum12 = __riscv_vfmacc_vf_f32m1(_sum12, val12_6, _w6, vl);
                    _sum13 = __riscv_vfmacc_vf_f32m1(_sum13, val13_6, _w6, vl);
                    _sum14 = __riscv_vfmacc_vf_f32m1(_sum14, val14_6, _w6, vl);
                    _sum15 = __riscv_vfmacc_vf_f32m1(_sum15, val15_6, _w6, vl);

                    _sum0 = __riscv_vfmacc_vf_f32m1(_sum0, val0_7, _w7, vl);
                    _sum1 = __riscv_vfmacc_vf_f32m1(_sum1, val1_7, _w7, vl);
                    _sum2 = __riscv_vfmacc_vf_f32m1(_sum2, val2_7, _w7, vl);
                    _sum3 = __riscv_vfmacc_vf_f32m1(_sum3, val3_7, _w7, vl);
                    _sum4 = __riscv_vfmacc_vf_f32m1(_sum4, val4_7, _w7, vl);
                    _sum5 = __riscv_vfmacc_vf_f32m1(_sum5, val5_7, _w7, vl);
                    _sum6 = __riscv_vfmacc_vf_f32m1(_sum6, val6_7, _w7, vl);
                    _sum7 = __riscv_vfmacc_vf_f32m1(_sum7, val7_7, _w7, vl);
                    _sum8 = __riscv_vfmacc_vf_f32m1(_sum8, val8_7, _w7, vl);
                    _sum9 = __riscv_vfmacc_vf_f32m1(_sum9, val9_7, _w7, vl);
                    _sum10 = __riscv_vfmacc_vf_f32m1(_sum10, val10_7, _w7, vl);
                    _sum11 = __riscv_vfmacc_vf_f32m1(_sum11, val11_7, _w7, vl);
                    _sum12 = __riscv_vfmacc_vf_f32m1(_sum12, val12_7, _w7, vl);
                    _sum13 = __riscv_vfmacc_vf_f32m1(_sum13, val13_7, _w7, vl);
                    _sum14 = __riscv_vfmacc_vf_f32m1(_sum14, val14_7, _w7, vl);
                    _sum15 = __riscv_vfmacc_vf_f32m1(_sum15, val15_7, _w7, vl);

                    tmpptr_block += 128;
                    kptr0_block += packn * 8;
                }
                
                // Fallback 4x unrolling for remaining elements
                for (; j + 3 < k_block_end; j += 4)
                {
                    // Standard prefetch
                    __builtin_prefetch(tmpptr_block + 64, 0, 1);
                    __builtin_prefetch(kptr0_block + packn * 4, 0, 1);

                    // Load input values for 4 iterations
                    float val0_0 = tmpptr_block[0], val1_0 = tmpptr_block[1], val2_0 = tmpptr_block[2], val3_0 = tmpptr_block[3];
                    float val4_0 = tmpptr_block[4], val5_0 = tmpptr_block[5], val6_0 = tmpptr_block[6], val7_0 = tmpptr_block[7];
                    float val8_0 = tmpptr_block[8], val9_0 = tmpptr_block[9], val10_0 = tmpptr_block[10], val11_0 = tmpptr_block[11];
                    float val12_0 = tmpptr_block[12], val13_0 = tmpptr_block[13], val14_0 = tmpptr_block[14], val15_0 = tmpptr_block[15];

                    float val0_1 = tmpptr_block[16], val1_1 = tmpptr_block[17], val2_1 = tmpptr_block[18], val3_1 = tmpptr_block[19];
                    float val4_1 = tmpptr_block[20], val5_1 = tmpptr_block[21], val6_1 = tmpptr_block[22], val7_1 = tmpptr_block[23];
                    float val8_1 = tmpptr_block[24], val9_1 = tmpptr_block[25], val10_1 = tmpptr_block[26], val11_1 = tmpptr_block[27];
                    float val12_1 = tmpptr_block[28], val13_1 = tmpptr_block[29], val14_1 = tmpptr_block[30], val15_1 = tmpptr_block[31];

                    float val0_2 = tmpptr_block[32], val1_2 = tmpptr_block[33], val2_2 = tmpptr_block[34], val3_2 = tmpptr_block[35];
                    float val4_2 = tmpptr_block[36], val5_2 = tmpptr_block[37], val6_2 = tmpptr_block[38], val7_2 = tmpptr_block[39];
                    float val8_2 = tmpptr_block[40], val9_2 = tmpptr_block[41], val10_2 = tmpptr_block[42], val11_2 = tmpptr_block[43];
                    float val12_2 = tmpptr_block[44], val13_2 = tmpptr_block[45], val14_2 = tmpptr_block[46], val15_2 = tmpptr_block[47];

                    float val0_3 = tmpptr_block[48], val1_3 = tmpptr_block[49], val2_3 = tmpptr_block[50], val3_3 = tmpptr_block[51];
                    float val4_3 = tmpptr_block[52], val5_3 = tmpptr_block[53], val6_3 = tmpptr_block[54], val7_3 = tmpptr_block[55];
                    float val8_3 = tmpptr_block[56], val9_3 = tmpptr_block[57], val10_3 = tmpptr_block[58], val11_3 = tmpptr_block[59];
                    float val12_3 = tmpptr_block[60], val13_3 = tmpptr_block[61], val14_3 = tmpptr_block[62], val15_3 = tmpptr_block[63];
                    
                    // Load 4 weight vectors
                    vfloat32m1_t _w0 = __riscv_vle32_v_f32m1(kptr0_block, vl);
                    vfloat32m1_t _w1 = __riscv_vle32_v_f32m1(kptr0_block + packn, vl);
                    vfloat32m1_t _w2 = __riscv_vle32_v_f32m1(kptr0_block + packn * 2, vl);
                    vfloat32m1_t _w3 = __riscv_vle32_v_f32m1(kptr0_block + packn * 3, vl);

                    // FMA operations - first weight
                    _sum0 = __riscv_vfmacc_vf_f32m1(_sum0, val0_0, _w0, vl);
                    _sum1 = __riscv_vfmacc_vf_f32m1(_sum1, val1_0, _w0, vl);
                    _sum2 = __riscv_vfmacc_vf_f32m1(_sum2, val2_0, _w0, vl);
                    _sum3 = __riscv_vfmacc_vf_f32m1(_sum3, val3_0, _w0, vl);
                    _sum4 = __riscv_vfmacc_vf_f32m1(_sum4, val4_0, _w0, vl);
                    _sum5 = __riscv_vfmacc_vf_f32m1(_sum5, val5_0, _w0, vl);
                    _sum6 = __riscv_vfmacc_vf_f32m1(_sum6, val6_0, _w0, vl);
                    _sum7 = __riscv_vfmacc_vf_f32m1(_sum7, val7_0, _w0, vl);
                    _sum8 = __riscv_vfmacc_vf_f32m1(_sum8, val8_0, _w0, vl);
                    _sum9 = __riscv_vfmacc_vf_f32m1(_sum9, val9_0, _w0, vl);
                    _sum10 = __riscv_vfmacc_vf_f32m1(_sum10, val10_0, _w0, vl);
                    _sum11 = __riscv_vfmacc_vf_f32m1(_sum11, val11_0, _w0, vl);
                    _sum12 = __riscv_vfmacc_vf_f32m1(_sum12, val12_0, _w0, vl);
                    _sum13 = __riscv_vfmacc_vf_f32m1(_sum13, val13_0, _w0, vl);
                    _sum14 = __riscv_vfmacc_vf_f32m1(_sum14, val14_0, _w0, vl);
                    _sum15 = __riscv_vfmacc_vf_f32m1(_sum15, val15_0, _w0, vl);

                    // FMA operations - second weight
                    _sum0 = __riscv_vfmacc_vf_f32m1(_sum0, val0_1, _w1, vl);
                    _sum1 = __riscv_vfmacc_vf_f32m1(_sum1, val1_1, _w1, vl);
                    _sum2 = __riscv_vfmacc_vf_f32m1(_sum2, val2_1, _w1, vl);
                    _sum3 = __riscv_vfmacc_vf_f32m1(_sum3, val3_1, _w1, vl);
                    _sum4 = __riscv_vfmacc_vf_f32m1(_sum4, val4_1, _w1, vl);
                    _sum5 = __riscv_vfmacc_vf_f32m1(_sum5, val5_1, _w1, vl);
                    _sum6 = __riscv_vfmacc_vf_f32m1(_sum6, val6_1, _w1, vl);
                    _sum7 = __riscv_vfmacc_vf_f32m1(_sum7, val7_1, _w1, vl);
                    _sum8 = __riscv_vfmacc_vf_f32m1(_sum8, val8_1, _w1, vl);
                    _sum9 = __riscv_vfmacc_vf_f32m1(_sum9, val9_1, _w1, vl);
                    _sum10 = __riscv_vfmacc_vf_f32m1(_sum10, val10_1, _w1, vl);
                    _sum11 = __riscv_vfmacc_vf_f32m1(_sum11, val11_1, _w1, vl);
                    _sum12 = __riscv_vfmacc_vf_f32m1(_sum12, val12_1, _w1, vl);
                    _sum13 = __riscv_vfmacc_vf_f32m1(_sum13, val13_1, _w1, vl);
                    _sum14 = __riscv_vfmacc_vf_f32m1(_sum14, val14_1, _w1, vl);
                    _sum15 = __riscv_vfmacc_vf_f32m1(_sum15, val15_1, _w1, vl);

                    // FMA operations - third weight
                    _sum0 = __riscv_vfmacc_vf_f32m1(_sum0, val0_2, _w2, vl);
                    _sum1 = __riscv_vfmacc_vf_f32m1(_sum1, val1_2, _w2, vl);
                    _sum2 = __riscv_vfmacc_vf_f32m1(_sum2, val2_2, _w2, vl);
                    _sum3 = __riscv_vfmacc_vf_f32m1(_sum3, val3_2, _w2, vl);
                    _sum4 = __riscv_vfmacc_vf_f32m1(_sum4, val4_2, _w2, vl);
                    _sum5 = __riscv_vfmacc_vf_f32m1(_sum5, val5_2, _w2, vl);
                    _sum6 = __riscv_vfmacc_vf_f32m1(_sum6, val6_2, _w2, vl);
                    _sum7 = __riscv_vfmacc_vf_f32m1(_sum7, val7_2, _w2, vl);
                    _sum8 = __riscv_vfmacc_vf_f32m1(_sum8, val8_2, _w2, vl);
                    _sum9 = __riscv_vfmacc_vf_f32m1(_sum9, val9_2, _w2, vl);
                    _sum10 = __riscv_vfmacc_vf_f32m1(_sum10, val10_2, _w2, vl);
                    _sum11 = __riscv_vfmacc_vf_f32m1(_sum11, val11_2, _w2, vl);
                    _sum12 = __riscv_vfmacc_vf_f32m1(_sum12, val12_2, _w2, vl);
                    _sum13 = __riscv_vfmacc_vf_f32m1(_sum13, val13_2, _w2, vl);
                    _sum14 = __riscv_vfmacc_vf_f32m1(_sum14, val14_2, _w2, vl);
                    _sum15 = __riscv_vfmacc_vf_f32m1(_sum15, val15_2, _w2, vl);

                    // FMA operations - fourth weight
                    _sum0 = __riscv_vfmacc_vf_f32m1(_sum0, val0_3, _w3, vl);
                    _sum1 = __riscv_vfmacc_vf_f32m1(_sum1, val1_3, _w3, vl);
                    _sum2 = __riscv_vfmacc_vf_f32m1(_sum2, val2_3, _w3, vl);
                    _sum3 = __riscv_vfmacc_vf_f32m1(_sum3, val3_3, _w3, vl);
                    _sum4 = __riscv_vfmacc_vf_f32m1(_sum4, val4_3, _w3, vl);
                    _sum5 = __riscv_vfmacc_vf_f32m1(_sum5, val5_3, _w3, vl);
                    _sum6 = __riscv_vfmacc_vf_f32m1(_sum6, val6_3, _w3, vl);
                    _sum7 = __riscv_vfmacc_vf_f32m1(_sum7, val7_3, _w3, vl);
                    _sum8 = __riscv_vfmacc_vf_f32m1(_sum8, val8_3, _w3, vl);
                    _sum9 = __riscv_vfmacc_vf_f32m1(_sum9, val9_3, _w3, vl);
                    _sum10 = __riscv_vfmacc_vf_f32m1(_sum10, val10_3, _w3, vl);
                    _sum11 = __riscv_vfmacc_vf_f32m1(_sum11, val11_3, _w3, vl);
                    _sum12 = __riscv_vfmacc_vf_f32m1(_sum12, val12_3, _w3, vl);
                    _sum13 = __riscv_vfmacc_vf_f32m1(_sum13, val13_3, _w3, vl);
                    _sum14 = __riscv_vfmacc_vf_f32m1(_sum14, val14_3, _w3, vl);
                    _sum15 = __riscv_vfmacc_vf_f32m1(_sum15, val15_3, _w3, vl);

                    tmpptr_block += 64;
                    kptr0_block += packn * 4;
                }
                
                // Handle remaining elements in this K block
                for (; j < k_block_end; j++)
                {
                    float val0 = tmpptr_block[0], val1 = tmpptr_block[1], val2 = tmpptr_block[2], val3 = tmpptr_block[3];
                    float val4 = tmpptr_block[4], val5 = tmpptr_block[5], val6 = tmpptr_block[6], val7 = tmpptr_block[7];
                    float val8 = tmpptr_block[8], val9 = tmpptr_block[9], val10 = tmpptr_block[10], val11 = tmpptr_block[11];
                    float val12 = tmpptr_block[12], val13 = tmpptr_block[13], val14 = tmpptr_block[14], val15 = tmpptr_block[15];
                    
                    vfloat32m1_t _w0 = __riscv_vle32_v_f32m1(kptr0_block, vl);
                    _sum0 = __riscv_vfmacc_vf_f32m1(_sum0, val0, _w0, vl);
                    _sum1 = __riscv_vfmacc_vf_f32m1(_sum1, val1, _w0, vl);
                    _sum2 = __riscv_vfmacc_vf_f32m1(_sum2, val2, _w0, vl);
                    _sum3 = __riscv_vfmacc_vf_f32m1(_sum3, val3, _w0, vl);
                    _sum4 = __riscv_vfmacc_vf_f32m1(_sum4, val4, _w0, vl);
                    _sum5 = __riscv_vfmacc_vf_f32m1(_sum5, val5, _w0, vl);
                    _sum6 = __riscv_vfmacc_vf_f32m1(_sum6, val6, _w0, vl);
                    _sum7 = __riscv_vfmacc_vf_f32m1(_sum7, val7, _w0, vl);
                    _sum8 = __riscv_vfmacc_vf_f32m1(_sum8, val8, _w0, vl);
                    _sum9 = __riscv_vfmacc_vf_f32m1(_sum9, val9, _w0, vl);
                    _sum10 = __riscv_vfmacc_vf_f32m1(_sum10, val10, _w0, vl);
                    _sum11 = __riscv_vfmacc_vf_f32m1(_sum11, val11, _w0, vl);
                    _sum12 = __riscv_vfmacc_vf_f32m1(_sum12, val12, _w0, vl);
                    _sum13 = __riscv_vfmacc_vf_f32m1(_sum13, val13, _w0, vl);
                    _sum14 = __riscv_vfmacc_vf_f32m1(_sum14, val14, _w0, vl);
                    _sum15 = __riscv_vfmacc_vf_f32m1(_sum15, val15, _w0, vl);

                    tmpptr_block += 16;
                    kptr0_block += packn;
                }
                
                k_block_start = k_block_end;
            }

            // Store results
            __riscv_vse32_v_f32m1(outptr0, _sum0, vl);
            __riscv_vse32_v_f32m1(outptr0 + packn, _sum1, vl);
            __riscv_vse32_v_f32m1(outptr0 + packn * 2, _sum2, vl);
            __riscv_vse32_v_f32m1(outptr0 + packn * 3, _sum3, vl);
            __riscv_vse32_v_f32m1(outptr0 + packn * 4, _sum4, vl);
            __riscv_vse32_v_f32m1(outptr0 + packn * 5, _sum5, vl);
            __riscv_vse32_v_f32m1(outptr0 + packn * 6, _sum6, vl);
            __riscv_vse32_v_f32m1(outptr0 + packn * 7, _sum7, vl);
            __riscv_vse32_v_f32m1(outptr0 + packn * 8, _sum8, vl);
            __riscv_vse32_v_f32m1(outptr0 + packn * 9, _sum9, vl);
            __riscv_vse32_v_f32m1(outptr0 + packn * 10, _sum10, vl);
            __riscv_vse32_v_f32m1(outptr0 + packn * 11, _sum11, vl);
            __riscv_vse32_v_f32m1(outptr0 + packn * 12, _sum12, vl);
            __riscv_vse32_v_f32m1(outptr0 + packn * 13, _sum13, vl);
            __riscv_vse32_v_f32m1(outptr0 + packn * 14, _sum14, vl);
            __riscv_vse32_v_f32m1(outptr0 + packn * 15, _sum15, vl);

            outptr0 += packn * 16;
        }
        for (; i + 7 < size; i += 8)
        {
            const float* tmpptr = tmp.channel(i / 16 + (i % 16) / 8);
            const float* kptr0 = kernel.channel(p);

            int nn = nn_total;

            vfloat32m1_t _sum0 = __riscv_vfmv_v_f_f32m1(0.f, vl);
            vfloat32m1_t _sum1 = __riscv_vfmv_v_f_f32m1(0.f, vl);
            vfloat32m1_t _sum2 = __riscv_vfmv_v_f_f32m1(0.f, vl);
            vfloat32m1_t _sum3 = __riscv_vfmv_v_f_f32m1(0.f, vl);
            vfloat32m1_t _sum4 = __riscv_vfmv_v_f_f32m1(0.f, vl);
            vfloat32m1_t _sum5 = __riscv_vfmv_v_f_f32m1(0.f, vl);
            vfloat32m1_t _sum6 = __riscv_vfmv_v_f_f32m1(0.f, vl);
            vfloat32m1_t _sum7 = __riscv_vfmv_v_f_f32m1(0.f, vl);

            if (bias)
            {
                vfloat32m1_t _bias = __riscv_vle32_v_f32m1(bias + p * packn, vl);
                _sum0 = _bias; _sum1 = _bias; _sum2 = _bias; _sum3 = _bias;
                _sum4 = _bias; _sum5 = _bias; _sum6 = _bias; _sum7 = _bias;
            }

            // Enhanced 4x unrolling for 8-tile processing 
            int j = 0;
            for (; j + 3 < nn; j += 4)
            {
                // Improved prefetch for 8-element processing
                __builtin_prefetch(tmpptr + 32, 0, 1);
                __builtin_prefetch(kptr0 + packn * 4, 0, 1);

                // Load input values for 4 iterations
                float val0_0 = tmpptr[0], val1_0 = tmpptr[1], val2_0 = tmpptr[2], val3_0 = tmpptr[3];
                float val4_0 = tmpptr[4], val5_0 = tmpptr[5], val6_0 = tmpptr[6], val7_0 = tmpptr[7];
                float val0_1 = tmpptr[8], val1_1 = tmpptr[9], val2_1 = tmpptr[10], val3_1 = tmpptr[11];
                float val4_1 = tmpptr[12], val5_1 = tmpptr[13], val6_1 = tmpptr[14], val7_1 = tmpptr[15];
                float val0_2 = tmpptr[16], val1_2 = tmpptr[17], val2_2 = tmpptr[18], val3_2 = tmpptr[19];
                float val4_2 = tmpptr[20], val5_2 = tmpptr[21], val6_2 = tmpptr[22], val7_2 = tmpptr[23];
                float val0_3 = tmpptr[24], val1_3 = tmpptr[25], val2_3 = tmpptr[26], val3_3 = tmpptr[27];
                float val4_3 = tmpptr[28], val5_3 = tmpptr[29], val6_3 = tmpptr[30], val7_3 = tmpptr[31];
                
                // Load 4 weight vectors
                vfloat32m1_t _w0 = __riscv_vle32_v_f32m1(kptr0, vl);
                vfloat32m1_t _w1 = __riscv_vle32_v_f32m1(kptr0 + packn, vl);
                vfloat32m1_t _w2 = __riscv_vle32_v_f32m1(kptr0 + packn * 2, vl);
                vfloat32m1_t _w3 = __riscv_vle32_v_f32m1(kptr0 + packn * 3, vl);

                // FMA operations for all 8 accumulators and 4 weights
                _sum0 = __riscv_vfmacc_vf_f32m1(_sum0, val0_0, _w0, vl);
                _sum1 = __riscv_vfmacc_vf_f32m1(_sum1, val1_0, _w0, vl);
                _sum2 = __riscv_vfmacc_vf_f32m1(_sum2, val2_0, _w0, vl);
                _sum3 = __riscv_vfmacc_vf_f32m1(_sum3, val3_0, _w0, vl);
                _sum4 = __riscv_vfmacc_vf_f32m1(_sum4, val4_0, _w0, vl);
                _sum5 = __riscv_vfmacc_vf_f32m1(_sum5, val5_0, _w0, vl);
                _sum6 = __riscv_vfmacc_vf_f32m1(_sum6, val6_0, _w0, vl);
                _sum7 = __riscv_vfmacc_vf_f32m1(_sum7, val7_0, _w0, vl);

                _sum0 = __riscv_vfmacc_vf_f32m1(_sum0, val0_1, _w1, vl);
                _sum1 = __riscv_vfmacc_vf_f32m1(_sum1, val1_1, _w1, vl);
                _sum2 = __riscv_vfmacc_vf_f32m1(_sum2, val2_1, _w1, vl);
                _sum3 = __riscv_vfmacc_vf_f32m1(_sum3, val3_1, _w1, vl);
                _sum4 = __riscv_vfmacc_vf_f32m1(_sum4, val4_1, _w1, vl);
                _sum5 = __riscv_vfmacc_vf_f32m1(_sum5, val5_1, _w1, vl);
                _sum6 = __riscv_vfmacc_vf_f32m1(_sum6, val6_1, _w1, vl);
                _sum7 = __riscv_vfmacc_vf_f32m1(_sum7, val7_1, _w1, vl);

                _sum0 = __riscv_vfmacc_vf_f32m1(_sum0, val0_2, _w2, vl);
                _sum1 = __riscv_vfmacc_vf_f32m1(_sum1, val1_2, _w2, vl);
                _sum2 = __riscv_vfmacc_vf_f32m1(_sum2, val2_2, _w2, vl);
                _sum3 = __riscv_vfmacc_vf_f32m1(_sum3, val3_2, _w2, vl);
                _sum4 = __riscv_vfmacc_vf_f32m1(_sum4, val4_2, _w2, vl);
                _sum5 = __riscv_vfmacc_vf_f32m1(_sum5, val5_2, _w2, vl);
                _sum6 = __riscv_vfmacc_vf_f32m1(_sum6, val6_2, _w2, vl);
                _sum7 = __riscv_vfmacc_vf_f32m1(_sum7, val7_2, _w2, vl);

                _sum0 = __riscv_vfmacc_vf_f32m1(_sum0, val0_3, _w3, vl);
                _sum1 = __riscv_vfmacc_vf_f32m1(_sum1, val1_3, _w3, vl);
                _sum2 = __riscv_vfmacc_vf_f32m1(_sum2, val2_3, _w3, vl);
                _sum3 = __riscv_vfmacc_vf_f32m1(_sum3, val3_3, _w3, vl);
                _sum4 = __riscv_vfmacc_vf_f32m1(_sum4, val4_3, _w3, vl);
                _sum5 = __riscv_vfmacc_vf_f32m1(_sum5, val5_3, _w3, vl);
                _sum6 = __riscv_vfmacc_vf_f32m1(_sum6, val6_3, _w3, vl);
                _sum7 = __riscv_vfmacc_vf_f32m1(_sum7, val7_3, _w3, vl);

                tmpptr += 32;
                kptr0 += packn * 4;
            }

            // Fallback 2x unrolling for remaining elements  
            for (; j + 1 < nn; j += 2)
            {
                float val0_0 = tmpptr[0], val1_0 = tmpptr[1], val2_0 = tmpptr[2], val3_0 = tmpptr[3];
                float val4_0 = tmpptr[4], val5_0 = tmpptr[5], val6_0 = tmpptr[6], val7_0 = tmpptr[7];
                float val0_1 = tmpptr[8], val1_1 = tmpptr[9], val2_1 = tmpptr[10], val3_1 = tmpptr[11];
                float val4_1 = tmpptr[12], val5_1 = tmpptr[13], val6_1 = tmpptr[14], val7_1 = tmpptr[15];
                
                vfloat32m1_t _w0 = __riscv_vle32_v_f32m1(kptr0, vl);
                vfloat32m1_t _w1 = __riscv_vle32_v_f32m1(kptr0 + packn, vl);

                _sum0 = __riscv_vfmacc_vf_f32m1(_sum0, val0_0, _w0, vl);
                _sum1 = __riscv_vfmacc_vf_f32m1(_sum1, val1_0, _w0, vl);
                _sum2 = __riscv_vfmacc_vf_f32m1(_sum2, val2_0, _w0, vl);
                _sum3 = __riscv_vfmacc_vf_f32m1(_sum3, val3_0, _w0, vl);
                _sum4 = __riscv_vfmacc_vf_f32m1(_sum4, val4_0, _w0, vl);
                _sum5 = __riscv_vfmacc_vf_f32m1(_sum5, val5_0, _w0, vl);
                _sum6 = __riscv_vfmacc_vf_f32m1(_sum6, val6_0, _w0, vl);
                _sum7 = __riscv_vfmacc_vf_f32m1(_sum7, val7_0, _w0, vl);

                _sum0 = __riscv_vfmacc_vf_f32m1(_sum0, val0_1, _w1, vl);
                _sum1 = __riscv_vfmacc_vf_f32m1(_sum1, val1_1, _w1, vl);
                _sum2 = __riscv_vfmacc_vf_f32m1(_sum2, val2_1, _w1, vl);
                _sum3 = __riscv_vfmacc_vf_f32m1(_sum3, val3_1, _w1, vl);
                _sum4 = __riscv_vfmacc_vf_f32m1(_sum4, val4_1, _w1, vl);
                _sum5 = __riscv_vfmacc_vf_f32m1(_sum5, val5_1, _w1, vl);
                _sum6 = __riscv_vfmacc_vf_f32m1(_sum6, val6_1, _w1, vl);
                _sum7 = __riscv_vfmacc_vf_f32m1(_sum7, val7_1, _w1, vl);

                tmpptr += 16;
                kptr0 += packn * 2;
            }

            for (; j < nn; j++)
            {
                float val0 = tmpptr[0], val1 = tmpptr[1], val2 = tmpptr[2], val3 = tmpptr[3];
                float val4 = tmpptr[4], val5 = tmpptr[5], val6 = tmpptr[6], val7 = tmpptr[7];
                
                vfloat32m1_t _w0 = __riscv_vle32_v_f32m1(kptr0, vl);
                _sum0 = __riscv_vfmacc_vf_f32m1(_sum0, val0, _w0, vl);
                _sum1 = __riscv_vfmacc_vf_f32m1(_sum1, val1, _w0, vl);
                _sum2 = __riscv_vfmacc_vf_f32m1(_sum2, val2, _w0, vl);
                _sum3 = __riscv_vfmacc_vf_f32m1(_sum3, val3, _w0, vl);
                _sum4 = __riscv_vfmacc_vf_f32m1(_sum4, val4, _w0, vl);
                _sum5 = __riscv_vfmacc_vf_f32m1(_sum5, val5, _w0, vl);
                _sum6 = __riscv_vfmacc_vf_f32m1(_sum6, val6, _w0, vl);
                _sum7 = __riscv_vfmacc_vf_f32m1(_sum7, val7, _w0, vl);

                tmpptr += 8;
                kptr0 += packn;
            }

            __riscv_vse32_v_f32m1(outptr0, _sum0, vl);
            __riscv_vse32_v_f32m1(outptr0 + packn, _sum1, vl);
            __riscv_vse32_v_f32m1(outptr0 + packn * 2, _sum2, vl);
            __riscv_vse32_v_f32m1(outptr0 + packn * 3, _sum3, vl);
            __riscv_vse32_v_f32m1(outptr0 + packn * 4, _sum4, vl);
            __riscv_vse32_v_f32m1(outptr0 + packn * 5, _sum5, vl);
            __riscv_vse32_v_f32m1(outptr0 + packn * 6, _sum6, vl);
            __riscv_vse32_v_f32m1(outptr0 + packn * 7, _sum7, vl);

            outptr0 += packn * 8;
        }
        for (; i + 3 < size; i += 4)
        {
            const float* tmpptr = tmp.channel(i / 16 + (i % 16) / 8 + (i % 8) / 4);
            const float* kptr0 = kernel.channel(p);

            int nn = nn_total;

            vfloat32m1_t _sum0 = __riscv_vfmv_v_f_f32m1(0.f, vl);
            vfloat32m1_t _sum1 = __riscv_vfmv_v_f_f32m1(0.f, vl);
            vfloat32m1_t _sum2 = __riscv_vfmv_v_f_f32m1(0.f, vl);
            vfloat32m1_t _sum3 = __riscv_vfmv_v_f_f32m1(0.f, vl);

            if (bias)
            {
                vfloat32m1_t _bias = __riscv_vle32_v_f32m1(bias + p * packn, vl);
                _sum0 = _bias; _sum1 = _bias; _sum2 = _bias; _sum3 = _bias;
            }

            // Standard 1x processing for 4-element path
            for (int j = 0; j < nn; j++)
            {
                float val0 = tmpptr[0], val1 = tmpptr[1], val2 = tmpptr[2], val3 = tmpptr[3];
                vfloat32m1_t _w0 = __riscv_vle32_v_f32m1(kptr0, vl);
                
                _sum0 = __riscv_vfmacc_vf_f32m1(_sum0, val0, _w0, vl);
                _sum1 = __riscv_vfmacc_vf_f32m1(_sum1, val1, _w0, vl);
                _sum2 = __riscv_vfmacc_vf_f32m1(_sum2, val2, _w0, vl);
                _sum3 = __riscv_vfmacc_vf_f32m1(_sum3, val3, _w0, vl);

                tmpptr += 4;
                kptr0 += packn;
            }

            __riscv_vse32_v_f32m1(outptr0, _sum0, vl);
            __riscv_vse32_v_f32m1(outptr0 + packn, _sum1, vl);
            __riscv_vse32_v_f32m1(outptr0 + packn * 2, _sum2, vl);
            __riscv_vse32_v_f32m1(outptr0 + packn * 3, _sum3, vl);

            outptr0 += packn * 4;
        }
        for (; i + 1 < size; i += 2)
        {
            const float* tmpptr = tmp.channel(i / 16 + (i % 16) / 8 + (i % 8) / 4 + (i % 4) / 2);
            const float* kptr0 = kernel.channel(p);

            int nn = nn_total;

            vfloat32m1_t _sum0 = __riscv_vfmv_v_f_f32m1(0.f, vl);
            vfloat32m1_t _sum1 = __riscv_vfmv_v_f_f32m1(0.f, vl);

            if (bias)
            {
                vfloat32m1_t _bias = __riscv_vle32_v_f32m1(bias + p * packn, vl);
                _sum0 = _bias; _sum1 = _bias;
            }

            // Standard 1x processing for 2-element path
            for (int j = 0; j < nn; j++)
            {
                float val0 = tmpptr[0], val1 = tmpptr[1];
                vfloat32m1_t _w0 = __riscv_vle32_v_f32m1(kptr0, vl);
                _sum0 = __riscv_vfmacc_vf_f32m1(_sum0, val0, _w0, vl);
                _sum1 = __riscv_vfmacc_vf_f32m1(_sum1, val1, _w0, vl);

                tmpptr += 2;
                kptr0 += packn;
            }

            __riscv_vse32_v_f32m1(outptr0, _sum0, vl);
            __riscv_vse32_v_f32m1(outptr0 + packn, _sum1, vl);

            outptr0 += packn * 2;
        }
        for (; i < size; i++)
        {
            const float* tmpptr = tmp.channel(i / 16 + (i % 16) / 8 + (i % 8) / 4 + (i % 4) / 2 + i % 2);
            const float* kptr0 = kernel.channel(p);

            int nn = nn_total;

            vfloat32m1_t _sum = __riscv_vfmv_v_f_f32m1(0.f, vl);

            if (bias)
            {
                _sum = __riscv_vle32_v_f32m1(bias + p * packn, vl);
            }

            // Standard 1x processing for 1-element path
            for (int j = 0; j < nn; j++)
            {
                float val = tmpptr[0];
                vfloat32m1_t _w0 = __riscv_vle32_v_f32m1(kptr0, vl);
                _sum = __riscv_vfmacc_vf_f32m1(_sum, val, _w0, vl);

                tmpptr++;
                kptr0 += packn;
            }

            __riscv_vse32_v_f32m1(outptr0, _sum, vl);

            outptr0 += packn;
        }
    }
}

static void convolution_im2col_sgemm_packn_rvv(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& _bias, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, const Option& opt)
{
    const int packn = csrr_vlenb() / 4;
    const size_t vl = __riscv_vsetvl_e32m1(packn);

    int w = bottom_blob.w;
    int inch = bottom_blob.c;

    int outw = top_blob.w;
    int outh = top_blob.h;
    const int size = outw * outh;

    const int maxk = kernel_w * kernel_h;

    // im2col
    Mat bottom_im2col(size, maxk, inch, 4u * packn, packn, opt.workspace_allocator);
    {
        const int gap = (w * stride_h - outw * stride_w) * packn;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < inch; p++)
        {
            const Mat img = bottom_blob.channel(p);
            float* ptr = bottom_im2col.channel(p);

            for (int u = 0; u < kernel_h; u++)
            {
                for (int v = 0; v < kernel_w; v++)
                {
                    const float* sptr = img.row<const float>(dilation_h * u) + dilation_w * v * packn;

                    for (int i = 0; i < outh; i++)
                    {
                        int j = 0;
                        for (; j < outw; j++)
                        {
                            vfloat32m1_t _val = __riscv_vle32_v_f32m1(sptr, vl);
                            __riscv_vse32_v_f32m1(ptr, _val, vl);

                            sptr += stride_w * packn;
                            ptr += packn;
                        }

                        sptr += gap;
                    }
                }
            }
        }
    }

    im2col_sgemm_packn_rvv(bottom_im2col, top_blob, kernel, _bias, opt);
}
