#include "hip/hip_runtime.h"
/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#pragma once

#ifndef OPENCV_CUDEV_GRID_INTEGRAL_DETAIL_HPP
#define OPENCV_CUDEV_GRID_INTEGRAL_DETAIL_HPP

#include "../../common.hpp"
#include "../../warp/shuffle.hpp"
#include "../../block/scan.hpp"
#include "../../ptr2d/glob.hpp"

namespace cv { namespace cudev {

namespace integral_detail
{
    // horizontal_pass

    template <int NUM_SCAN_THREADS, class SrcPtr, typename D>
    __global__ void horizontal_pass(const SrcPtr src, GlobPtr<D> dst, const int cols)
    {
        __shared__ D smem[NUM_SCAN_THREADS * 2];
        __shared__ D carryElem;

        carryElem = 0;

        __syncthreads();

        D* dst_row = dst.row(hipBlockIdx_x);

        int numBuckets = divUp(cols, NUM_SCAN_THREADS);
        int offsetX = 0;

        while (numBuckets--)
        {
            const int curElemOffs = offsetX + hipThreadIdx_x;

            D curElem = 0.0f;

            if (curElemOffs < cols)
                curElem = src(hipBlockIdx_x, curElemOffs);

            const D curScanElem = blockScanInclusive<NUM_SCAN_THREADS>(curElem, smem, hipThreadIdx_x);

            if (curElemOffs < cols)
                dst_row[curElemOffs] = carryElem + curScanElem;

            offsetX += NUM_SCAN_THREADS;

            __syncthreads();

            if (hipThreadIdx_x == NUM_SCAN_THREADS - 1)
            {
                carryElem += curScanElem;
            }

            __syncthreads();
        }
    }

    template <int NUM_SCAN_THREADS, typename T, typename D>
    __global__ void horizontal_pass(const GlobPtr<T> src, GlobPtr<D> dst, const int cols)
    {
        __shared__ D smem[NUM_SCAN_THREADS * 2];
        __shared__ D carryElem;

        carryElem = 0;

        __syncthreads();

        const T* src_row = src.row(hipBlockIdx_x);
        D* dst_row = dst.row(hipBlockIdx_x);

        int numBuckets = divUp(cols, NUM_SCAN_THREADS);
        int offsetX = 0;

        while (numBuckets--)
        {
            const int curElemOffs = offsetX + hipThreadIdx_x;

            D curElem = 0.0f;

            if (curElemOffs < cols)
                curElem = src_row[curElemOffs];

            const D curScanElem = blockScanInclusive<NUM_SCAN_THREADS>(curElem, smem, hipThreadIdx_x);

            if (curElemOffs < cols)
                dst_row[curElemOffs] = carryElem + curScanElem;

            offsetX += NUM_SCAN_THREADS;

            __syncthreads();

            if (hipThreadIdx_x == NUM_SCAN_THREADS - 1)
            {
                carryElem += curScanElem;
            }

            __syncthreads();
        }
    }

    template <class SrcPtr, typename D>
    __host__ void horizontal_pass(const SrcPtr& src, const GlobPtr<D>& dst, int rows, int cols, hipStream_t stream)
    {
        const int NUM_SCAN_THREADS = 256;

        const dim3 block(NUM_SCAN_THREADS);
        const dim3 grid(rows);

        hipLaunchKernelGGL((horizontal_pass<NUM_SCAN_THREADS>), dim3(grid), dim3(block), 0, stream, src, dst, cols);
        CV_CUDEV_SAFE_CALL( hipGetLastError() );
    }

    // horisontal_pass_8u_shfl

    __device__ static uchar4 int_to_uchar4(unsigned int in)
    {
        uchar4 bytes;
        bytes.x = (in & 0x000000ff) >>  0;
        bytes.y = (in & 0x0000ff00) >>  8;
        bytes.z = (in & 0x00ff0000) >> 16;
        bytes.w = (in & 0xff000000) >> 24;
        return bytes;
    }

    __global__ static void horisontal_pass_8u_shfl_kernel(const GlobPtr<uint4> img, GlobPtr<uint4> integral)
    {
    #if CV_CUDEV_ARCH >= 300
        __shared__ int sums[128];

        const int id = hipThreadIdx_x;
        const int lane_id = id % warpSize;
        const int warp_id = id / warpSize;

        const uint4 data = img(hipBlockIdx_x, id);

        const uchar4 a = int_to_uchar4(data.x);
        const uchar4 b = int_to_uchar4(data.y);
        const uchar4 c = int_to_uchar4(data.z);
        const uchar4 d = int_to_uchar4(data.w);

        int result[16];

        result[0]  =              a.x;
        result[1]  = result[0]  + a.y;
        result[2]  = result[1]  + a.z;
        result[3]  = result[2]  + a.w;

        result[4]  = result[3]  + b.x;
        result[5]  = result[4]  + b.y;
        result[6]  = result[5]  + b.z;
        result[7]  = result[6]  + b.w;

        result[8]  = result[7]  + c.x;
        result[9]  = result[8]  + c.y;
        result[10] = result[9]  + c.z;
        result[11] = result[10] + c.w;

        result[12] = result[11] + d.x;
        result[13] = result[12] + d.y;
        result[14] = result[13] + d.z;
        result[15] = result[14] + d.w;

        int sum = result[15];

        // the prefix sum for each thread's 16 value is computed,
        // now the final sums (result[15]) need to be shared
        // with the other threads and add.  To do this,
        // the shfl_up() instruction is used and a shuffle scan
        // operation is performed to distribute the sums to the correct
        // threads
        #pragma unroll
        for (int i = 1; i < 32; i *= 2)
        {
            const int n = shfl_up(sum, i, 32);

            if (lane_id >= i)
            {
                #pragma unroll
                for (int k = 0; k < 16; ++k)
                    result[k] += n;

                sum += n;
            }
        }

        // Now the final sum for the warp must be shared
        // between warps.  This is done by each warp
        // having a thread store to shared memory, then
        // having some other warp load the values and
        // compute a prefix sum, again by using shfl_up.
        // The results are uniformly added back to the warps.
        // last thread in the warp holding sum of the warp
        // places that in shared
        if (hipThreadIdx_x % warpSize == warpSize - 1)
            sums[warp_id] = result[15];

        __syncthreads();

        if (warp_id == 0)
        {
            int warp_sum = sums[lane_id];

            #pragma unroll
            for (int i = 1; i <= 32; i *= 2)
            {
                const int n = shfl_up(warp_sum, i, 32);

                if (lane_id >= i)
                    warp_sum += n;
            }

            sums[lane_id] = warp_sum;
        }

        __syncthreads();

        int blockSum = 0;

        // fold in unused warp
        if (warp_id > 0)
        {
            blockSum = sums[warp_id - 1];

            #pragma unroll
            for (int k = 0; k < 16; ++k)
                result[k] += blockSum;
        }

        // assemble result
        // Each thread has 16 values to write, which are
        // now integer data (to avoid overflow).  Instead of
        // each thread writing consecutive uint4s, the
        // approach shown here experiments using
        // the shuffle command to reformat the data
        // inside the registers so that each thread holds
        // consecutive data to be written so larger contiguous
        // segments can be assembled for writing.

        /*
            For example data that needs to be written as

            GMEM[16] <- x0 x1 x2 x3 y0 y1 y2 y3 z0 z1 z2 z3 w0 w1 w2 w3
            but is stored in registers (r0..r3), in four threads (0..3) as:

            threadId   0  1  2  3
              r0      x0 y0 z0 w0
              r1      x1 y1 z1 w1
              r2      x2 y2 z2 w2
              r3      x3 y3 z3 w3

              after apply shfl_xor operations to move data between registers r1..r3:

            threadId  00 01 10 11
                      x0 y0 z0 w0
             xor(01)->y1 x1 w1 z1
             xor(10)->z2 w2 x2 y2
             xor(11)->w3 z3 y3 x3

             and now x0..x3, and z0..z3 can be written out in order by all threads.

             In the current code, each register above is actually representing
             four integers to be written as uint4's to GMEM.
        */

        result[4]  = shfl_xor(result[4] , 1, 32);
        result[5]  = shfl_xor(result[5] , 1, 32);
        result[6]  = shfl_xor(result[6] , 1, 32);
        result[7]  = shfl_xor(result[7] , 1, 32);

        result[8]  = shfl_xor(result[8] , 2, 32);
        result[9]  = shfl_xor(result[9] , 2, 32);
        result[10] = shfl_xor(result[10], 2, 32);
        result[11] = shfl_xor(result[11], 2, 32);

        result[12] = shfl_xor(result[12], 3, 32);
        result[13] = shfl_xor(result[13], 3, 32);
        result[14] = shfl_xor(result[14], 3, 32);
        result[15] = shfl_xor(result[15], 3, 32);

        uint4* integral_row = integral.row(hipBlockIdx_x);
        uint4 output;

        ///////

        if (hipThreadIdx_x % 4 == 0)
            output = make_uint4(result[0], result[1], result[2], result[3]);

        if (hipThreadIdx_x % 4 == 1)
            output = make_uint4(result[4], result[5], result[6], result[7]);

        if (hipThreadIdx_x % 4 == 2)
            output = make_uint4(result[8], result[9], result[10], result[11]);

        if (hipThreadIdx_x % 4 == 3)
            output = make_uint4(result[12], result[13], result[14], result[15]);

        integral_row[hipThreadIdx_x % 4 + (hipThreadIdx_x / 4) * 16] = output;

        ///////

        if (hipThreadIdx_x % 4 == 2)
            output = make_uint4(result[0], result[1], result[2], result[3]);

        if (hipThreadIdx_x % 4 == 3)
            output = make_uint4(result[4], result[5], result[6], result[7]);

        if (hipThreadIdx_x % 4 == 0)
            output = make_uint4(result[8], result[9], result[10], result[11]);

        if (hipThreadIdx_x % 4 == 1)
            output = make_uint4(result[12], result[13], result[14], result[15]);

        integral_row[(hipThreadIdx_x + 2) % 4 + (hipThreadIdx_x / 4) * 16 + 8] = output;

        // continuning from the above example,
        // this use of shfl_xor() places the y0..y3 and w0..w3 data
        // in order.

        #pragma unroll
        for (int i = 0; i < 16; ++i)
            result[i] = shfl_xor(result[i], 1, 32);

        if (hipThreadIdx_x % 4 == 0)
            output = make_uint4(result[0], result[1], result[2], result[3]);

        if (hipThreadIdx_x % 4 == 1)
            output = make_uint4(result[4], result[5], result[6], result[7]);

        if (hipThreadIdx_x % 4 == 2)
            output = make_uint4(result[8], result[9], result[10], result[11]);

        if (hipThreadIdx_x % 4 == 3)
            output = make_uint4(result[12], result[13], result[14], result[15]);

        integral_row[hipThreadIdx_x % 4 + (hipThreadIdx_x / 4) * 16 + 4] = output;

        ///////

        if (hipThreadIdx_x % 4 == 2)
            output = make_uint4(result[0], result[1], result[2], result[3]);

        if (hipThreadIdx_x % 4 == 3)
            output = make_uint4(result[4], result[5], result[6], result[7]);

        if (hipThreadIdx_x % 4 == 0)
            output = make_uint4(result[8], result[9], result[10], result[11]);

        if (hipThreadIdx_x % 4 == 1)
            output = make_uint4(result[12], result[13], result[14], result[15]);

        integral_row[(hipThreadIdx_x + 2) % 4 + (hipThreadIdx_x / 4) * 16 + 12] = output;
    #endif
    }

    __host__ static void horisontal_pass_8u_shfl(const GlobPtr<uchar> src, GlobPtr<uint> integral, int rows, int cols, hipStream_t stream)
    {
        // each thread handles 16 values, use 1 block/row
        // save, because step is actually can't be less 512 bytes
        const int block = cols / 16;

        // launch 1 block / row
        const int grid = rows;

#ifdef HIP_TODO
        CV_CUDEV_SAFE_CALL( hipFuncSetCacheConfig(horisontal_pass_8u_shfl_kernel, hipFuncCachePreferL1) );
#endif  // HIP_TODO

        GlobPtr<uint4> src4 = globPtr((uint4*) src.data, src.step);
        GlobPtr<uint4> integral4 = globPtr((uint4*) integral.data, integral.step);

        hipLaunchKernelGGL((horisontal_pass_8u_shfl_kernel), dim3(grid), dim3(block), 0, stream, src4, integral4);
        CV_CUDEV_SAFE_CALL( hipGetLastError() );
    }

    // vertical

    template <typename T>
    __global__ void vertical_pass(GlobPtr<T> integral, const int rows, const int cols)
    {
    #if CV_CUDEV_ARCH >= 300
        __shared__ T sums[32][9];

        const int tidx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
        const int lane_id = tidx % 8;

        sums[hipThreadIdx_x][hipThreadIdx_y] = 0;
        __syncthreads();

        T stepSum = 0;

        int numBuckets = divUp(rows, hipBlockDim_y);
        int y = hipThreadIdx_y;

        while (numBuckets--)
        {
            T* p = integral.row(y) + tidx;

            T sum = (tidx < cols) && (y < rows) ? *p : 0;

            sums[hipThreadIdx_x][hipThreadIdx_y] = sum;
            __syncthreads();

            // place into SMEM
            // shfl scan reduce the SMEM, reformating so the column
            // sums are computed in a warp
            // then read out properly
            const int j = hipThreadIdx_x % 8;
            const int k = hipThreadIdx_x / 8 + hipThreadIdx_y * 4;

            T partial_sum = sums[k][j];

            for (int i = 1; i <= 8; i *= 2)
            {
                T n = shfl_up(partial_sum, i, 32);

                if (lane_id >= i)
                    partial_sum += n;
            }

            sums[k][j] = partial_sum;
            __syncthreads();

            if (hipThreadIdx_y > 0)
                sum += sums[hipThreadIdx_x][hipThreadIdx_y - 1];

            sum += stepSum;
            stepSum += sums[hipThreadIdx_x][hipBlockDim_y - 1];

            __syncthreads();

            if ((tidx < cols) && (y < rows))
            {
                *p = sum;
            }

            y += hipBlockDim_y;
        }
    #else
        __shared__ T smem[32][32];
        __shared__ T prevVals[32];

        volatile T* smem_row = &smem[0][0] + 64 * hipThreadIdx_y;

        if (hipThreadIdx_y == 0)
            prevVals[hipThreadIdx_x] = 0;

        __syncthreads();

        const int x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

        int numBuckets = divUp(rows, 8 * 4);
        int offsetY = 0;

        while (numBuckets--)
        {
            const int curRowOffs = offsetY + hipThreadIdx_y;

            T curElems[4];
            T temp[4];

            // load patch

            smem[hipThreadIdx_y +  0][hipThreadIdx_x] = 0.0f;
            smem[hipThreadIdx_y +  8][hipThreadIdx_x] = 0.0f;
            smem[hipThreadIdx_y + 16][hipThreadIdx_x] = 0.0f;
            smem[hipThreadIdx_y + 24][hipThreadIdx_x] = 0.0f;

            if (x < cols)
            {
                for (int i = 0; i < 4; ++i)
                {
                    if (curRowOffs + i * 8 < rows)
                        smem[hipThreadIdx_y + i * 8][hipThreadIdx_x] = integral(curRowOffs + i * 8, x);
                }
            }

            __syncthreads();

            // reduce

            curElems[0] = smem[hipThreadIdx_x][hipThreadIdx_y     ];
            curElems[1] = smem[hipThreadIdx_x][hipThreadIdx_y +  8];
            curElems[2] = smem[hipThreadIdx_x][hipThreadIdx_y + 16];
            curElems[3] = smem[hipThreadIdx_x][hipThreadIdx_y + 24];

            __syncthreads();

            temp[0] = curElems[0] = warpScanInclusive(curElems[0], smem_row, hipThreadIdx_x);
            temp[1] = curElems[1] = warpScanInclusive(curElems[1], smem_row, hipThreadIdx_x);
            temp[2] = curElems[2] = warpScanInclusive(curElems[2], smem_row, hipThreadIdx_x);
            temp[3] = curElems[3] = warpScanInclusive(curElems[3], smem_row, hipThreadIdx_x);

            curElems[0] += prevVals[hipThreadIdx_y     ];
            curElems[1] += prevVals[hipThreadIdx_y +  8];
            curElems[2] += prevVals[hipThreadIdx_y + 16];
            curElems[3] += prevVals[hipThreadIdx_y + 24];

            __syncthreads();

            if (hipThreadIdx_x == 31)
            {
                prevVals[hipThreadIdx_y     ] += temp[0];
                prevVals[hipThreadIdx_y +  8] += temp[1];
                prevVals[hipThreadIdx_y + 16] += temp[2];
                prevVals[hipThreadIdx_y + 24] += temp[3];
            }

            smem[hipThreadIdx_y     ][hipThreadIdx_x] = curElems[0];
            smem[hipThreadIdx_y +  8][hipThreadIdx_x] = curElems[1];
            smem[hipThreadIdx_y + 16][hipThreadIdx_x] = curElems[2];
            smem[hipThreadIdx_y + 24][hipThreadIdx_x] = curElems[3];

            __syncthreads();

            // store patch

            if (x < cols)
            {
                // read 4 value from source
                for (int i = 0; i < 4; ++i)
                {
                    if (curRowOffs + i * 8 < rows)
                        integral(curRowOffs + i * 8, x) = smem[hipThreadIdx_x][hipThreadIdx_y + i * 8];
                }
            }

            __syncthreads();

            offsetY += 8 * 4;
        }
    #endif
    }

    template <typename T>
    __host__ void vertical_pass(const GlobPtr<T>& integral, int rows, int cols, hipStream_t stream)
    {
        const dim3 block(32, 8);
        const dim3 grid(divUp(cols, block.x));

        hipLaunchKernelGGL((vertical_pass), dim3(grid), dim3(block), 0, stream, integral, rows, cols);
        CV_CUDEV_SAFE_CALL( hipGetLastError() );
    }

    // integral

    template <class SrcPtr, typename D>
    __host__ void integral(const SrcPtr& src, const GlobPtr<D>& dst, int rows, int cols, hipStream_t stream)
    {
        horizontal_pass(src, dst, rows, cols, stream);
        vertical_pass(dst, rows, cols, stream);

        if (stream == 0)
            CV_CUDEV_SAFE_CALL( hipDeviceSynchronize() );
    }

    __host__ static void integral(const GlobPtr<uchar>& src, const GlobPtr<uint>& dst, int rows, int cols, hipStream_t stream)
    {
        if (deviceSupports(FEATURE_SET_COMPUTE_30)
            && (cols % 64 == 0)
            && reinterpret_cast<intptr_t>(src.data) % 32 == 0
            && reinterpret_cast<intptr_t>(dst.data) % 32 == 0)
        {
            horisontal_pass_8u_shfl(src, dst, rows, cols, stream);
        }
        else
        {
            horizontal_pass(src, dst, rows, cols, stream);
        }

        vertical_pass(dst, rows, cols, stream);

        if (stream == 0)
            CV_CUDEV_SAFE_CALL( hipDeviceSynchronize() );
    }

    __host__ __forceinline__ void integral(const GlobPtr<uchar>& src, const GlobPtr<int>& dst, int rows, int cols, hipStream_t stream)
    {
        GlobPtr<uint> dstui = globPtr((uint*) dst.data, dst.step);
        integral(src, dstui, rows, cols, stream);
    }
}

}}

#endif
