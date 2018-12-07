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

#ifndef OPENCV_CUDEV_GRID_REDUCE_TO_ROW_DETAIL_HPP
#define OPENCV_CUDEV_GRID_REDUCE_TO_ROW_DETAIL_HPP

#include "../../common.hpp"
#include "../../util/saturate_cast.hpp"
#include "../../block/reduce.hpp"

namespace cv { namespace cudev {

namespace grid_reduce_to_vec_detail
{
    template <class Reductor, int BLOCK_SIZE_X, int BLOCK_SIZE_Y, class SrcPtr, typename ResType, class MaskPtr>
    __global__ void reduceToRow(const SrcPtr src, ResType* dst, const MaskPtr mask, const int rows, const int cols)
    {
        typedef typename Reductor::work_type work_type;

        __shared__ work_type smem[BLOCK_SIZE_X * BLOCK_SIZE_Y];

        const int x = hipBlockIdx_x * BLOCK_SIZE_X + hipThreadIdx_x;

        work_type myVal = Reductor::initialValue();

        Reductor op;

        if (x < cols)
        {
            for (int y = hipThreadIdx_y; y < rows; y += BLOCK_SIZE_Y)
            {
                if (mask(y, x))
                {
                    myVal = op(myVal, saturate_cast<work_type>(src(y, x)));
                }
            }
        }

        smem[hipThreadIdx_x * BLOCK_SIZE_Y + hipThreadIdx_y] = myVal;

        __syncthreads();

        volatile work_type* srow = smem + hipThreadIdx_y * BLOCK_SIZE_X;

        myVal = srow[hipThreadIdx_x];
        blockReduce<BLOCK_SIZE_X>(srow, myVal, hipThreadIdx_x, op);

        if (hipThreadIdx_x == 0)
            srow[0] = myVal;

        __syncthreads();

        if (hipThreadIdx_y == 0 && x < cols)
            dst[x] = saturate_cast<ResType>(Reductor::result(smem[hipThreadIdx_x * BLOCK_SIZE_X], rows));
    }

    template <class Reductor, class SrcPtr, typename ResType, class MaskPtr>
    __host__ void reduceToRow(const SrcPtr& src, ResType* dst, const MaskPtr& mask, int rows, int cols, hipStream_t stream)
    {
        const int BLOCK_SIZE_X = 16;
        const int BLOCK_SIZE_Y = 16;

        const dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
        const dim3 grid(divUp(cols, block.x));

        hipLaunchKernelGGL((reduceToRow<Reductor, BLOCK_SIZE_X, BLOCK_SIZE_Y>), dim3(grid), dim3(block), 0, stream, src, dst, mask, rows, cols);
        CV_CUDEV_SAFE_CALL( hipGetLastError() );

        if (stream == 0)
            CV_CUDEV_SAFE_CALL( hipDeviceSynchronize() );
    }
}

}}

#endif
