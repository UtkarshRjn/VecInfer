#pragma once

#include "Scalar.cuh"

#include <cuda.h>
#include <math_constants.h>

namespace core {

    namespace Reduction {
        template <typename T>
        struct MaxReduction {
            __device__ T operator()(T a, T b) const { return Scalar::CuScalar::max<T>()(a, b); }
            static __device__ __forceinline__ T Identity() { return Scalar::to_cuscalar<T>(-CUDART_INF_F); }
        };
    
        template <typename T>
        struct SumReduction {
            __device__ T operator()(T a, T b) const { return Scalar::CuScalar::add<T>()(a, b); }
            static __device__ __forceinline__ T Identity() { return Scalar::to_cuscalar<T>(0.0f); }
        };
    } // namespace Reduction

} // namespace core



