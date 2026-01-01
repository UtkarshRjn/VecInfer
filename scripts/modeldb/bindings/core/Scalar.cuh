#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

namespace core {

    namespace Scalar {

        // Helper for converting float to cuscalar_t based on the type
        template <typename T>
        __device__ T to_cuscalar(float val) { return static_cast<T>(val); }
        
        template <>
        __device__ __half to_cuscalar<__half>(float val) { return __float2half(val); }
        
        namespace AtScalar {
            using f16 = at::Half;
            using f32 = float;
        }
        
        namespace CuScalar {
            using f16 = __half;
            using f32 = float;

            // ====== Compile-time Ploymorphic Operations ======
            // Addition
            template <typename T>
            struct add {
                __device__ T operator()(T a, T b) const { return a + b; }
            };
            template <> struct add<f16> { __device__ f16 operator()(f16 a, f16 b) const { return __hadd(a, b); } };

            
            // Subtraction            
            template <typename T>
            struct sub {
                __device__ T operator()(T a, T b) const { return a - b; }
            };
            template <> struct sub<f16> { __device__ f16 operator()(f16 a, f16 b) const { return __hsub(a, b); } };


            // Multiplication
            template <typename T>
            struct mul {
                __device__ T operator()(T a, T b) const { return a * b; }
            };
            template <> struct mul<f16> { __device__ f16 operator()(f16 a, f16 b) const { return __hmul(a, b); } };
            
            // Division
            template <typename T>
            struct div {
                __device__ T operator()(T a, T b) const { return a / b; }
            };
            template <> struct div<f16> { __device__ f16 operator()(f16 a, f16 b) const { return __hdiv(a, b); } };
            
            // Maximum
            template <typename T>
            struct max {
                __device__ T operator()(T a, T b) const { return fmaxf(a, b); }
            };
            template <> struct max<f16> { __device__ f16 operator()(f16 a, f16 b) const { return __hmax(a, b); } };
            
            // Exponential
            template <typename T>
            struct exp {
                __device__ T operator()(T a) const { return expf(a); }
            };
            template <> struct exp<f16> { __device__ f16 operator()(f16 a) const { return hexp(a); } };
            
            // Logarithm
            template <typename T>
            struct log {
                __device__ T operator()(T a) const { return logf(a); }
            };
            template <> struct log<f16> { __device__ f16 operator()(f16 a) const { return hlog(a); } };
        } // namespace CuScalar
        
        // map from AtScalar to CuScalar
        template <typename T> struct at2cu { using type = void; };
        template <> struct at2cu<AtScalar::f32> { using type = CuScalar::f32; };
        template <> struct at2cu<AtScalar::f16> { using type = CuScalar::f16; };
        
    } // namespace Scalar

} // namespace core
