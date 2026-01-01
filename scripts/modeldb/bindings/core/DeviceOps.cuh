#pragma once

#include "Reduction.cuh"

namespace core {

    namespace DeviceOps {

        // ====== Device Operations ======
    
        template <typename cuscalar_t, template <typename> class Reduction>
        __device__ __forceinline__ cuscalar_t single_thread_reduce(
            const cuscalar_t* src,
            const int n_elem
        ) {
            cuscalar_t res = Reduction<cuscalar_t>::Identity();
            for (int i=0; i<n_elem; ++i) {
                res = Reduction<cuscalar_t>()(res, src[i]);
            }
            return res;
        }

        template <typename cuscalar_t>
        __device__ void block_copy(
            cuscalar_t* dst,
            const cuscalar_t* src,
            const int n_elem,
            const unsigned int n_threads,
            const int tid
        )
        {
            /**
            *   @brief Copy from src to dst in a block-wise coalesced fashion.
            *       Vectorized copy is used to minimize the number of memory transactions.
            */
    
            // assert that n_elem * sizeof(cuscalar_t) is a multiple of 16
            // static_assert(n_elem * sizeof(cuscalar_t) % 16 == 0, "n_elem * sizeof(cuscalar_t) must be a multiple of 16");
    
            using v16_t = typename Vector::VectorType<16>::type; // 16 bytes
    
            v16_t* a = reinterpret_cast<v16_t*>(const_cast<cuscalar_t*>(src));
            v16_t* b = reinterpret_cast<v16_t*>(dst);
    
    
            int n_wide = n_elem * sizeof(cuscalar_t) / sizeof(v16_t);
        
            for (int i = tid; i < n_wide; i += n_threads) {
                if (i < n_wide) {
                    b[i] = a[i];
                }
            }
        }
    
        template <typename cuscalar_t, template <typename> class Op> 
        __device__ void block_op(
            cuscalar_t* dst,
            const cuscalar_t* loperand,
            const cuscalar_t* roperand,
            const int n_elem,
            const unsigned int n_threads,
            const int tid
        )
        {
            /**
            *   @brief Perform element-wise operation on two arrays in a block-wise coalesced fashion.
            */
    
            for (int i = tid; i < n_elem; i += n_threads) {
                dst[i] = Op<cuscalar_t>()(loperand[i], roperand[i]);
            }
        }
    
        template <typename cuscalar_t, template <typename> class Op> 
        __device__ void block_op(
            cuscalar_t* dst,
            const cuscalar_t* loperand,
            const cuscalar_t roperand,
            const int n_elem,
            const unsigned int n_threads,
            const int tid
        )
        {
            /**
            *   @brief Perform element-wise operation on an array(loperand) and a scalar(roperand) in a block-wise coalesced fashion.
            */
    
            for (int i = tid; i < n_elem; i += n_threads) {
                dst[i] = Op<cuscalar_t>()(loperand[i], roperand);
            }
        }
    
        template <typename cuscalar_t, template <typename> class Op> 
        __device__ void block_op(
            cuscalar_t* dst,
            const cuscalar_t loperand,
            const cuscalar_t* roperand,
            const int n_elem,
            const unsigned int n_threads,
            const int tid
        )
        {
            /**
            *   @brief Perform element-wise operation on a scalar(loperand) and an array(roperand) in a block-wise coalesced fashion.
            */
    
            for (int i = tid; i < n_elem; i += n_threads) {
                dst[i] = Op<cuscalar_t>()(loperand, roperand[i]);
            }
        }
    
        template <typename cuscalar_t>
        __device__ void block_fill_zero(
            cuscalar_t* dst,
            const int n_elem,
            const unsigned int n_threads,
            const int tid
        )
        {
            /**
            *   @brief Fill an array with zeros in a block-wise coalesced fashion.
            */
    
            for (int i = tid; i < n_elem; i += n_threads) {
                dst[i] = Scalar::to_cuscalar<cuscalar_t>(0.0f);
            }
        }
    
    
    } // namespace DeviceOps

} // namespace core

