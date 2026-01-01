#pragma once

#include <cuda.h>
#include <cuda_fp16.h>

/**
 * @brief Choose wide vector type based on compile-time width.
 *
 * This template maps a byte width (4, 8, or 16) to a corresponding CUDA vector type:
 * - 4 bytes: `float`
 * - 8 bytes: `float2`
 * - 16 bytes: `float4`
 *
*/

namespace core {

    namespace Vector{

        // ========= Templates ==========
        
        template <int w_byte> // width in bytes
        struct VectorType {
            static_assert(w_byte == 4 || w_byte == 8 || w_byte == 16, "Width must be 4, 8, or 16 bytes.");
            using type = void;
        };
        
        // ========= Specializations =========
        
        template <> struct VectorType<4> { using type = float; };
        template <> struct VectorType<8> { using type = float2; };
        template <> struct VectorType<16> { using type = float4; }; 
    
    } // namespace Vector
    
} // namespace core
