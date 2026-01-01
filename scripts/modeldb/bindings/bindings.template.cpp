#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <omp.h>
#include <torch/extension.h>

using u8 = uint8_t;
using u16 = uint16_t;
using f16 = at::Half;
using f32 = float;

#define declare_flash_decoding_allocated_buffer(T, code_alias, Nsv, Ltv, dv, Mv_key, Mv_value, Cv) \
	torch::Tensor flash_decoding_allocated_buffer_##T##code_alias##_Ns##Nsv##Lt##Ltv##d##dv##M_key##Mv_key##M_value##Mv_value##C##Cv( \
		const torch::Tensor query, \
		const torch::Tensor key_codes, \
		const torch::Tensor value_codes, \
		const torch::Tensor key_cents, \
		const torch::Tensor value_cents, \
		const torch::Tensor key_residuals, \
		const torch::Tensor value_residuals, \
		const int r, \
		const torch::Tensor partial_out_buffer, \
		const torch::Tensor partial_lse_buffer \
	)
