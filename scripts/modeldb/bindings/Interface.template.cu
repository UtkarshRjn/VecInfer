#include "Kernel.cuh"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

using u8 = uint8_t;
using f16 = core::Scalar::AtScalar::f16;

template
<
	typename scalar_t,
	typename code_t,
	const int Ns,
	const int Lt,
	const int d,
	const int M_key,
	const int M_value,
	const int C
>
torch::Tensor flash_decoding_allocated_buffer
(
	const torch::Tensor query, // shaped (bs, nh, 1, d)
	const torch::Tensor key_codes, // shaped (bs, nh_k, nk, M_key)
	const torch::Tensor value_codes, // shaped (bs, nh_k, nk, M_value)
	const torch::Tensor key_cents, // shaped (1, C, d/M_key)
	const torch::Tensor value_cents, // shaped (1, C, d/M_value)
	const torch::Tensor key_residuals, // shaped (bs, nh_k, Lt, d)
	const torch::Tensor value_residuals, // shaped (bs, nh_k, Lt, d)
	const int r, // residual length, 0 < r <= Lt
	const torch::Tensor partial_out_buffer, // shaped (bs, nh, Ns+1, d)
	const torch::Tensor partial_lse_buffer // shaped (bs, nh, Ns+1)
)
{
	const int bs = query.size(0);
	const int nh = query.size(1);
	const int nh_k = key_codes.size(1);
	const int nk = key_codes.size(2);

	// convert torch type to cuda type
	using cuscalar_t = typename cs::at2cu<scalar_t>::type;

	// get pointer to partial_out, partial_lse
	cuscalar_t *partial_out = reinterpret_cast<cuscalar_t*>(partial_out_buffer.data_ptr<scalar_t>());
	cuscalar_t *partial_lse = reinterpret_cast<cuscalar_t*>(partial_lse_buffer.data_ptr<scalar_t>());

	const int Ls = (nk + Ns - 1) / Ns; // number of key_codes and value_codes inside a split.

	// Fill partial_out[:, :, :Ns] and partial_lse[:, :, :Ns]
	// Skip this part if nk=0 (no key_codes to process)
	if (nk > 0 ) 
	{
		auto query_reshaped = query.reshape({bs, nh, M_key, d/M_key});
		auto ad_lut = at::matmul(query_reshaped, key_cents.transpose(1, 2));
		query_reshaped = torch::Tensor(); // release memory
		// query_reshaped.reset();
		{
			dim3 split_grid(bs, nh, Ns);
			dim3 split_block(Lt, 1, 1);
			
			flash_decoding_split_kernel<cuscalar_t, code_t, Ns, Lt, d, M_key, M_value, C><<<split_grid, split_block>>>(
				reinterpret_cast<const cuscalar_t*>(ad_lut.data_ptr<scalar_t>()),
				key_codes.data_ptr<code_t>(),
				value_codes.data_ptr<code_t>(),
				reinterpret_cast<const cuscalar_t*>(value_cents.data_ptr<scalar_t>()),
				partial_out,
				partial_lse,
				bs,
				nh,
				nh_k,
				nk,
				Ls
			);
			gpuErrchk(cudaPeekAtLastError());
		}
		ad_lut = torch::Tensor(); // release memory
		// ad_lut.reset();
	}else{
		// Initialize partial_out and partial_lse buffers for splits to zero
		// Since there are no key_codes, all splits should have zero contribution
		cudaMemset(partial_out, 0, bs * nh * Ns * d * sizeof(cuscalar_t));
		
		// Initialize partial_lse for splits to -inf (or a very small number)
		// This ensures proper log-sum-exp computation
		size_t lse_split_size = bs * nh * Ns;
		cuscalar_t neg_inf = -std::numeric_limits<scalar_t>::infinity();
		std::vector<cuscalar_t> neg_inf_vec(lse_split_size, neg_inf);
		cudaMemcpy(partial_lse, neg_inf_vec.data(), lse_split_size * sizeof(cuscalar_t), cudaMemcpyHostToDevice);
	}
	

	// Fill partial_out[:, :, -1] and partial_lse[:, :, -1]
	{
		dim3 residual_grid(bs, nh);
		dim3 residual_block(d, 1, 1);
		// constexpr int DEN = sizeof(float) / sizeof(cuscalar_t); // scalar density, for example, float -> 1, half -> 2
		flash_decoding_residual_kernel<cuscalar_t, Ns, Lt, d><<<residual_grid, residual_block>>>(
			reinterpret_cast<const cuscalar_t*>(query.data_ptr<scalar_t>()),
			reinterpret_cast<const cuscalar_t*>(key_residuals.data_ptr<scalar_t>()),
			reinterpret_cast<const cuscalar_t*>(value_residuals.data_ptr<scalar_t>()),
			r,
			partial_out,
			partial_lse,
			bs,
			nh,
			nh_k
		);

		gpuErrchk(cudaPeekAtLastError());
		
		// cudaError_t err = cudaGetLastError();
		// if (err != cudaSuccess) {
		// 	fprintf(stderr, "CUDA Error: %s (%s) at %s:%d\n",
		// 			cudaGetErrorName(err),  
		// 			cudaGetErrorString(err),
		// 			__FILE__, __LINE__);
		// }

	}
		
	// Reduce partial_out and partial_lse to out
	torch::Tensor out = torch::zeros({bs, nh, 1, d}, query.options());
	{
		dim3 reduce_grid(bs, nh);
		dim3 reduce_block(d, 1, 1);
		
		flash_decoding_reduce_kernel<cuscalar_t, Ns, d><<<reduce_grid, reduce_block>>>(
			partial_out,
			partial_lse,
			reinterpret_cast<cuscalar_t*>(out.data_ptr<scalar_t>()),
			bs,
			nh
		);
		
		gpuErrchk(cudaPeekAtLastError());
	}
	return out;
}

#define register_flash_decoding_allocated_buffer(T, code_alias, Nsv, Ltv, dv, Mv_key, Mv_value, Cv) \
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
	) { \
		return flash_decoding_allocated_buffer<T, code_alias, Nsv, Ltv, dv, Mv_key, Mv_value, Cv>( \
			query, \
			key_codes, \
			value_codes, \
			key_cents, \
			value_cents, \
			key_residuals, \
			value_residuals, \
			r, \
			partial_out_buffer, \
			partial_lse_buffer \
		); \
}
