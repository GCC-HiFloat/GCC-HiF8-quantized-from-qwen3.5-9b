#define HIF8_BATCHED_EPILOGUE 1
#define HIF8_GEMM_KERNEL_CLASS SoftHif8PrefillGemmKernel
#define HIF8_GEMM_KERNEL_ENTRY soft_hif8_prefill_gemm

#include "soft_hif8_gemm.cpp"
