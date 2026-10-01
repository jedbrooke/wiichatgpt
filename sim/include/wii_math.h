#pragma once

// Math abstraction layer for WiiChatGPT.
// Application code calls these functions. The backend is selected at compile
// time via WII_MATH_USE_CPU or WII_MATH_USE_GX.

#ifdef __cplusplus
extern "C" {
#endif

// Initialize the math backend.
// For GX: sets up EFB state, TEV defaults, texture cache.
// For CPU: no-op.
void wii_math_init(void);

// Shutdown the math backend and free resources.
void wii_math_shutdown(void);

// Matrix multiply: C = A * B
// A is M x K (row-major), B is K x N (row-major), C is M x N (row-major).
void wii_matmul(const float* A, const float* B, float* C,
                int M, int K, int N);

// Matrix-vector multiply: y = A * x
// A is M x K (row-major), x is length K, y is length M.
void wii_matvec(const float* A, const float* x, float* y,
                int M, int K);

// Elementwise addition: y[i] = a[i] + b[i]
void wii_add(const float* a, const float* b, float* y, int n);

// Elementwise multiplication: y[i] = a[i] * b[i]
void wii_mul(const float* a, const float* b, float* y, int n);

// ReLU activation: y[i] = max(0, x[i])
void wii_relu(const float* x, float* y, int n);

// SiLU activation: y[i] = x[i] * sigmoid(x[i])
void wii_silu(const float* x, float* y, int n);

// Softmax: y = softmax(x) over n elements
void wii_softmax(const float* x, float* y, int n);

// RMS normalization: y[i] = x[i] / rms(x) * weight[i]
void wii_rmsnorm(const float* x, const float* weight, float* y, int n);

// INT8 matrix-vector multiply with fused dequant: y = dequant(W) * x
// W is M x K int8 (row-major), scale is per-tensor fp32 scale factor.
// x is length K fp32, y is length M fp32.
// CPU backend: dequants row-by-row (no scratch buffer needed).
// On real Wii hardware: psq_l (int8 load+dequant) + ps_madd.
void wii_matvec_q8(const signed char* W, float scale, const float* x,
                   float* y, int M, int K);

#ifdef __cplusplus
}
#endif
