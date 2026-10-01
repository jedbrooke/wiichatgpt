// CPU reference implementation of wii_math.h
// Optimized scalar C++ targeting PPC750 (Broadway) in the Wii.
// No SIMD, no platform-specific intrinsics. This is ground truth
// for validating the GX backend.

#include "wii_math.h"
#include <cmath>
#include <cstring>
#include <cstdint>
#include <cstdlib>

void wii_math_init(void) {}

void wii_math_shutdown(void) {}

// ---------------------------------------------------------------------------
// Fast exp approximation (Schraudolph, 1999)
//
// Uses the IEEE 754 float bit layout to approximate 2^x via integer
// arithmetic, then converts base with ln(2). Relative error ~2-4% in
// the range [-10, 10], which is acceptable for inference softmax/silu.
// ---------------------------------------------------------------------------
static inline float fast_expf(float x) {
    // Clamp to avoid overflow/underflow in the integer path
    if (x < -80.0f) return 0.0f;
    if (x >  80.0f) return HUGE_VALF;

    // 2^23 / ln(2) ~= 12102203.16, bias = 127 * 2^23 = 1065353216
    // exp(x) = 2^(x / ln2), and we approximate 2^y by stuffing
    // (y * 2^23 + bias) into the IEEE bits.
    float val = x * 12102203.16f + 1065353216.0f;
    int32_t bits = (val >= 0.0f) ? (int32_t)val : 0;
    float result;
    memcpy(&result, &bits, sizeof(float));
    return result;
}

// ---------------------------------------------------------------------------
// Matrix multiply: C = A * B
// A is M x K, B is K x N, C is M x N, all row-major.
//
// Transpose B into Bt so the inner loop walks contiguous memory for both
// A's row and Bt's row (column of B). This turns a stride-N access pattern
// into stride-1 and is a large win on Broadway's small 32KB L1D cache.
// ---------------------------------------------------------------------------
void wii_matmul(const float* A, const float* B, float* C,
                int M, int K, int N) {
    // Transpose B -> Bt (Bt is N x K, row-major)
    float* Bt = (float*)malloc((size_t)N * K * sizeof(float));
    if (!Bt) {
        // Fallback: naive matmul if allocation fails
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float sum = 0.0f;
                for (int k = 0; k < K; k++) {
                    sum += A[i * K + k] * B[k * N + j];
                }
                C[i * N + j] = sum;
            }
        }
        return;
    }

    for (int j = 0; j < N; j++) {
        for (int k = 0; k < K; k++) {
            Bt[j * K + k] = B[k * N + j];
        }
    }

    for (int i = 0; i < M; i++) {
        const float* a_row = A + i * K;
        for (int j = 0; j < N; j++) {
            const float* bt_row = Bt + j * K;
            float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
            int k = 0;
            int k_end = K - 3;
            for (; k < k_end; k += 4) {
                s0 += a_row[k + 0] * bt_row[k + 0];
                s1 += a_row[k + 1] * bt_row[k + 1];
                s2 += a_row[k + 2] * bt_row[k + 2];
                s3 += a_row[k + 3] * bt_row[k + 3];
            }
            float sum = (s0 + s1) + (s2 + s3);
            for (; k < K; k++) {
                sum += a_row[k] * bt_row[k];
            }
            C[i * N + j] = sum;
        }
    }

    free(Bt);
}

// ---------------------------------------------------------------------------
// Matrix-vector multiply: y = A * x
// A is M x K, x is length K, y is length M.
//
// 4 independent accumulators to break the FP dependency chain. PPC750's
// FPU has ~5-cycle multiply-add latency but can issue one per cycle if
// successive ops are independent.
// ---------------------------------------------------------------------------
void wii_matvec(const float* A, const float* x, float* y,
                int M, int K) {
    for (int i = 0; i < M; i++) {
        const float* a_row = A + i * K;
        float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
        int k = 0;
        int k_end = K - 3;
        for (; k < k_end; k += 4) {
            s0 += a_row[k + 0] * x[k + 0];
            s1 += a_row[k + 1] * x[k + 1];
            s2 += a_row[k + 2] * x[k + 2];
            s3 += a_row[k + 3] * x[k + 3];
        }
        float sum = (s0 + s1) + (s2 + s3);
        for (; k < K; k++) {
            sum += a_row[k] * x[k];
        }
        y[i] = sum;
    }
}

void wii_add(const float* a, const float* b, float* y, int n) {
    for (int i = 0; i < n; i++) {
        y[i] = a[i] + b[i];
    }
}

void wii_mul(const float* a, const float* b, float* y, int n) {
    for (int i = 0; i < n; i++) {
        y[i] = a[i] * b[i];
    }
}

// Branchless ReLU: 0.5 * (x + |x|) avoids a conditional branch on PPC.
void wii_relu(const float* x, float* y, int n) {
    for (int i = 0; i < n; i++) {
        y[i] = 0.5f * (x[i] + fabsf(x[i]));
    }
}

void wii_silu(const float* x, float* y, int n) {
    for (int i = 0; i < n; i++) {
        float sig = 1.0f / (1.0f + expf(-x[i]));
        y[i] = x[i] * sig;
    }
}

void wii_softmax(const float* x, float* y, int n) {
    float max_val = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        y[i] = expf(x[i] - max_val);
        sum += y[i];
    }
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < n; i++) {
        y[i] *= inv_sum;
    }
}

// RMS normalization with 4-way accumulator for the sum-of-squares.
void wii_rmsnorm(const float* x, const float* weight, float* y, int n) {
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    int i = 0;
    int i_end = n - 3;
    for (; i < i_end; i += 4) {
        s0 += x[i + 0] * x[i + 0];
        s1 += x[i + 1] * x[i + 1];
        s2 += x[i + 2] * x[i + 2];
        s3 += x[i + 3] * x[i + 3];
    }
    float ss = (s0 + s1) + (s2 + s3);
    for (; i < n; i++) {
        ss += x[i] * x[i];
    }
    ss = 1.0f / sqrtf(ss / n + 1e-5f);
    for (i = 0; i < n; i++) {
        y[i] = x[i] * ss * weight[i];
    }
}

// INT8 matvec with 4-way unrolled inner loop.
void wii_matvec_q8(const signed char* W, float scale, const float* x,
                   float* y, int M, int K) {
    for (int i = 0; i < M; i++) {
        const signed char* w_row = W + i * K;
        float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
        int k = 0;
        int k_end = K - 3;
        for (; k < k_end; k += 4) {
            s0 += (float)w_row[k + 0] * x[k + 0];
            s1 += (float)w_row[k + 1] * x[k + 1];
            s2 += (float)w_row[k + 2] * x[k + 2];
            s3 += (float)w_row[k + 3] * x[k + 3];
        }
        float sum = (s0 + s1) + (s2 + s3);
        for (; k < K; k++) {
            sum += (float)w_row[k] * x[k];
        }
        y[i] = sum * scale;
    }
}