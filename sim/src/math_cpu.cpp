// CPU reference implementation of wii_math.h
// Naive loops, no SIMD. This is the ground truth for validating the GX backend.

#include "wii_math.h"
#include <cmath>
#include <cstring>

void wii_math_init(void) {}

void wii_math_shutdown(void) {}

// TODO upgrade this to use ggml/microBLAS backend if perf is an issue
void wii_matmul(const float* A, const float* B, float* C,
                int M, int K, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

void wii_matvec(const float* A, const float* x, float* y,
                int M, int K) {
    for (int i = 0; i < M; i++) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[i * K + k] * x[k];
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

void wii_relu(const float* x, float* y, int n) {
    for (int i = 0; i < n; i++) {
        y[i] = x[i] > 0.0f ? x[i] : 0.0f;
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
    for (int i = 0; i < n; i++) {
        y[i] /= sum;
    }
}

void wii_rmsnorm(const float* x, const float* weight, float* y, int n) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++) {
        ss += x[i] * x[i];
    }
    ss = 1.0f / sqrtf(ss / n + 1e-5f);
    for (int i = 0; i < n; i++) {
        y[i] = x[i] * ss * weight[i];
    }
}

void wii_matvec_q8(const signed char* W, float scale, const float* x,
                   float* y, int M, int K) {
    for (int i = 0; i < M; i++) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += (float)W[i * K + k] * x[k];
        }
        y[i] = sum * scale;
    }
}
