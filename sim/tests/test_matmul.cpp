// Tests for wii_math.h
// Runs against whichever backend is compiled in (CPU or GX).
// Validates matmul, matvec, elementwise ops, and activations.

#include "wii_math.h"
#include <cmath>
#include <cstdio>
#include <cstring>

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) static void name()
#define RUN(name) do { \
    printf("  %-50s ", #name); \
    name(); \
    printf("PASS\n"); \
    tests_passed++; \
} while(0)

#define ASSERT_NEAR(a, b, tol) do { \
    if (fabs((double)(a) - (double)(b)) > (tol)) { \
        printf("FAIL\n    %s:%d: %.6f != %.6f (tol=%.6f)\n", \
               __FILE__, __LINE__, (double)(a), (double)(b), (double)(tol)); \
        tests_failed++; \
        return; \
    } \
} while(0)

// Tolerance: CPU backend is exact, GX backend has quantization error from
// mapping floats to 8-bit (1/255 ~ 0.004 per value, accumulates over K).
// For the GX backend with values in [0,1], expect error of about K * 0.004.
#ifdef WII_MATH_USE_GX
static const float TOL = 0.05f;
static const float MATMUL_TOL = 0.1f;  // larger for accumulated quantization
#else
static const float TOL = 1e-5f;
static const float MATMUL_TOL = 1e-5f;
#endif

// 2x3 * 3x2 = 2x2 matmul
TEST(test_matmul_2x3_3x2) {
    // A = [[1, 2, 3],
    //      [4, 5, 6]]
    // B = [[7, 8],
    //      [9, 10],
    //      [11, 12]]
    // C = [[58, 64],
    //      [139, 154]]

#ifdef WII_MATH_USE_GX
    // For GX, scale to [0,1] range
    float A[] = {0.1f, 0.2f, 0.3f,
                 0.4f, 0.5f, 0.6f};
    float B[] = {0.7f, 0.8f,
                 0.9f, 1.0f,
                 0.5f, 0.6f};
    // Expected: C[0][0] = 0.1*0.7 + 0.2*0.9 + 0.3*0.5 = 0.07+0.18+0.15 = 0.40
    //           C[0][1] = 0.1*0.8 + 0.2*1.0 + 0.3*0.6 = 0.08+0.20+0.18 = 0.46
    //           C[1][0] = 0.4*0.7 + 0.5*0.9 + 0.6*0.5 = 0.28+0.45+0.30 = 1.03
    //           C[1][1] = 0.4*0.8 + 0.5*1.0 + 0.6*0.6 = 0.32+0.50+0.36 = 1.18
    float expected[] = {0.40f, 0.46f, 1.03f, 1.18f};
#else
    float A[] = {1, 2, 3, 4, 5, 6};
    float B[] = {7, 8, 9, 10, 11, 12};
    float expected[] = {58, 64, 139, 154};
#endif

    float C[4];
    wii_matmul(A, B, C, 2, 3, 2);

    ASSERT_NEAR(C[0], expected[0], MATMUL_TOL);
    ASSERT_NEAR(C[1], expected[1], MATMUL_TOL);
    ASSERT_NEAR(C[2], expected[2], MATMUL_TOL);
    ASSERT_NEAR(C[3], expected[3], MATMUL_TOL);
}

// Identity matmul: I * A = A
TEST(test_matmul_identity) {
#ifdef WII_MATH_USE_GX
    float I[] = {1.0f, 0.0f,
                 0.0f, 1.0f};
    float A[] = {0.3f, 0.7f,
                 0.5f, 0.2f};
#else
    float I[] = {1, 0, 0, 1};
    float A[] = {3, 7, 5, 2};
#endif

    float C[4];
    wii_matmul(I, A, C, 2, 2, 2);

    ASSERT_NEAR(C[0], A[0], MATMUL_TOL);
    ASSERT_NEAR(C[1], A[1], MATMUL_TOL);
    ASSERT_NEAR(C[2], A[2], MATMUL_TOL);
    ASSERT_NEAR(C[3], A[3], MATMUL_TOL);
}

// Matrix-vector multiply
TEST(test_matvec) {
#ifdef WII_MATH_USE_GX
    float A[] = {0.1f, 0.2f, 0.3f,
                 0.4f, 0.5f, 0.6f};
    float x[] = {0.7f, 0.8f, 0.9f};
    // y[0] = 0.1*0.7 + 0.2*0.8 + 0.3*0.9 = 0.07+0.16+0.27 = 0.50
    // y[1] = 0.4*0.7 + 0.5*0.8 + 0.6*0.9 = 0.28+0.40+0.54 = 1.22
    float expected[] = {0.50f, 1.22f};
#else
    float A[] = {1, 2, 3, 4, 5, 6};
    float x[] = {7, 8, 9};
    float expected[] = {50, 122};
#endif

    float y[2];
    wii_matvec(A, x, y, 2, 3);

    ASSERT_NEAR(y[0], expected[0], MATMUL_TOL);
    ASSERT_NEAR(y[1], expected[1], MATMUL_TOL);
}

// Elementwise add
TEST(test_add) {
#ifdef WII_MATH_USE_GX
    float a[] = {0.1f, 0.2f, 0.3f, 0.4f};
    float b[] = {0.05f, 0.1f, 0.15f, 0.2f};
    float expected[] = {0.15f, 0.3f, 0.45f, 0.6f};
#else
    float a[] = {1, 2, 3, 4};
    float b[] = {5, 6, 7, 8};
    float expected[] = {6, 8, 10, 12};
#endif

    float y[4];
    wii_add(a, b, y, 4);

    for (int i = 0; i < 4; i++) {
        ASSERT_NEAR(y[i], expected[i], TOL);
    }
}

// Elementwise multiply
TEST(test_mul) {
#ifdef WII_MATH_USE_GX
    float a[] = {0.5f, 0.6f, 0.7f, 0.8f};
    float b[] = {0.2f, 0.3f, 0.4f, 0.5f};
    float expected[] = {0.1f, 0.18f, 0.28f, 0.4f};
#else
    float a[] = {2, 3, 4, 5};
    float b[] = {6, 7, 8, 9};
    float expected[] = {12, 21, 32, 45};
#endif

    float y[4];
    wii_mul(a, b, y, 4);

    for (int i = 0; i < 4; i++) {
        ASSERT_NEAR(y[i], expected[i], TOL);
    }
}

// ReLU
TEST(test_relu) {
    float x[] = {-2.0f, -0.5f, 0.0f, 0.5f, 2.0f};
    float expected[] = {0.0f, 0.0f, 0.0f, 0.5f, 2.0f};
    float y[5];
    wii_relu(x, y, 5);

    for (int i = 0; i < 5; i++) {
        ASSERT_NEAR(y[i], expected[i], 1e-5f);
    }
}

// SiLU
TEST(test_silu) {
    float x[] = {0.0f, 1.0f, -1.0f};
    float y[3];
    wii_silu(x, y, 3);

    // silu(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
    ASSERT_NEAR(y[0], 0.0f, 1e-5f);
    // silu(1) = 1 * sigmoid(1) = 1 * 0.7311 = 0.7311
    ASSERT_NEAR(y[1], 0.7311f, 0.001f);
    // silu(-1) = -1 * sigmoid(-1) = -1 * 0.2689 = -0.2689
    ASSERT_NEAR(y[2], -0.2689f, 0.001f);
}

// Softmax
TEST(test_softmax) {
    float x[] = {1.0f, 2.0f, 3.0f};
    float y[3];
    wii_softmax(x, y, 3);

    // Sum should be 1.0
    float sum = y[0] + y[1] + y[2];
    ASSERT_NEAR(sum, 1.0f, 1e-5f);
    // Ordering should be preserved
    if (!(y[0] < y[1] && y[1] < y[2])) {
        printf("FAIL\n    softmax ordering violated\n");
        tests_failed++;
        return;
    }
    // Known values
    ASSERT_NEAR(y[0], 0.0900f, 0.001f);
    ASSERT_NEAR(y[1], 0.2447f, 0.001f);
    ASSERT_NEAR(y[2], 0.6652f, 0.001f);
}

// INT8 matvec with fused dequant
TEST(test_matvec_q8) {
    // W = [[1, 2, 3], [4, 5, 6]] as int8, scale = 0.1
    // So dequanted W = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    // x = [7, 8, 9]
    // y[0] = 0.1*7 + 0.2*8 + 0.3*9 = 0.7 + 1.6 + 2.7 = 5.0
    // y[1] = 0.4*7 + 0.5*8 + 0.6*9 = 2.8 + 4.0 + 5.4 = 12.2
    signed char W[] = {1, 2, 3, 4, 5, 6};
    float scale = 0.1f;
    float x[] = {7.0f, 8.0f, 9.0f};
    float y[2];
    wii_matvec_q8(W, scale, x, y, 2, 3);

    ASSERT_NEAR(y[0], 5.0f, 0.001f);
    ASSERT_NEAR(y[1], 12.2f, 0.001f);
}

// INT8 matvec: verify scale factor
TEST(test_matvec_q8_scale) {
    // W = [127, -128] as int8, scale = 0.01
    // dequanted = [1.27, -1.28]
    // x = [1.0, 1.0]
    // y[0] = 1.27 + (-1.28) = -0.01
    signed char W[] = {127, -128};
    float scale = 0.01f;
    float x[] = {1.0f, 1.0f};
    float y[1];
    wii_matvec_q8(W, scale, x, y, 1, 2);

    ASSERT_NEAR(y[0], -0.01f, 0.001f);
}

// RMS normalization
TEST(test_rmsnorm) {
    float x[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float w[] = {1.0f, 1.0f, 1.0f, 1.0f};
    float y[4];
    wii_rmsnorm(x, w, y, 4);

    // rms = sqrt((1+4+9+16)/4) = sqrt(7.5) = 2.7386
    // ss = 1/sqrt(7.5 + 1e-5) = 0.36514
    float ss = 1.0f / sqrtf(7.5f + 1e-5f);
    ASSERT_NEAR(y[0], 1.0f * ss, 1e-4f);
    ASSERT_NEAR(y[1], 2.0f * ss, 1e-4f);
    ASSERT_NEAR(y[2], 3.0f * ss, 1e-4f);
    ASSERT_NEAR(y[3], 4.0f * ss, 1e-4f);
}

int main() {
#ifdef WII_MATH_USE_GX
    printf("test_matmul (GX backend):\n");
#else
    printf("test_matmul (CPU backend):\n");
#endif

    wii_math_init();

    RUN(test_matmul_2x3_3x2);
    RUN(test_matmul_identity);
    RUN(test_matvec);
    RUN(test_add);
    RUN(test_mul);
    RUN(test_relu);
    RUN(test_silu);
    RUN(test_softmax);
    RUN(test_matvec_q8);
    RUN(test_matvec_q8_scale);
    RUN(test_rmsnorm);

    wii_math_shutdown();

    printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
