// Tests and benchmarks ggml_gemm_q4_K_8x8_q8_K against the generic reference.
// Only usable when GGML_BACKEND_DL is off (functions are statically linked).

#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include "repack.h"
#include "quants.h"

#include "ggml-cpu.h"

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// Copied from repack.cpp (static there).
// Interleaves 8 block_q4_K structures into one block_q4_Kx8.
static block_q4_Kx8 pack_q4_Kx8(block_q4_K * in, int blck_size_interleave) {
    block_q4_Kx8 out;

    for (int i = 0; i < 8; i++) out.d[i]    = in[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d;
    for (int i = 0; i < 8; i++) out.dmin[i] = in[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin;

    const int end = QK_K * 4 / blck_size_interleave;
    for (int i = 0; i < end; ++i) {
        int src_id  = i % 8;
        int src_off = (i / 8) * blck_size_interleave;
        int dst_off = i * blck_size_interleave;
        uint64_t elems;
        memcpy(&elems,        &in[src_id].qs[src_off], blck_size_interleave);
        memcpy(&out.qs[dst_off], &elems,               blck_size_interleave);
    }

    uint8_t s[8], m[8];
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 8; j++) {
            s[j] = in[j].scales[i]     & 63;
            m[j] = in[j].scales[i + 4] & 63;
        }
        out.scales[i*12]    = (s[0]&63)+((s[4]&48)<<2);
        out.scales[i*12+1]  = (s[1]&63)+((s[5]&48)<<2);
        out.scales[i*12+2]  = (s[2]&63)+((s[6]&48)<<2);
        out.scales[i*12+3]  = (s[3]&63)+((s[7]&48)<<2);
        out.scales[i*12+4]  = (m[0]&63)+((m[4]&48)<<2);
        out.scales[i*12+5]  = (m[1]&63)+((m[5]&48)<<2);
        out.scales[i*12+6]  = (m[2]&63)+((m[6]&48)<<2);
        out.scales[i*12+7]  = (m[3]&63)+((m[7]&48)<<2);
        out.scales[i*12+8]  = (s[4]&15)+((m[4]&15)<<4);
        out.scales[i*12+9]  = (s[5]&15)+((m[5]&15)<<4);
        out.scales[i*12+10] = (s[6]&15)+((m[6]&15)<<4);
        out.scales[i*12+11] = (s[7]&15)+((m[7]&15)<<4);
    }
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 8; j++) {
            s[j] = ((in[j].scales[i]   & 192) >> 2) | ( in[j].scales[i+8] & 15);
            m[j] = ((in[j].scales[i+4] & 192) >> 2) | ((in[j].scales[i+8] & 240) >> 4);
        }
        out.scales[i*12+48] = (s[0]&63)+((s[4]&48)<<2);
        out.scales[i*12+49] = (s[1]&63)+((s[5]&48)<<2);
        out.scales[i*12+50] = (s[2]&63)+((s[6]&48)<<2);
        out.scales[i*12+51] = (s[3]&63)+((s[7]&48)<<2);
        out.scales[i*12+52] = (m[0]&63)+((m[4]&48)<<2);
        out.scales[i*12+53] = (m[1]&63)+((m[5]&48)<<2);
        out.scales[i*12+54] = (m[2]&63)+((m[6]&48)<<2);
        out.scales[i*12+55] = (m[3]&63)+((m[7]&48)<<2);
        out.scales[i*12+56] = (s[4]&15)+((m[4]&15)<<4);
        out.scales[i*12+57] = (s[5]&15)+((m[5]&15)<<4);
        out.scales[i*12+58] = (s[6]&15)+((m[6]&15)<<4);
        out.scales[i*12+59] = (s[7]&15)+((m[7]&15)<<4);
    }
    return out;
}

// Pack nc rows of block_q4_K (nc*nb blocks, row-major) into block_q4_Kx8 layout.
// Output: (nc/8)*nb block_q4_Kx8, indexed as [col_group * nb + superblock].
static std::vector<block_q4_Kx8> pack_weights(const block_q4_K * src, int nc, int nb) {
    assert(nc % 8 == 0);
    std::vector<block_q4_Kx8> dst((nc / 8) * nb);
    for (int g = 0; g < nc / 8; g++) {
        for (int b = 0; b < nb; b++) {
            block_q4_K tmp[8];
            for (int i = 0; i < 8; i++) tmp[i] = src[(g * 8 + i) * nb + b];
            dst[g * nb + b] = pack_q4_Kx8(tmp, 8);
        }
    }
    return dst;
}

struct TestCase {
    int n;   // inner dim, multiple of QK_K=256
    int nr;  // A rows,  multiple of 4
    int nc;  // B rows (= output cols), multiple of 8
};

static bool test_correctness(const TestCase & tc, bool verbose = false) {
    const int nb = tc.n / QK_K;

    std::vector<float> A(tc.nr * tc.n), B(tc.nc * tc.n);
    for (float & v : A) v = (float)(rand() % 201 - 100) / 50.0f;
    for (float & v : B) v = (float)(rand() % 201 - 100) / 50.0f;

    // Quantize B row-by-row to block_q4_K
    std::vector<block_q4_K> q4k(tc.nc * nb);
    for (int r = 0; r < tc.nc; r++) {
        quantize_row_q4_K(B.data() + r * tc.n, &q4k[r * nb], tc.n);
    }

    std::vector<block_q4_Kx8> vx = pack_weights(q4k.data(), tc.nc, nb);

    // Pack A groups of 4 rows into block_q8_Kx4
    std::vector<block_q8_Kx4> vy((tc.nr / 4) * nb);
    for (int g = 0; g < tc.nr / 4; g++) {
        ggml_quantize_mat_q8_K_4x4(A.data() + g * 4 * tc.n, &vy[g * nb], tc.n);
    }

    std::vector<float> s_ref(tc.nr * tc.nc, 0.0f);
    std::vector<float> s_opt(tc.nr * tc.nc, 0.0f);

    ggml_gemm_q4_K_8x8_q8_K_generic(tc.n, s_ref.data(), tc.nc, vx.data(), vy.data(), tc.nr, tc.nc);
    ggml_gemm_q4_K_8x8_q8_K        (tc.n, s_opt.data(), tc.nc, vx.data(), vy.data(), tc.nr, tc.nc);

    // Compute max absolute value in ref to normalize error
    float max_abs_ref = 0.0f;
    for (int i = 0; i < tc.nr * tc.nc; i++) {
        max_abs_ref = fmaxf(max_abs_ref, fabsf(s_ref[i]));
    }
    const float floor_val = fmaxf(max_abs_ref * 1e-3f, 1e-3f);

    float max_err = 0.0f;
    int   max_idx = 0;
    for (int i = 0; i < tc.nr * tc.nc; i++) {
        float ref = s_ref[i];
        float err = fabsf(s_opt[i] - ref);
        float rel = err / (fabsf(ref) + floor_val);
        if (rel > max_err) { max_err = rel; max_idx = i; }
    }

    bool ok = max_err < 1e-3f;
    printf("  n=%-5d nr=%-3d nc=%-4d  max_rel_err=%.2e  %s\n",
           tc.n, tc.nr, tc.nc, max_err, ok ? "PASS" : "FAIL");

    if (verbose || !ok) {
        int row = max_idx / tc.nc, col = max_idx % tc.nc;
        printf("    worst: s[%d][%d]  ref=%.6f  opt=%.6f\n",
               row, col, s_ref[max_idx], s_opt[max_idx]);
        printf("    first 8 ref: ");
        for (int i = 0; i < 8 && i < tc.nr * tc.nc; i++) printf("%.3f ", s_ref[i]);
        printf("\n    first 8 opt: ");
        for (int i = 0; i < 8 && i < tc.nr * tc.nc; i++) printf("%.3f ", s_opt[i]);
        printf("\n");
    }
    return ok;
}

static void bench(const TestCase & tc, int iters) {
    const int nb = tc.n / QK_K;

    std::vector<float> A(tc.nr * tc.n), B(tc.nc * tc.n);
    for (float & v : A) v = (float)(rand() % 201 - 100) / 50.0f;
    for (float & v : B) v = (float)(rand() % 201 - 100) / 50.0f;

    std::vector<block_q4_K> q4k(tc.nc * nb);
    for (int r = 0; r < tc.nc; r++) {
        quantize_row_q4_K(B.data() + r * tc.n, &q4k[r * nb], tc.n);
    }
    std::vector<block_q4_Kx8> vx = pack_weights(q4k.data(), tc.nc, nb);

    std::vector<block_q8_Kx4> vy((tc.nr / 4) * nb);
    for (int g = 0; g < tc.nr / 4; g++) {
        ggml_quantize_mat_q8_K_4x4(A.data() + g * 4 * tc.n, &vy[g * nb], tc.n);
    }

    std::vector<float> s(tc.nr * tc.nc);

    // Warm up
    ggml_gemm_q4_K_8x8_q8_K(tc.n, s.data(), tc.nc, vx.data(), vy.data(), tc.nr, tc.nc);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++) {
        ggml_gemm_q4_K_8x8_q8_K(tc.n, s.data(), tc.nc, vx.data(), vy.data(), tc.nr, tc.nc);
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    double us      = std::chrono::duration<double, std::micro>(t1 - t0).count();
    double ops     = 2.0 * tc.nr * tc.nc * tc.n * iters;
    double ms_iter = us / iters / 1000.0;
    double gflops  = ops / us / 1e3;
    printf("  n=%-5d nr=%-3d nc=%-4d  %.3f ms/iter  %.1f GFLOPS\n",
           tc.n, tc.nr, tc.nc, ms_iter, gflops);
}

int main() {
    ggml_cpu_init();
    srand(42);

    static const TestCase correctness_cases[] = {
        {  256,  4,   8 },
        {  512,  4,   8 },
        { 4096,  4,   8 },
        { 4096, 16,   8 },
        { 4096, 16,  16 },
        { 4096, 16,  32 },
        { 4096, 64,  64 },
    };

    printf("=== Correctness ===\n");
    bool all_ok = true;
    for (const auto & tc : correctness_cases) {
        all_ok &= test_correctness(tc);
    }

    // Realistic LLM sizes: n=hidden_dim, nr=batch, nc=output_dim
    printf("\n=== Benchmark ===\n");
    bench({ 4096,   4,  4096 }, 500);   // Llama-7B: decode
    bench({ 4096,  32,  4096 }, 200);   // Llama-7B: small prefill
    bench({ 4096, 128,  4096 },  50);   // Llama-7B: larger prefill
    bench({ 4096,   4, 11008 }, 500);   // Llama-7B: FFN gate/up, decode
    bench({ 4096,  32, 11008 }, 200);   // Llama-7B: FFN gate/up, prefill
    bench({ 8192,   4,  8192 }, 200);   // Llama-70B: attn, decode
    bench({ 8192,  32,  8192 },  50);   // Llama-70B: attn, prefill

    return all_ok ? 0 : 1;
}
