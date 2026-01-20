//============================================================
// UniversalInterpolator_GPU_Periodic.cu
// Complete CUDA Interpolator (wrap-safe, oceanized, normalized)
// Author: Mark Devereux (ApotheCode R&D)
// Date: 2025-11-10
//============================================================

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
#include <string>
#include <algorithm>
#include <cassert>

//============================================================
// CONSTANTS
//============================================================
constexpr int INPUT_W  = 1440;
constexpr int INPUT_H  = 720;
constexpr int OUTPUT_W = 3600;
constexpr int OUTPUT_H = 1800;
constexpr int MONTHS   = 12;
constexpr int THREADS  = 256;

//============================================================
// DEVICE HELPERS
//============================================================
__device__ __forceinline__ int wrapX(int x, int W) {
    int t = x % W; return t < 0 ? t + W : t;
}
__device__ __forceinline__ int clampY(int y, int H) {
    return y < 0 ? 0 : (y >= H ? H - 1 : y);
}
__device__ __forceinline__ double lerp(double a, double b, double t) {
    return a + (b - a) * t;
}
__device__ __forceinline__ double cosKernel(double t) {
    return 0.5 * (1.0 - cos(2.0 * M_PI * t));
}

//============================================================
// KERNEL: Bilinear Upscaling (periodic longitude wrap)
//============================================================
__global__ void bilinearUpscaleKernel(const double* __restrict__ in,
                                      double* __restrict__ out,
                                      int W, int H, int W2, int H2) {
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox >= W2 || oy >= H2) return;

    double sx = ((ox + 0.5) * (double)W / (double)W2) - 0.5;
    double sy = ((oy + 0.5) * (double)H / (double)H2) - 0.5;

    int x0 = (int)floor(sx), y0 = (int)floor(sy);
    int x1 = x0 + 1, y1 = y0 + 1;
    double tx = sx - x0, ty = sy - y0;

    x0 = wrapX(x0, W); x1 = wrapX(x1, W);
    y0 = clampY(y0, H); y1 = clampY(y1, H);

    double p00 = in[y0 * W + x0];
    double p10 = in[y0 * W + x1];
    double p01 = in[y1 * W + x0];
    double p11 = in[y1 * W + x1];

    double a = lerp(p00, p10, tx);
    double b = lerp(p01, p11, tx);
    out[oy * W2 + ox] = lerp(a, b, ty);
}

//============================================================
// KERNEL: Cosine-weighted radial fill (for NaNs / gaps)
//============================================================
__global__ void fillMissingKernel(const double* in, const uint8_t* valid,
                                  double* out, int W, int H, int Rmax) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    int idx = y * W + x;
    if (valid[idx]) { out[idx] = in[idx]; return; }

    double wsum = 0.0, vsum = 0.0;
    for (int r = 1; r <= Rmax; ++r) {
        int samples = max(8, 8 * r);
        for (int k = 0; k < samples; ++k) {
            double ang = (2.0 * M_PI * k) / samples;
            int xi = wrapX(x + (int)round(r * cos(ang)), W);
            int yi = clampY(y + (int)round(r * sin(ang)), H);
            int j = yi * W + xi;
            if (!valid[j]) continue;
            double w = 0.5 * (1.0 + cos(M_PI * r / (double)Rmax));
            wsum += w; vsum += w * in[j];
        }
        if (wsum > 0.0) break;
    }
    out[idx] = (wsum > 0.0) ? (vsum / wsum) : 0.0;
}

//============================================================
// KERNEL: Temporal cosine smoothing (circular months)
//============================================================
__global__ void temporalSmoothKernel(const double* in,
                                     double* out, int cells) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= cells) return;
    double sm[MONTHS];
    for (int m = 0; m < MONTHS; ++m) {
        double num = 0, den = 0;
        for (int k = -2; k <= 2; ++k) {
            int mm = (m + k + MONTHS) % MONTHS;
            double w = cosKernel((k + 2) / 4.0);
            num += w * in[c * MONTHS + mm];
            den += w;
        }
        sm[m] = num / den;
    }
    for (int m = 0; m < MONTHS; ++m)
        out[c * MONTHS + m] = sm[m];
}

//============================================================
// KERNEL: Robust normalization (min/max clamp)
//============================================================
__global__ void normalizeKernel(double* data, int n, double vmin, double vmax) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    double v = data[idx];
    v = (v - vmin) / (vmax - vmin);
    v = fmin(1.0, fmax(0.0, v));
    data[idx] = v;
}

//============================================================
// KERNEL: Seam validation
//============================================================
__global__ void seamMaxKernel(const double* f, int W, int H, double* maxOut) {
    __shared__ double smax[THREADS];
    int tid = threadIdx.x;
    double localMax = 0.0;
    for (int y = tid; y < H; y += blockDim.x) {
        double d = fabs(f[y * W + 0] - f[y * W + (W - 1)]);
        if (d > localMax) localMax = d;
    }
    smax[tid] = localMax;
    __syncthreads();
    // block reduce
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            smax[tid] = fmax(smax[tid], smax[tid + stride]);
        __syncthreads();
    }
    if (tid == 0)
        atomicMax((unsigned long long*)maxOut, __double_as_longlong(smax[0]));
}

//============================================================
// HOST UTILITIES
//============================================================
inline void checkSeamError(double* d_field, int W, int H, const char* tag) {
    double* d_max; double h_max = 0;
    cudaMalloc(&d_max, sizeof(double)); cudaMemset(d_max, 0, sizeof(double));
    seamMaxKernel<<<1, THREADS>>>(d_field, W, H, d_max);
    cudaMemcpy(&h_max, d_max, sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_max);
    if (h_max > 1e-6) {
        std::cerr << "❌ Seam error in " << tag << " : " << h_max << "\n";
        exit(EXIT_FAILURE);
    } else {
        std::cout << "✅ Seam continuity OK (" << tag << ")\n";
    }
}

//============================================================
// MAIN PIPELINE
//============================================================
int main() {
    std::cout << "=== Universal Interpolator (Periodic/Ocean-Safe) ===\n";

    size_t inputCells  = INPUT_W * INPUT_H;
    size_t outputCells = OUTPUT_W * OUTPUT_H;
    size_t bytesIn  = inputCells  * sizeof(double);
    size_t bytesOut = outputCells * sizeof(double);

    std::vector<double> h_in(inputCells);
    std::vector<double> h_out(outputCells);
    // (for demonstration — replace with actual CSV loader)
    for (int i = 0; i < inputCells; ++i)
        h_in[i] = sin((i % INPUT_W) * 2 * M_PI / INPUT_W);

    double* d_in; double* d_out;
    cudaMalloc(&d_in, bytesIn);
    cudaMalloc(&d_out, bytesOut);
    cudaMemcpy(d_in, h_in.data(), bytesIn, cudaMemcpyHostToDevice);

    // --- Upscale
    dim3 block(32, 8);
    dim3 grid((OUTPUT_W + block.x - 1) / block.x,
              (OUTPUT_H + block.y - 1) / block.y);
    bilinearUpscaleKernel<<<grid, block>>>(d_in, d_out,
                                           INPUT_W, INPUT_H,
                                           OUTPUT_W, OUTPUT_H);
    cudaDeviceSynchronize();
    checkSeamError(d_out, OUTPUT_W, OUTPUT_H, "Upscaled Field");

    // --- Normalize (simple 0–1 demo; in practice compute percentiles)
    double vmin = -1.0, vmax = 1.0;
    normalizeKernel<<<(outputCells + THREADS - 1) / THREADS, THREADS>>>(d_out, outputCells, vmin, vmax);
    cudaDeviceSynchronize();

    // --- Download result
    cudaMemcpy(h_out.data(), d_out, bytesOut, cudaMemcpyDeviceToHost);

    // --- Write binary
    std::ofstream fout("OceanWorld_Climate.bin", std::ios::binary);
    fout.write(reinterpret_cast<char*>(h_out.data()), bytesOut);
    fout.close();

    cudaFree(d_in); cudaFree(d_out);
    std::cout << "✅ OceanWorld_Climate.bin generated successfully.\n";
    return 0;
}
