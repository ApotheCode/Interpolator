//============================================================
// UniversalInterpolator_GPU_Periodic.cu
// Full CUDA pipeline: wrap-safe upscale → fill → temporal smooth →
// on-GPU percentile normalization, with multi-stream execution.
// Author: ApotheCode R&D (Mark Devereux)
// Date: 2025-11-10
//============================================================

#include <cuda_runtime.h>
#include <io.h>
#include <share.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <cassert>

// -----------------------------
// CONFIG (match your project)
// -----------------------------
constexpr int IN_W   = 1440;
constexpr int IN_H   = 720;
constexpr int OUT_W  = 3600;
constexpr int OUT_H  = 1800;
constexpr int MONTHS = 12;
constexpr int VARS   = 14;     // total variables processed
constexpr int STREAMS= 4;      // concurrent streams (fits < ~12 GB)
constexpr double M_PI = 3.14159265358979323846;

// Histogram settings for percentile normalization
constexpr int HIST_BINS = 2048;
constexpr double P_LO = 0.01;   // 1st percentile
constexpr double P_HI = 0.99;   // 99th percentile

// CUDA launch helpers
#define CUDA_OK(stmt) do { cudaError_t e = (stmt); if(e!=cudaSuccess){ \
  fprintf(stderr,"CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1);} } while(0)

inline size_t cells_in()  { return (size_t)IN_W * IN_H; }
inline size_t cells_out() { return (size_t)OUT_W * OUT_H; }
inline size_t bytes_in()  { return cells_in()  * sizeof(double); }
inline size_t bytes_out() { return cells_out() * sizeof(double); }
inline size_t bytes_out_all_months() { return (size_t)MONTHS * bytes_out(); }

__global__ void enforcePeriodicX(double* img, int W, int H){
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= H) return;
    double v0 = img[y*W + 0];
    img[y*W + (W-1)] = v0; // copy left edge into right edge
}

//============================================================
// .bin writer
//============================================================
struct BinWriter {
    FILE* f = nullptr;
    size_t cells = 0;

    bool open(const char* path, size_t cells_out){
        cells = cells_out;
        // open binary, no share deny (Windows)
        errno_t e = _fsopen_s(&f, path, "wb", _SH_DENYNO);
        return (e==0 && f);
    }
    void close(){ if (f) { fflush(f); fclose(f); f=nullptr; } }

    bool writeVarMonth(int v, int m, const double* data){
        // Layout: [cell][variable][month]
        // offset = ((v*MONTHS) + m) * cells * sizeof(double)
        unsigned long long off = (unsigned long long)(( (size_t)v * MONTHS + (size_t)m )
                                 * cells * sizeof(double));
        if (_fseeki64(f, (__int64)off, SEEK_SET) != 0) return false;
        size_t n = fwrite(data, sizeof(double), cells, f);
        return n == cells;
    }
};


//============================================================
// DEVICE HELPERS
//============================================================
__device__ __forceinline__ int wrapX(int x, int W){
    int t = x % W; return t < 0 ? t + W : t;
}
__device__ __forceinline__ int clampY(int y, int H){
    return y < 0 ? 0 : (y >= H ? H-1 : y);
}
__device__ __forceinline__ double lerp(double a, double b, double t){
    return a + (b - a) * t;
}
__device__ __forceinline__ double cosKernel(double t){ // t in [0,1]
    return 0.5 * (1.0 - cos(2.0 * M_PI * t));
}

//============================================================
// UPSCALE (bilinear; periodic in X)
// Layout per month: input [y][x] → output [y][x]
//============================================================
__global__ void bilinearUpscaleKernel(const double* __restrict__ in,
                                      double* __restrict__ out,
                                      int W, int H, int W2, int H2){
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox >= W2 || oy >= H2) return;

    double sx = ((ox + 0.5) * (double)W / (double)W2);
    if (sx >= (double)W) sx -= (double)W;
    double sy = ((oy + 0.5) * (double)H / (double)H2) - 0.5;

    int x0 = (int)floor(sx), y0 = (int)floor(sy);
    int x1 = x0 + 1,         y1 = y0 + 1;
    double tx = sx - x0,     ty = sy - y0;

    x0 = wrapX(x0, W); x1 = wrapX(x1, W);
    y0 = clampY(y0, H); y1 = clampY(y1, H);

    double p00 = in[y0*W + x0];
    double p10 = in[y0*W + x1];
    double p01 = in[y1*W + x0];
    double p11 = in[y1*W + x1];

    double a = lerp(p00, p10, tx);
    double b = lerp(p01, p11, tx);
    out[oy*W2 + ox] = lerp(a, b, ty);
}

//============================================================
// FILL MISSING (cosine-taper radial; periodic in X)
// valid==1 uses value; valid==0 fills from neighbors
//============================================================
__global__ void fillMissingKernel(const double* in, const uint8_t* valid,
                                  double* out, int W, int H, int Rmax){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    int idx = y*W + x;
    if (valid[idx]) { out[idx] = in[idx]; return; }

    double wsum=0.0, vsum=0.0;
    for (int r=1; r<=Rmax; ++r){
        int samples = max(8, 8*r);
        for (int k=0; k<samples; ++k){
            double ang = (2.0*M_PI*k)/samples;
            int xi = wrapX(x + (int)llrint(r*cos(ang)), W);
            int yi = clampY(y + (int)llrint(r*sin(ang)), H);
            int j = yi*W + xi;
            if (!valid[j]) continue;
            double w = 0.5*(1.0 + cos(M_PI * (double)r / (double)Rmax));
            wsum += w; vsum += w * in[j];
        }
        if (wsum > 0.0) break;
    }
    out[idx] = (wsum>0.0) ? (vsum/wsum) : 0.0;
}

//============================================================
// TEMPORAL SMOOTH (circular 12 months)
// Input/Output layout: [cell][month] contiguous by month
//============================================================
__global__ void temporalSmoothKernel(const double* in, double* out, int cells){
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= cells) return;
    double sm[MONTHS];
    #pragma unroll
    for (int m=0; m<MONTHS; ++m){
        double num=0.0, den=0.0;
        #pragma unroll
        for (int k=-2; k<=2; ++k){
            int mm = (m + k + MONTHS) % MONTHS;
            double w = cosKernel((k+2)/4.0);
            num += w * in[c*MONTHS + mm];
            den += w;
        }
        sm[m] = num/den;
    }
    #pragma unroll
    for (int m=0; m<MONTHS; ++m)
        out[c*MONTHS + m] = sm[m];
}

//============================================================
// REDUCTIONS for hist/percentiles
// 1) min/max over big array
//============================================================
__global__ void reduceMinMaxKernel(const double* a, size_t n, double* outMin, double* outMax){
    __shared__ double smin[256], smax[256];
    double lmin = INFINITY, lmax = -INFINITY;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x){
        double v = a[i];
        lmin = fmin(lmin, v);
        lmax = fmax(lmax, v);
    }
    smin[threadIdx.x] = lmin; smax[threadIdx.x] = lmax;
    __syncthreads();
    for (int s=blockDim.x/2; s>0; s>>=1){
        if (threadIdx.x < s){
            smin[threadIdx.x] = fmin(smin[threadIdx.x], smin[threadIdx.x + s]);
            smax[threadIdx.x] = fmax(smax[threadIdx.x], smax[threadIdx.x + s]);
        }
        __syncthreads();
    }
    if (threadIdx.x==0){
        atomicMin((unsigned long long*)outMin, __double_as_longlong(smin[0]));
        atomicMax((unsigned long long*)outMax, __double_as_longlong(smax[0]));
    }
}

//============================================================
// 2) histogram over [vmin, vmax] into HIST_BINS bins
// (warp-aggregated via shared memory stripes)
//============================================================
__global__ void histogramKernel(const double* a, size_t n, double vmin, double vmax, unsigned int* hist){
    __shared__ unsigned int sh[HIST_BINS];
    for (int i=threadIdx.x; i<HIST_BINS; i+=blockDim.x) sh[i]=0u;
    __syncthreads();

    double scale = (HIST_BINS-1) / (vmax - vmin + 1e-18);
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x){
        double v = a[i];
        int bin = (int)((v - vmin) * scale);
        bin = bin < 0 ? 0 : (bin >= HIST_BINS ? HIST_BINS-1 : bin);
        atomicAdd(&sh[bin], 1u);
    }
    __syncthreads();
    for (int i=threadIdx.x; i<HIST_BINS; i+=blockDim.x){
        atomicAdd(&hist[i], sh[i]);
    }
}

//============================================================
// 3) Prefix-scan (exclusive) of histogram (single-block scan)
//============================================================
__global__ void prefixScanExclusive(unsigned int* hist, unsigned int* cdf){
    __shared__ unsigned int temp[HIST_BINS];
    int t = threadIdx.x;
    if (t < HIST_BINS) temp[t] = hist[t];
    __syncthreads();

    // Blelloch scan
    for (int offset=1; offset < HIST_BINS; offset<<=1){
        unsigned int val = 0;
        if (t >= offset) val = temp[t - offset];
        __syncthreads();
        if (t < HIST_BINS) temp[t] += val;
        __syncthreads();
    }
    if (t < HIST_BINS) cdf[t] = (t==0 ? 0u : temp[t-1]);
}

//============================================================
// 4) Determine P1, P99 thresholds from CDF on device
//============================================================
__global__ void pickPercentilesKernel(const unsigned int* cdf,
                                      unsigned int total,
                                      double vmin, double vmax,
                                      double* outLo, double* outHi){
    double scale = (vmax - vmin) / (HIST_BINS - 1);
    unsigned int idxLo = 0, idxHi = HIST_BINS-1;
    unsigned int targetLo = (unsigned int)(P_LO * total);
    unsigned int targetHi = (unsigned int)(P_HI * total);

    // linear search is fine (HIST_BINS small); could binary search
    for (int i=0; i<HIST_BINS; ++i){
        unsigned int c = cdf[i];
        if (c <= targetLo) idxLo = i;
        if (c <= targetHi) idxHi = i;
    }
    *outLo = vmin + idxLo * scale;
    *outHi = vmin + idxHi * scale;
}

//============================================================
// NORMALIZE to [0,1] with clamp
//============================================================
__global__ void normalizeKernel(double* a, size_t n, double lo, double hi){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int)n) return;
    double v = a[i];
    double t = (v - lo) / (hi - lo + 1e-18);
    t = fmin(1.0, fmax(0.0, t));
    a[i] = t;
}

//============================================================
// Seam validation (max abs diff of col0 vs colW-1)
//============================================================
__global__ void seamMaxKernel(const double* f, int W, int H, double* outMax){
    __shared__ double sdata[256];
    int tid = threadIdx.x;
    double local = 0.0;
    for (int y = tid; y < H; y += blockDim.x){
        double d = fabs(f[y*W + 0] - f[y*W + (W-1)]);
        if (d > local) local = d;
    }
    sdata[tid] = local;
    __syncthreads();
    for (int s=blockDim.x/2; s>0; s>>=1){
        if (tid < s) sdata[tid] = fmax(sdata[tid], sdata[tid+s]);
        __syncthreads();
    }
    if (tid == 0) *outMax = sdata[0];
}

inline void seamAssert(cudaStream_t st, const double* d, const char* tag){
    double *dmax; double hmax=0;
    CUDA_OK(cudaMallocAsync(&dmax, sizeof(double), st));
    seamMaxKernel<<<1,256,0,st>>>(d, OUT_W, OUT_H, dmax);
    CUDA_OK(cudaMemcpyAsync(&hmax, dmax, sizeof(double), cudaMemcpyDeviceToHost, st));
    CUDA_OK(cudaStreamSynchronize(st));
    CUDA_OK(cudaFreeAsync(dmax, st));
    if (hmax > 1e-12){ // tighten now that we enforce
        fprintf(stderr, "❌ Seam discontinuity (%s): %.9f\n", tag, hmax);
        exit(2);
    } else {
        printf("✅ Seam OK: %s (%.3g)\n", tag, hmax);
    }
}

//============================================================
// HOST PIPELINE (per stream buffers)
// Buffers per stream (to keep <~12 GB total):
//  - d_in_month[12]      : IN_W*IN_H doubles per month (~11.5 MB each)
//  - d_upscaled_month[12]: OUT_W*OUT_H doubles per month (~51.8 MB each)
//  - d_work_month[12]    : OUT_W*OUT_H doubles per month (same size)
//  - d_valid_month[12]   : OUT_W*OUT_H uint8 per month (optional)
//  - histogram/cdf scratch
//============================================================
struct StreamBuffers {
    // Input (packed months): we use one big slab per category for simplicity
    double* d_in[MONTHS];
    double* d_up[MONTHS];
    double* d_work[MONTHS];
    uint8_t* d_valid[MONTHS];   // if you have a validity mask; else nullptr

    // Normalization scratch
    double *d_min, *d_max;
    unsigned int *d_hist, *d_cdf;
    double *d_lo, *d_hi;
};

static void alloc_stream_buffers(StreamBuffers& sb, cudaStream_t st){
    for (int m=0;m<MONTHS;++m){
        CUDA_OK(cudaMallocAsync(&sb.d_in[m],   bytes_in(),  st));
        CUDA_OK(cudaMallocAsync(&sb.d_up[m],   bytes_out(), st));
        CUDA_OK(cudaMallocAsync(&sb.d_work[m], bytes_out(), st));
        CUDA_OK(cudaMallocAsync(&sb.d_valid[m], cells_out()*sizeof(uint8_t), st));
        // default: mark all valid (ocean world). Fill once here.
        CUDA_OK(cudaMemsetAsync(sb.d_valid[m], 1, cells_out()*sizeof(uint8_t), st));
    }
    CUDA_OK(cudaMallocAsync(&sb.d_min, sizeof(double), st));
    CUDA_OK(cudaMallocAsync(&sb.d_max, sizeof(double), st));
    CUDA_OK(cudaMallocAsync(&sb.d_hist, HIST_BINS*sizeof(unsigned int), st));
    CUDA_OK(cudaMallocAsync(&sb.d_cdf, HIST_BINS*sizeof(unsigned int), st));
    CUDA_OK(cudaMallocAsync(&sb.d_lo, sizeof(double), st));
    CUDA_OK(cudaMallocAsync(&sb.d_hi, sizeof(double), st));
}

static void free_stream_buffers(StreamBuffers& sb, cudaStream_t st){
    for (int m=0;m<MONTHS;++m){
        CUDA_OK(cudaFreeAsync(sb.d_in[m],   st));
        CUDA_OK(cudaFreeAsync(sb.d_up[m],   st));
        CUDA_OK(cudaFreeAsync(sb.d_work[m], st));
        CUDA_OK(cudaFreeAsync(sb.d_valid[m], st));
    }
    CUDA_OK(cudaFreeAsync(sb.d_min, st));
    CUDA_OK(cudaFreeAsync(sb.d_max, st));
    CUDA_OK(cudaFreeAsync(sb.d_hist, st));
    CUDA_OK(cudaFreeAsync(sb.d_cdf, st));
    CUDA_OK(cudaFreeAsync(sb.d_lo, st));
    CUDA_OK(cudaFreeAsync(sb.d_hi, st));
}

// ------------------------------------------------------------
// TODO: Wire these to your real disk I/O.
//  - loadInputMonth: fills h_in with IN_W*IN_H doubles
//  - storeOutputMonth: writes OUT_W*OUT_H doubles (normalized 0..1)
// ------------------------------------------------------------
static bool loadInputMonth(int variable, int month, std::vector<double>& h_in){
    // TODO: Replace with real loader (CSV/NETCDF/BIN). Return false on failure.
    // Placeholder: deterministic pattern (for testing seam + pipeline).
    h_in.resize(cells_in());
    for (size_t y=0;y<IN_H;++y){
        for(size_t x=0;x<IN_W;++x){
            double v = 0.6*sin(2*M_PI*(double)x/IN_W) + 0.4*cos(4*M_PI*(double)y/IN_H) + 0.1*variable + 0.02*month;
            h_in[y*IN_W + x] = v;
        }
    }
    return true;
}

BinWriter writer;
if (!writer.open("OceanWorld_Climate.bin", cells_out())){
    fprintf(stderr, "Failed to open OceanWorld_Climate.bin for writing\n");
    return 5;
}
...
// After normalization, when you have h_out for each month:
for (int m=0;m<MONTHS;++m){
    CUDA_OK(cudaMemcpyAsync(h_out.data(), buf.d_up[m], bytes_out(),
                            cudaMemcpyDeviceToHost, st));
    CUDA_OK(cudaStreamSynchronize(st));
    if (!writer.writeVarMonth(v, m, h_out.data())){
        fprintf(stderr, "Write failed (var=%d, month=%d)\n", v, m+1);
        return 6;
    }
}
...
writer.close();
printf("✅ Wrote %dx%dx12x14 doubles to OceanWorld_Climate.bin\n", OUT_W, OUT_H);


//============================================================
// MAIN
//============================================================
int main(){
    printf("=== Universal Interpolator (Periodic, Percentile-Norm, Multi-Stream) ===\n");

    // Create streams & buffers
    cudaStream_t streams[STREAMS];
    StreamBuffers sb[STREAMS];
    for (int s=0;s<STREAMS;++s){
        CUDA_OK(cudaStreamCreateWithFlags(&streams[s], cudaStreamNonBlocking));
        alloc_stream_buffers(sb[s], streams[s]);
    }

    // Host staging buffers for async H2D/D2H
    std::vector<double> h_in, h_out;
    h_out.resize(cells_out());

    // Launch geometry
    dim3 blk2D(32, 8);
    dim3 grdIn ( (IN_W  + blk2D.x-1)/blk2D.x, (IN_H  + blk2D.y-1)/blk2D.y );
    dim3 grdOut( (OUT_W + blk2D.x-1)/blk2D.x, (OUT_H + blk2D.y-1)/blk2D.y );
    dim3 blk1D(256);
    dim3 grd1D( (int)((cells_out()* (size_t)MONTHS + blk1D.x - 1) / blk1D.x) );

    // Process all variables with a rolling window of STREAMS streams
    for (int v=0; v<VARS; ++v){
        int s = v % STREAMS;
        auto& st = streams[s];
        auto& buf = sb[s];

        // --- 1) Load all months for this variable (H→D), upscale (D), optional fill (D)
        for (int m=0;m<MONTHS;++m){
            // H2D load
            if (!loadInputMonth(v, m, h_in)){
                fprintf(stderr,"Load failed: var=%d month=%d\n", v, m+1);
                return 3;
            }
            CUDA_OK(cudaMemcpyAsync(buf.d_in[m], h_in.data(), bytes_in(), cudaMemcpyHostToDevice, st));

            // Upscale IN → UP (wrap-safe)
            bilinearUpscaleKernel<<<grdOut, blk2D, 0, st>>>(buf.d_in[m], buf.d_up[m], IN_W, IN_H, OUT_W, OUT_H);

            // (Optional) If you have real missing-mask per month, run fillMissing here:
            // fillMissingKernel<<<grdOut, blk2D, 0, st>>>(buf.d_up[m], buf.d_valid[m], buf.d_work[m], OUT_W, OUT_H, /*Rmax=*/150);
            // swap to d_up[m] for uniformity:
            // std::swap(buf.d_up[m], buf.d_work[m]);
        }

        // OLD (single check on month 0)
        // seamAssert(st, buf.d_up[0], "Upscale");

        // NEW (inside the loop over months)
        for (int m=0; m<MONTHS; ++m){
            bilinearUpscaleKernel<<<grdOut, blk2D, 0, st>>>(buf.d_in[m], buf.d_up[m],
                                                            IN_W, IN_H, OUT_W, OUT_H);
            enforcePeriodicX<<< (OUT_H+255)/256, 256, 0, st >>>(buf.d_up[m], OUT_W, OUT_H);
            seamAssert(st, buf.d_up[m], "Post-Upscale");
}
        // --- 2) Pack months into [cell][month] layout for temporal smoothing
        // We'll reuse d_work as the packed layout buffer, then unpack back if needed.
        // Pack: contiguous by month already; we can smooth month-wise per cell by reading strided.
        // Implement a pack-free smoother by launching MONTHS independent kernels across cells:
        // Convert to a contiguous [cell][month] view by alias arithmetic.

        // Allocate a contiguous slab [cell][month] for in/out
        double *d_series_in, *d_series_out;
        size_t seriesBytes = (size_t)cells_out() * MONTHS * sizeof(double);
        CUDA_OK(cudaMallocAsync(&d_series_in,  seriesBytes, st));
        CUDA_OK(cudaMallocAsync(&d_series_out, seriesBytes, st));

        // Interleave months into d_series_in
        // (Could fuse earlier, but this is clearer.)
        for (int m=0;m<MONTHS;++m){
            // Copy each month to offset m
            CUDA_OK(cudaMemcpyAsync(d_series_in + (size_t)m*cells_out(),
                                    buf.d_up[m],
                                    bytes_out(), cudaMemcpyDeviceToDevice, st));
        }

        // Temporal smoothing (circular)
        temporalSmoothKernel<<< (int)((cells_out()+blk1D.x-1)/blk1D.x), blk1D, 0, st >>>(d_series_in, d_series_out, (int)cells_out());

            for (int m=0; m<MONTHS; ++m){
        cudaMemcpyAsync(buf.d_up[m],
                        d_series_out + (size_t)m*cells_out(),
                        bytes_out(), cudaMemcpyDeviceToDevice, st);
        enforcePeriodicX<<< (OUT_H+255)/256, 256, 0, st >>>(buf.d_up[m], OUT_W, OUT_H);
        seamAssert(st, buf.d_up[m], "Post-Temporal");
    }

        // Unpack back to per-month d_up[m]
        for (int m=0;m<MONTHS;++m){
            CUDA_OK(cudaMemcpyAsync(buf.d_up[m],
                                    d_series_out + (size_t)m*cells_out(),
                                    bytes_out(), cudaMemcpyDeviceToDevice, st));
        }
        CUDA_OK(cudaFreeAsync(d_series_in, st));
        CUDA_OK(cudaFreeAsync(d_series_out, st));

        // --- 3) Robust percentile normalization across ALL months for this variable
        // Concatenate virtually: we iterate over months in the kernels (launch over n = cells_out()*MONTHS)
        size_t N = (size_t)cells_out() * MONTHS;

        // Find min/max
        CUDA_OK(cudaMemsetAsync(buf.d_min, 0x7F, sizeof(double), st));   // +INF pattern
        CUDA_OK(cudaMemsetAsync(buf.d_max, 0x80, sizeof(double), st));   // -INF pattern

        // Create a temporary alias pointer over 12 slabs: we scan month by month
        // For simplicity, do 12 launches and reduce into d_min/d_max
        int redBlocks = 256;
        for (int m=0;m<MONTHS;++m){
            reduceMinMaxKernel<<<redBlocks, 256, 0, st>>>(buf.d_up[m], cells_out(), buf.d_min, buf.d_max);
        }

        double hmin=0, hmax=0;
        CUDA_OK(cudaMemcpyAsync(&hmin, buf.d_min, sizeof(double), cudaMemcpyDeviceToHost, st));
        CUDA_OK(cudaMemcpyAsync(&hmax, buf.d_max, sizeof(double), cudaMemcpyDeviceToHost, st));
        CUDA_OK(cudaStreamSynchronize(st));
        // Guard degenerate
        if (!(hmax > hmin)){ hmax = hmin + 1.0; }

        // Histogram init
        CUDA_OK(cudaMemsetAsync(buf.d_hist, 0, HIST_BINS*sizeof(unsigned int), st));

        // Build histogram across all months
        int histBlocks = 512;
        for (int m=0;m<MONTHS;++m){
            histogramKernel<<<histBlocks, 256, 0, st>>>(buf.d_up[m], cells_out(), hmin, hmax, buf.d_hist);
        }

        // CDF (exclusive scan) in one block
        prefixScanExclusive<<<1, HIST_BINS, 0, st>>>(buf.d_hist, buf.d_cdf);

        // Pick percentiles
        unsigned int totalCount = (unsigned int)N;
        pickPercentilesKernel<<<1,1,0,st>>>(buf.d_cdf, totalCount, hmin, hmax, buf.d_lo, buf.d_hi);
        double plo=0, phi=1;
        CUDA_OK(cudaMemcpyAsync(&plo, buf.d_lo, sizeof(double), cudaMemcpyDeviceToHost, st));
        CUDA_OK(cudaMemcpyAsync(&phi, buf.d_hi, sizeof(double), cudaMemcpyDeviceToHost, st));
        CUDA_OK(cudaStreamSynchronize(st));
        if (!(phi > plo)){ phi = plo + 1.0; }

        // Normalize each month
        for (int m=0;m<MONTHS;++m){
            normalizeKernel<<< (int)((cells_out()+blk1D.x-1)/blk1D.x), blk1D, 0, st >>>(buf.d_up[m], cells_out(), plo, phi);
        }

                for (int m=0; m<MONTHS; ++m){
            normalizeKernel<<< (int)((cells_out()+blk1D.x-1)/blk1D.x), blk1D, 0, st >>>(
                buf.d_up[m], cells_out(), plo, phi);
            enforcePeriodicX<<< (OUT_H+255)/256, 256, 0, st >>>(buf.d_up[m], OUT_W, OUT_H);
            seamAssert(st, buf.d_up[m], "Post-Normalize");
        }


        // --- 4) Download / store each month (async)
        for (int m=0;m<MONTHS;++m){
            CUDA_OK(cudaMemcpyAsync(h_out.data(), buf.d_up[m], bytes_out(), cudaMemcpyDeviceToHost, st));
            CUDA_OK(cudaStreamSynchronize(st)); // ensure data ready for store
            if (!storeOutputMonth(v, m, h_out)){
                fprintf(stderr,"Store failed: var=%d month=%d\n", v, m+1);
                return 4;
            }
        }

        printf("✔ Variable %d complete (plo=%.6g, phi=%.6g)\n", v, plo, phi);
    }

    // Cleanup
    for (int s=0;s<STREAMS;++s){
        free_stream_buffers(sb[s], streams[s]);
        CUDA_OK(cudaStreamDestroy(streams[s]));
    }
    printf("✅ All variables processed successfully.\n");
    return 0;
}
