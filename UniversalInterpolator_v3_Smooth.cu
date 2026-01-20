//============================================================
// UniversalInterpolator_v3_Smooth.cu
// Emphasis on smooth, fluid gradients:
// - Removes continental ghosts via diffusion
// - Extra-smooth Gaussian weighting for fluid transitions
// - Increased diffusion for uniform smoothness
// - Minimal noise (preserves smooth quality)
// Author: Claude (2025-11-12)
//============================================================

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <cuda_runtime.h>
#include <iomanip>
#include <cstring>
#include <random>

constexpr int OUTPUT_WIDTH = 3600;
constexpr int OUTPUT_HEIGHT = 1800;
constexpr int MONTHS = 12;
constexpr int THREADS_PER_BLOCK = 256;
constexpr int MAX_SEARCH_RADIUS = 4800;
constexpr int RADIUS_INCREMENT = 1;
constexpr int TOTAL_VARIABLES = 14;
constexpr double M_PI = 3.14159265358979323846;

// Snow mask visualization control
constexpr bool VISUALIZE_SNOW_MASKS = false;

// TUNED FOR SMOOTHNESS
constexpr int DIFFUSION_PASSES = 8;           // More passes = smoother (was 5)
constexpr double DIFFUSION_STRENGTH = 0.25;   // Gentle blending (was 0.3)
constexpr double NOISE_AMPLITUDE = 0.005;     // Minimal noise (was 0.02)
constexpr double GAUSSIAN_SMOOTHNESS = 1.8;   // Lower = smoother (was 2.5)

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                  << " (" << __FILE__ << ":" << __LINE__ << ")\n"; \
        exit(EXIT_FAILURE); \
    } \
} while (0)

struct DatasetConfig {
    std::string name;
    double minValue;
    double maxValue;
    double sentinel;
    bool needsUpscale;
    std::string filenamePattern;
    bool isSnowMask;
};

//============================================================
// Device: Utilities
//============================================================
__device__ __forceinline__ bool isMissing(double v, double sentinel) {
    return !isfinite(v) || fabs(v - sentinel) < 1e-9;
}

__device__ __forceinline__ double smoothstep(double edge0, double edge1, double x) {
    double t = fmax(0.0, fmin(1.0, (x - edge0) / (edge1 - edge0)));
    return t * t * (3.0 - 2.0 * t);
}

__device__ __forceinline__ double hashNoise(int x, int y, int seed) {
    int n = x + y * 57 + seed * 131;
    n = (n << 13) ^ n;
    return 1.0 - ((n * (n * n * 15731 + 789221) + 1376312589) & 0x7fffffff) / 1073741824.0;
}

//============================================================
// Device: Periodic Bilinear Sampling
//============================================================
__device__ double bilinearSampleMaskedPeriodic(const double* __restrict__ input,
                                               int inWidth, int inHeight,
                                               double x, double y, double sentinel)
{
    x = fmod(x + inWidth * 1000.0, static_cast<double>(inWidth));
    if (x < 0.0) x += inWidth;
    
    y = fmax(0.0, fmin(y, inHeight - 1.001));
    
    int x0 = static_cast<int>(x);
    int y0 = static_cast<int>(y);
    
    int x1 = (x0 + 1) % inWidth;
    int y1 = min(y0 + 1, inHeight - 1);
    
    double fx = x - x0;
    double fy = y - y0;
    
    double v00 = input[y0 * inWidth + x0];
    double v10 = input[y0 * inWidth + x1];
    double v01 = input[y1 * inWidth + x0];
    double v11 = input[y1 * inWidth + x1];
    
    if (isMissing(v00, sentinel)) v00 = NAN;
    if (isMissing(v10, sentinel)) v10 = NAN;
    if (isMissing(v01, sentinel)) v01 = NAN;
    if (isMissing(v11, sentinel)) v11 = NAN;
    
    double w00 = (1.0 - fx) * (1.0 - fy);
    double w10 = fx * (1.0 - fy);
    double w01 = (1.0 - fx) * fy;
    double w11 = fx * fy;
    
    double sum = 0.0;
    double weightSum = 0.0;
    
    if (isfinite(v00)) { sum += v00 * w00; weightSum += w00; }
    if (isfinite(v10)) { sum += v10 * w10; weightSum += w10; }
    if (isfinite(v01)) { sum += v01 * w01; weightSum += w01; }
    if (isfinite(v11)) { sum += v11 * w11; weightSum += w11; }
    
    return (weightSum > 0.0) ? (sum / weightSum) : NAN;
}

//============================================================
// Kernel 1: Bilinear Upscaling
//============================================================
__global__ void bilinearUpscaleKernel(const double* __restrict__ input,
                                      double* __restrict__ output,
                                      int inWidth, int inHeight,
                                      int outWidth, int outHeight,
                                      double sentinel)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalPixels = outWidth * outHeight;
    
    if (idx >= totalPixels) return;
    
    int outX = idx % outWidth;
    int outY = idx / outWidth;
    
    double inX = (outX + 0.5) * (static_cast<double>(inWidth) / outWidth) - 0.5;
    double inY = (outY + 0.5) * (static_cast<double>(inHeight) / outHeight) - 0.5;
    
    output[idx] = bilinearSampleMaskedPeriodic(input, inWidth, inHeight, inX, inY, sentinel);
}

//============================================================
// Kernel 2: Initialize Mask
//============================================================
__global__ void initMaskKernel(const double* __restrict__ data,
                               unsigned char* __restrict__ mask,
                               int totalPixels,
                               double sentinel)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= totalPixels) return;
    
    double val = data[idx];
    mask[idx] = (!isMissing(val, sentinel)) ? 1 : 0;
}

//============================================================
// Kernel 3: Continental Pre-Flood (Minimal Noise)
//============================================================
__global__ void prefloodContinentsKernel(double* __restrict__ data,
                                         const unsigned char* __restrict__ mask,
                                         int width, int height,
                                         double latitudinalBase,
                                         int noiseSeed,
                                         double noiseAmp)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalPixels = width * height;
    
    if (idx >= totalPixels) return;
    if (mask[idx] == 1) return;
    
    int x = idx % width;
    int y = idx / width;
    
    double lat = (static_cast<double>(y) / height - 0.5) * 180.0;
    double latFactor = cos(lat * M_PI / 180.0);
    
    double baseValue = latitudinalBase * (0.4 + 0.6 * latFactor);
    
    // Minimal noise for slight variation without breaking smoothness
    double noise = hashNoise(x, y, noiseSeed) * noiseAmp;
    baseValue = fmax(0.0, fmin(1.0, baseValue + noise));
    
    data[idx] = baseValue;
}

//============================================================
// Kernel 4: EXTRA-SMOOTH Spatial Fill (Wider Gaussian)
//============================================================
__global__ void fillLandPixelsKernel(double* __restrict__ data,
                                     unsigned char* __restrict__ mask,
                                     int width, int height,
                                     int searchRadius,
                                     int* __restrict__ changeCount,
                                     double gaussianSmoothness)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalPixels = width * height;
    
    if (idx >= totalPixels) return;
    if (mask[idx] == 1) return;
    
    int centerX = idx % width;
    int centerY = idx / width;
    
    double sumWeighted = 0.0;
    double sumWeights = 0.0;
    
    // SMOOTHER: Wider Gaussian for more fluid transitions
    for (int dy = -searchRadius; dy <= searchRadius; ++dy) {
        for (int dx = -searchRadius; dx <= searchRadius; ++dx) {
            double dist = sqrt(static_cast<double>(dx*dx + dy*dy));
            if (dist > searchRadius || dist < 0.5) continue;
            
            int nx = (centerX + dx + width * 1000) % width;
            int ny = centerY + dy;
            
            if (ny < 0 || ny >= height) continue;
            
            int neighborIdx = ny * width + nx;
            
            if (mask[neighborIdx] == 1) {
                // EXTRA-SMOOTH: Wider sigma = gentler falloff
                double sigma = searchRadius / gaussianSmoothness;
                double weight = exp(-0.5 * (dist * dist) / (sigma * sigma));
                
                sumWeighted += data[neighborIdx] * weight;
                sumWeights += weight;
            }
        }
    }
    
    if (sumWeights > 1e-6) {
        data[idx] = sumWeighted / sumWeights;
        mask[idx] = 1;
        atomicAdd(changeCount, 1);
    }
}

//============================================================
// Kernel 5: Multi-Pass Diffusion (Ghost Removal)
//============================================================
__global__ void diffusionSmoothKernel(double* __restrict__ data,
                                      const double* __restrict__ dataIn,
                                      const unsigned char* __restrict__ mask,
                                      int width, int height,
                                      double blendFactor)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalPixels = width * height;
    
    if (idx >= totalPixels) return;
    if (mask[idx] == 0) return;
    
    int x = idx % width;
    int y = idx / width;
    
    double center = dataIn[idx];
    double sum = center;
    double weight = 1.0;
    
    // 4-neighbor averaging with periodic boundaries
    int neighbors[4][2] = {{-1,0}, {1,0}, {0,-1}, {0,1}};
    
    for (int i = 0; i < 4; ++i) {
        int nx = (x + neighbors[i][0] + width) % width;
        int ny = y + neighbors[i][1];
        
        if (ny >= 0 && ny < height) {
            int nIdx = ny * width + nx;
            if (mask[nIdx] == 1) {
                sum += dataIn[nIdx];
                weight += 1.0;
            }
        }
    }
    
    double avg = sum / weight;
    data[idx] = center * (1.0 - blendFactor) + avg * blendFactor;
}

//============================================================
// Kernel 6: Temporal Smoothing
//============================================================
__global__ void temporalSmoothKernel(double* __restrict__ data,
                                     int width, int height,
                                     int months,
                                     double blendFactor)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int spatialPixels = width * height;
    
    if (idx >= spatialPixels) return;
    
    for (int m = 0; m < months; ++m) {
        int currIdx = m * spatialPixels + idx;
        int prevIdx = ((m - 1 + months) % months) * spatialPixels + idx;
        int nextIdx = ((m + 1) % months) * spatialPixels + idx;
        
        double curr = data[currIdx];
        double prev = data[prevIdx];
        double next = data[nextIdx];
        
        if (isfinite(curr) && isfinite(prev) && isfinite(next)) {
            double avg = (prev + curr + next) / 3.0;
            data[currIdx] = curr * (1.0 - blendFactor) + avg * blendFactor;
        }
    }
}

//============================================================
// Kernel 7: Smoothstep Normalization
//============================================================
__global__ void normalizeSmoothKernel(double* __restrict__ data,
                                      int totalPixels,
                                      double minVal,
                                      double maxVal,
                                      double softMargin)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= totalPixels) return;
    
    double val = data[idx];
    if (!isfinite(val)) return;
    
    double normalized = (val - minVal) / (maxVal - minVal);
    normalized = fmax(0.0, fmin(1.0, normalized));
    
    if (normalized < softMargin) {
        normalized = smoothstep(0.0, softMargin, normalized) * softMargin;
    } else if (normalized > (1.0 - softMargin)) {
        double t = (normalized - (1.0 - softMargin)) / softMargin;
        normalized = (1.0 - softMargin) + smoothstep(0.0, 1.0, t) * softMargin;
    }
    
    data[idx] = normalized;
}

//============================================================
// Host: Load CSV
//============================================================
std::vector<double> loadCSV(const std::string& filePath, 
                            int expectedWidth, int expectedHeight)
{
    std::ifstream file(filePath);
    if (!file) {
        std::cerr << "❌ Cannot open: " << filePath << "\n";
        return {};
    }
    
    std::vector<double> data;
    data.reserve(expectedWidth * expectedHeight);
    
    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string token;
        
        while (std::getline(ss, token, ',')) {
            try {
                data.push_back(std::stod(token));
            } catch (...) {
                data.push_back(NAN);
            }
        }
    }
    
    if (data.size() != expectedWidth * expectedHeight) {
        std::cerr << "⚠️ Size mismatch in " << filePath << ": got " 
                  << data.size() << ", expected " 
                  << (expectedWidth * expectedHeight) << "\n";
    }
    
    return data;
}

//============================================================
// Host: Process Dataset (SMOOTH PIPELINE)
//============================================================
void processDataset(const DatasetConfig& config, 
                   std::vector<double>& outputData)
{
    std::cout << "📊 Processing: " << config.name << "\n";
    
    if (config.isSnowMask && !VISUALIZE_SNOW_MASKS) {
        std::cout << "   ⏭️ Skipping visualization (snow mask) - data preserved in binary\n\n";
    }
    
    int inputWidth = config.needsUpscale ? 1440 : OUTPUT_WIDTH;
    int inputHeight = config.needsUpscale ? 720 : OUTPUT_HEIGHT;
    
    size_t spatialPixels = OUTPUT_WIDTH * OUTPUT_HEIGHT;
    size_t totalPixels = spatialPixels * MONTHS;
    
    outputData.resize(totalPixels, NAN);
    
    double* d_data = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, totalPixels * sizeof(double)));
    
    unsigned char* d_mask = nullptr;
    CUDA_CHECK(cudaMalloc(&d_mask, totalPixels * sizeof(unsigned char)));
    
    // Load and upscale
    std::cout << "   📂 Loading monthly files...\n";
    
    for (int month = 0; month < MONTHS; ++month) {
        std::string monthStr = (month < 9) ? "0" + std::to_string(month + 1) 
                                           : std::to_string(month + 1);
        
        std::string filename = config.filenamePattern;
        size_t pos = filename.find("{NAME}");
        if (pos != std::string::npos) 
            filename.replace(pos, 6, config.name);
        
        pos = filename.find("{MM}");
        if (pos != std::string::npos) 
            filename.replace(pos, 4, monthStr);
        
        std::vector<double> monthData = loadCSV(filename, inputWidth, inputHeight);
        
        if (monthData.empty()) continue;
        
        if (config.needsUpscale) {
            double* d_input = nullptr;
            CUDA_CHECK(cudaMalloc(&d_input, monthData.size() * sizeof(double)));
            CUDA_CHECK(cudaMemcpy(d_input, monthData.data(), 
                                 monthData.size() * sizeof(double), 
                                 cudaMemcpyHostToDevice));
            
            int blocks = (spatialPixels + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
            bilinearUpscaleKernel<<<blocks, THREADS_PER_BLOCK>>>(
                d_input, d_data + month * spatialPixels,
                inputWidth, inputHeight, OUTPUT_WIDTH, OUTPUT_HEIGHT,
                config.sentinel);
            
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaFree(d_input));
        } else {
            CUDA_CHECK(cudaMemcpy(d_data + month * spatialPixels, monthData.data(),
                                 spatialPixels * sizeof(double),
                                 cudaMemcpyHostToDevice));
        }
    }
    
    std::cout << "      ✅ Files loaded\n";
    
    // Initialize mask
    int blocks2D = (spatialPixels + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    for (int month = 0; month < MONTHS; ++month) {
        initMaskKernel<<<blocks2D, THREADS_PER_BLOCK>>>(
            d_data + month * spatialPixels,
            d_mask + month * spatialPixels,
            spatialPixels,
            config.sentinel);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Pre-flood with minimal noise
    std::cout << "   🏝️ Pre-flooding continents (minimal noise)...\n";
    std::random_device rd;
    int noiseSeed = rd();
    
    for (int month = 0; month < MONTHS; ++month) {
        double latBase = 0.5;
        prefloodContinentsKernel<<<blocks2D, THREADS_PER_BLOCK>>>(
            d_data + month * spatialPixels,
            d_mask + month * spatialPixels,
            OUTPUT_WIDTH, OUTPUT_HEIGHT,
            latBase, noiseSeed + month, NOISE_AMPLITUDE);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "      ✅ Pre-flooded\n";
    
    // Smooth spatial fill
    std::cout << "   🌊 Filling land pixels (extra-smooth Gaussian)...\n";
    
    int* d_changeCount = nullptr;
    CUDA_CHECK(cudaMalloc(&d_changeCount, sizeof(int)));
    
    int totalFilledAllMonths = 0;
    
    for (int month = 0; month < MONTHS; ++month) {
        double* d_monthData = d_data + month * spatialPixels;
        unsigned char* d_monthMask = d_mask + month * spatialPixels;
        
        int monthFilled = 0;
        for (int radius = 1; radius <= MAX_SEARCH_RADIUS; radius += RADIUS_INCREMENT) {
            int h_change = 0;
            CUDA_CHECK(cudaMemcpy(d_changeCount, &h_change, sizeof(int), 
                                 cudaMemcpyHostToDevice));
            
            fillLandPixelsKernel<<<blocks2D, THREADS_PER_BLOCK>>>(
                d_monthData, d_monthMask, OUTPUT_WIDTH, OUTPUT_HEIGHT, 
                radius, d_changeCount, GAUSSIAN_SMOOTHNESS);
            
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(&h_change, d_changeCount, sizeof(int), 
                                 cudaMemcpyDeviceToHost));
            
            monthFilled += h_change;
            if (h_change == 0 && radius > 20) break;
        }
        
        totalFilledAllMonths += monthFilled;
    }
    
    std::cout << "      ✅ Filled " << totalFilledAllMonths << " pixels\n";
    
    // Enhanced diffusion for ghost removal
    std::cout << "   🌀 Applying diffusion smoothing (" << DIFFUSION_PASSES 
              << " passes, " << (DIFFUSION_STRENGTH * 100) << "% blend)...\n";
    
    double* d_dataTemp = nullptr;
    CUDA_CHECK(cudaMalloc(&d_dataTemp, totalPixels * sizeof(double)));
    
    for (int pass = 0; pass < DIFFUSION_PASSES; ++pass) {
        for (int month = 0; month < MONTHS; ++month) {
            double* d_monthData = d_data + month * spatialPixels;
            double* d_monthTemp = d_dataTemp + month * spatialPixels;
            unsigned char* d_monthMask = d_mask + month * spatialPixels;
            
            CUDA_CHECK(cudaMemcpy(d_monthTemp, d_monthData,
                                 spatialPixels * sizeof(double),
                                 cudaMemcpyDeviceToDevice));
            
            diffusionSmoothKernel<<<blocks2D, THREADS_PER_BLOCK>>>(
                d_monthData, d_monthTemp, d_monthMask,
                OUTPUT_WIDTH, OUTPUT_HEIGHT, DIFFUSION_STRENGTH);
            
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }
    
    CUDA_CHECK(cudaFree(d_dataTemp));
    std::cout << "      ✅ Diffusion complete (ghosts removed)\n";
    
    CUDA_CHECK(cudaFree(d_mask));
    CUDA_CHECK(cudaFree(d_changeCount));
    
    // Temporal smoothing
    std::cout << "   🔄 Temporal smoothing...\n";
    temporalSmoothKernel<<<blocks2D, THREADS_PER_BLOCK>>>(
        d_data, OUTPUT_WIDTH, OUTPUT_HEIGHT, MONTHS, 0.15);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Smoothstep normalization
    std::cout << "   📏 Normalizing with soft tapering...\n";
    int blocksAll = (totalPixels + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    double softMargin = 0.05;
    normalizeSmoothKernel<<<blocksAll, THREADS_PER_BLOCK>>>(
        d_data, totalPixels, config.minValue, config.maxValue, softMargin);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy back
    CUDA_CHECK(cudaMemcpy(outputData.data(), d_data, 
                         totalPixels * sizeof(double), 
                         cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaFree(d_data));
    
    std::cout << "   ✅ Dataset complete\n\n";
}

//============================================================
// Host: Write Binary
//============================================================
void writeConsolidatedBinary(const std::string& outputPath,
                             const std::vector<std::vector<double>>& allData)
{
    std::cout << "💾 Writing OceanWorld_Climate.bin...\n";
    
    std::ofstream file(outputPath, std::ios::binary);
    if (!file) {
        std::cerr << "❌ Failed to create output file\n";
        return;
    }
    
    int depth = MONTHS;
    int height = OUTPUT_HEIGHT;
    int width = OUTPUT_WIDTH;
    
    file.write(reinterpret_cast<const char*>(&depth), sizeof(int));
    file.write(reinterpret_cast<const char*>(&height), sizeof(int));
    file.write(reinterpret_cast<const char*>(&width), sizeof(int));
    
    std::cout << "   Header: depth=" << depth << ", height=" << height 
              << ", width=" << width << "\n";
    std::cout << "   Variables: " << TOTAL_VARIABLES << "\n";
    std::cout << "   Layout: [cell][variable][month]\n";
    
    size_t spatialPixels = width * height;
    size_t bytesWritten = 12;
    
    for (size_t cellIdx = 0; cellIdx < spatialPixels; ++cellIdx) {
        for (int varIdx = 0; varIdx < TOTAL_VARIABLES; ++varIdx) {
            for (int month = 0; month < depth; ++month) {
                size_t dataIdx = month * spatialPixels + cellIdx;
                double value = allData[varIdx][dataIdx];
                file.write(reinterpret_cast<const char*>(&value), sizeof(double));
                bytesWritten += 8;
            }
        }
        
        if (cellIdx % 100000 == 0) {
            std::cout << "      Progress: " << (cellIdx * 100 / spatialPixels) 
                      << "%\r" << std::flush;
        }
    }
    
    file.close();
    
    std::cout << "\n   ✅ Wrote " << (bytesWritten / (1024.0*1024.0*1024.0)) 
              << " GB\n";
    
    size_t expectedSize = 12 + (spatialPixels * TOTAL_VARIABLES * depth * sizeof(double));
    if (bytesWritten != expectedSize) {
        std::cerr << "⚠️ Size mismatch! Expected " << expectedSize 
                  << " bytes, wrote " << bytesWritten << "\n";
    }
}

//============================================================
// Main
//============================================================
int main()
{
    std::cout << "🌍 Unified Climate Processor v3.0 (Ultra-Smooth Edition)\n";
    std::cout << "══════════════════════════════════════════════════════════\n";
    std::cout << "TUNED FOR SMOOTHNESS:\n";
    std::cout << "  • " << DIFFUSION_PASSES << " diffusion passes @ " 
              << (DIFFUSION_STRENGTH * 100) << "% blend (removes ghosts)\n";
    std::cout << "  • Extra-wide Gaussian (sigma/" << GAUSSIAN_SMOOTHNESS 
              << ") for fluid transitions\n";
    std::cout << "  • Minimal noise (" << (NOISE_AMPLITUDE * 100) 
              << "%) preserves smoothness\n";
    std::cout << "══════════════════════════════════════════════════════════\n\n";
    
    std::vector<DatasetConfig> datasets = {
        {"GPM_3IMERGM", 0.0, 300.0, 99999.0, false, "./input/{NAME}_2024-{MM}-01_rgb_3600x1800.CSV", false},
        {"MYD28M", -2.0, 35.0, 99999.0, false, "./input/{NAME}_2024-{MM}-01_rgb_3600x1800.CSV", false},
        {"MOD_LSTD_M", -50.0, 60.0, 99999.0, false, "./input/{NAME}_2024-{MM}-01_rgb_3600x1800.CSV", false},
        {"MOD_LSTN_M", -50.0, 60.0, 99999.0, false, "./input/{NAME}_2024-{MM}-01_rgb_3600x1800.CSV", false},
        {"MCD43C3_M_BSA", 0.0, 1.0, 99999.0, false, "./input/{NAME}_2024-{MM}-01_rgb_3600x1800.CSV", false},
        {"CERES_NETFLUX_M", 0.0, 500.0, 99999.0, true, "./input/{NAME}_2024-{MM}-01_rgb_1440x720.CSV", false},
        {"CERES_SWFLUX_M", 0.0, 600.0, 99999.0, true, "./input/{NAME}_2024-{MM}-01_rgb_1440x720.CSV", false},
        {"CERES_LWFLUX_M", 0.0, 400.0, 99999.0, true, "./input/{NAME}_2024-{MM}-01_rgb_1440x720.CSV", false},
        {"CERES_INSOL_M", 0.0, 500.0, 99999.0, true, "./input/{NAME}_2024-{MM}-01_rgb_1440x720.CSV", false},
        {"MOD15A2_M_LAI", 0.0, 7.0, 32767.0, false, "./input/{NAME}_2024-{MM}-01_rgb_3600x1800.CSV", false},
        {"NISE_D", 0.0, 1.0, 255.0, false, "./input/{NAME}_2024-{MM}-01_rgb_3600x1800.CSV", true},
        {"MOD10C1_M_SNOW", 0.0, 1.0, 255.0, false, "./input/{NAME}_2024-{MM}-01_rgb_3600x1800.CSV", true},
        {"MOD_NDVI_M", -0.2, 1.0, 32767.0, false, "./input/{NAME}_2024-{MM}-01_rgb_3600x1800.CSV", false}
    };
    
    std::vector<std::vector<double>> allData(TOTAL_VARIABLES);
    
    for (size_t i = 0; i < datasets.size(); ++i) {
        processDataset(datasets[i], allData[i]);
    }
    
    // Derive AirTemp
    std::cout << "🌡️ Deriving Air Temperature from SST...\n";
    size_t sstIndex = 1;
    size_t totalPixels = OUTPUT_WIDTH * OUTPUT_HEIGHT * MONTHS;
    
    allData[13].resize(totalPixels);
    
    for (size_t i = 0; i < totalPixels; ++i) {
        double sst = allData[sstIndex][i];
        if (isfinite(sst)) {
            double sstCelsius = sst * 37.0 - 2.0;
            double airTempCelsius = sstCelsius - 2.0;
            double airTempNorm = (airTempCelsius + 4.0) / 37.0;
            allData[13][i] = fmax(0.0, fmin(1.0, airTempNorm));
        } else {
            allData[13][i] = sst;
        }
    }
    std::cout << "   ✅ Air Temperature derived\n\n";
    
    writeConsolidatedBinary("./output/OceanWorld_Climate.bin", allData);
    
    std::cout << "\n══════════════════════════════════════════════════════════\n";
    std::cout << "✅ Ultra-smooth processing complete!\n";
    std::cout << "   Output: ./output/OceanWorld_Climate.bin\n";
    std::cout << "   Expected size: 7.60 GB\n";
    std::cout << "\n🎯 Smoothness settings:\n";
    std::cout << "   ✓ Diffusion: " << DIFFUSION_PASSES << " passes removes ghosts\n";
    std::cout << "   ✓ Gaussian: σ/" << GAUSSIAN_SMOOTHNESS << " creates fluid gradients\n";
    std::cout << "   ✓ Noise: " << (NOISE_AMPLITUDE * 100) << "% maintains smoothness\n";
    std::cout << "\n💡 Tuning tips (edit constants at top of file):\n";
    std::cout << "   • More ghost removal: Increase DIFFUSION_PASSES to 10-12\n";
    std::cout << "   • Even smoother: Decrease GAUSSIAN_SMOOTHNESS to 1.5\n";
    std::cout << "   • Less noise: Decrease NOISE_AMPLITUDE to 0.001\n";
    
    return 0;
}
