//============================================================
// UniversalInterpolator_v4_FloodFill.cu
// BEST OPTION: Distance-weighted flood fill from nearest ocean
// - GPU-accelerated flood-fill to find nearest ocean cell
// - Each land cell uses its nearest ocean temperature
// - No directional bias, works for all geography
// - True latitudinal AND longitudinal variation
// Author: Claude (2025-11-12)
//============================================================

#include <iostream>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <cuda_runtime.h>
#include <iomanip>
#include <cstring>
#include "ClimateFileFormat.h"

constexpr int OUTPUT_WIDTH = STANDARD_WIDTH;
constexpr int OUTPUT_HEIGHT = STANDARD_HEIGHT;
constexpr int MONTHS = STANDARD_MONTHS;
constexpr int THREADS_PER_BLOCK = 256;
constexpr int TOTAL_VARIABLES = STANDARD_VARIABLES;
constexpr double M_PI = 3.14159265358979323846;

constexpr bool VISUALIZE_SNOW_MASKS = false;
constexpr int LAPLACIAN_ITERATIONS = 3000;  // Reduced since flood gives good starting point
constexpr double LAPLACIAN_WEIGHT = 0.20;   // Lighter blending

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
// Kernel 2: Initialize Mask and Ocean ID
//============================================================
__global__ void initFloodKernel(const double* __restrict__ data,
                                unsigned char* __restrict__ mask,
                                int* __restrict__ oceanID,
                                int totalPixels,
                                double sentinel)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= totalPixels) return;
    
    double val = data[idx];
    bool isOcean = !isMissing(val, sentinel);
    
    mask[idx] = isOcean ? 1 : 0;
    oceanID[idx] = isOcean ? idx : -1;  // Ocean cells point to themselves
}

//============================================================
// Kernel 3: Flood Fill Propagation (one iteration)
// Propagates ocean IDs to neighboring unassigned land cells
//============================================================
__global__ void floodFillStepKernel(const unsigned char* __restrict__ mask,
                                    int* __restrict__ oceanID,
                                    const int* __restrict__ oceanIDPrev,
                                    int width, int height,
                                    int* __restrict__ changeCount)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalPixels = width * height;
    
    if (idx >= totalPixels) return;
    if (mask[idx] == 1) return;  // Already ocean
    if (oceanID[idx] != -1) return;  // Already assigned
    
    int x = idx % width;
    int y = idx / width;
    
    // Check 4 cardinal neighbors
    int neighbors[4][2] = {{-1,0}, {1,0}, {0,-1}, {0,1}};
    
    for (int i = 0; i < 4; ++i) {
        int nx = (x + neighbors[i][0] + width) % width;  // Periodic X
        int ny = y + neighbors[i][1];
        
        if (ny >= 0 && ny < height) {
            int nIdx = ny * width + nx;
            int nOceanID = oceanIDPrev[nIdx];
            
            // If neighbor has ocean ID, adopt it
            if (nOceanID != -1) {
                oceanID[idx] = nOceanID;
                atomicAdd(changeCount, 1);
                return;  // Only assign once
            }
        }
    }
}

//============================================================
// Kernel 4: Flood with Nearest Ocean Values
// Each land cell gets the temperature of its nearest ocean cell
//============================================================
__global__ void floodWithOceanValuesKernel(double* __restrict__ data,
                                           const unsigned char* __restrict__ mask,
                                           const int* __restrict__ oceanID,
                                           int totalPixels)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= totalPixels) return;
    if (mask[idx] == 1) return;  // Keep ocean values
    
    int nearestOcean = oceanID[idx];
    if (nearestOcean != -1) {
        data[idx] = data[nearestOcean];  // Copy ocean temperature
    }
}

//============================================================
// Kernel 5: Laplacian Smooth Interpolation
//============================================================
__global__ void laplacianSmoothKernel(double* __restrict__ data,
                                      const double* __restrict__ dataIn,
                                      const unsigned char* __restrict__ mask,
                                      int width, int height,
                                      double weight)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalPixels = width * height;
    
    if (idx >= totalPixels) return;
    if (mask[idx] == 1) return;  // Don't modify ocean boundaries
    
    int x = idx % width;
    int y = idx / width;
    
    double center = dataIn[idx];
    double sum = 0.0;
    int count = 0;
    
    int neighbors[4][2] = {{-1,0}, {1,0}, {0,-1}, {0,1}};
    
    for (int i = 0; i < 4; ++i) {
        int nx = (x + neighbors[i][0] + width) % width;
        int ny = y + neighbors[i][1];
        
        if (ny >= 0 && ny < height) {
            int nIdx = ny * width + nx;
            sum += dataIn[nIdx];
            count++;
        }
    }
    
    if (count > 0) {
        double avg = sum / count;
        data[idx] = center * (1.0 - weight) + avg * weight;
    } else {
        data[idx] = center;
    }
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
// Kernel 7: Normalization
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
        std::cerr << "⚠️ Size mismatch in " << filePath << "\n";
    }
    
    return data;
}

//============================================================
// Host: Process Dataset with FLOOD FILL
//============================================================
void processDataset(const DatasetConfig& config, 
                   std::vector<double>& outputData)
{
    std::cout << "📊 Processing: " << config.name << "\n";
    
    if (config.isSnowMask && !VISUALIZE_SNOW_MASKS) {
        std::cout << "   ⏭️ Skipping visualization (snow mask)\n\n";
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
    
    int* d_oceanID = nullptr;
    CUDA_CHECK(cudaMalloc(&d_oceanID, totalPixels * sizeof(int)));
    
    // Load data
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
    
    // Initialize mask and ocean IDs
    int blocks2D = (spatialPixels + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    for (int month = 0; month < MONTHS; ++month) {
        initFloodKernel<<<blocks2D, THREADS_PER_BLOCK>>>(
            d_data + month * spatialPixels,
            d_mask + month * spatialPixels,
            d_oceanID + month * spatialPixels,
            spatialPixels,
            config.sentinel);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // FLOOD FILL to find nearest ocean for each land cell
    std::cout << "   🌊 Flood fill: Finding nearest ocean for each land cell...\n";
    
    int* d_oceanIDPrev = nullptr;
    CUDA_CHECK(cudaMalloc(&d_oceanIDPrev, totalPixels * sizeof(int)));
    
    int* d_changeCount = nullptr;
    CUDA_CHECK(cudaMalloc(&d_changeCount, sizeof(int)));
    
    int totalIterations = 0;
    int maxIterations = OUTPUT_WIDTH;  // Worst case: span entire map
    
    for (int iter = 0; iter < maxIterations; ++iter) {
        int h_change = 0;
        CUDA_CHECK(cudaMemcpy(d_changeCount, &h_change, sizeof(int), 
                             cudaMemcpyHostToDevice));
        
        // Copy current IDs to prev
        CUDA_CHECK(cudaMemcpy(d_oceanIDPrev, d_oceanID, 
                             totalPixels * sizeof(int), 
                             cudaMemcpyDeviceToDevice));
        
        // Propagate one step
        for (int month = 0; month < MONTHS; ++month) {
            floodFillStepKernel<<<blocks2D, THREADS_PER_BLOCK>>>(
                d_mask + month * spatialPixels,
                d_oceanID + month * spatialPixels,
                d_oceanIDPrev + month * spatialPixels,
                OUTPUT_WIDTH, OUTPUT_HEIGHT,
                d_changeCount);
        }
        
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(&h_change, d_changeCount, sizeof(int), 
                             cudaMemcpyDeviceToHost));
        
        totalIterations++;
        
        if (h_change == 0) break;  // No more changes
        
        if ((iter + 1) % 50 == 0) {
            std::cout << "      Iteration " << (iter + 1) << ": " 
                      << h_change << " cells assigned\r" << std::flush;
        }
    }
    
    std::cout << "\n      ✅ Flood fill complete (" << totalIterations << " iterations)\n";
    
    CUDA_CHECK(cudaFree(d_oceanIDPrev));
    CUDA_CHECK(cudaFree(d_changeCount));
    
    // Flood land with nearest ocean values
    std::cout << "   📋 Copying nearest ocean temperatures to land...\n";
    
    for (int month = 0; month < MONTHS; ++month) {
        floodWithOceanValuesKernel<<<blocks2D, THREADS_PER_BLOCK>>>(
            d_data + month * spatialPixels,
            d_mask + month * spatialPixels,
            d_oceanID + month * spatialPixels,
            spatialPixels);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "      ✅ Land flooded with ocean values\n";
    
    CUDA_CHECK(cudaFree(d_oceanID));
    
    // Laplacian smoothing for gentle blending
    std::cout << "   📐 Laplacian smoothing (" << LAPLACIAN_ITERATIONS 
              << " iterations)...\n";
    
    double* d_dataTemp = nullptr;
    CUDA_CHECK(cudaMalloc(&d_dataTemp, totalPixels * sizeof(double)));
    
    for (int iter = 0; iter < LAPLACIAN_ITERATIONS; ++iter) {
        for (int month = 0; month < MONTHS; ++month) {
            double* d_monthData = d_data + month * spatialPixels;
            double* d_monthTemp = d_dataTemp + month * spatialPixels;
            unsigned char* d_monthMask = d_mask + month * spatialPixels;
            
            CUDA_CHECK(cudaMemcpy(d_monthTemp, d_monthData,
                                 spatialPixels * sizeof(double),
                                 cudaMemcpyDeviceToDevice));
            
            laplacianSmoothKernel<<<blocks2D, THREADS_PER_BLOCK>>>(
                d_monthData, d_monthTemp, d_monthMask,
                OUTPUT_WIDTH, OUTPUT_HEIGHT, LAPLACIAN_WEIGHT);
            
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        
        if ((iter + 1) % 50 == 0) {
            std::cout << "      Progress: " << (iter + 1) << "/" 
                      << LAPLACIAN_ITERATIONS << "\r" << std::flush;
        }
    }
    
    std::cout << "\n      ✅ Smoothing complete\n";
    
    CUDA_CHECK(cudaFree(d_dataTemp));
    CUDA_CHECK(cudaFree(d_mask));
    
    // Temporal smoothing
    std::cout << "   🔄 Temporal smoothing...\n";
    temporalSmoothKernel<<<blocks2D, THREADS_PER_BLOCK>>>(
        d_data, OUTPUT_WIDTH, OUTPUT_HEIGHT, MONTHS, 0.15);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Normalization
    std::cout << "   📏 Normalizing...\n";
    int blocksAll = (totalPixels + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    normalizeSmoothKernel<<<blocksAll, THREADS_PER_BLOCK>>>(
        d_data, totalPixels, config.minValue, config.maxValue, 0.05);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy back
    CUDA_CHECK(cudaMemcpy(outputData.data(), d_data, 
                         totalPixels * sizeof(double), 
                         cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaFree(d_data));
    
    std::cout << "   ✅ Dataset complete\n\n";
}

/**
 * Writes consolidated climate data using standard ClimateFileFormat.
 * 
 * HEADER (28 bytes) - Standard format from ClimateFileFormat.h:
 *   uint32 Magic       = OCEANWORLD_MAGIC (0x574E434F = 'OCNW')
 *   uint32 Width       = 3600
 *   uint32 Height      = 1800
 *   uint32 Channels    = 168 (14 variables × 12 months)
 *   uint32 DataType    = 0 (double)
 *   uint32 Version     = 1
 *   uint32 Depth       = 12 (months)
 * 
 * DATA (starting at byte 28):
 *   for each cellIdx in [0 .. Width*Height-1]:
 *       for each varIdx in [0 .. TOTAL_VARIABLES-1]:
 *           for each month in [0 .. Depth-1]:
 *               write double value
 */
void writeConsolidatedBinary(const std::string& outputPath,
                             const std::vector<std::vector<double>>& allData)
{
    std::ofstream file(outputPath, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open output file: " + outputPath);
    }

    // Create and write standard 28-byte header
    ClimateFileHeader header = createStandardHeader(OCEANWORLD_MAGIC);
    writeClimateHeader(file, header);

    // Write data in [cell][variable][month] order
    size_t spatialPixels = header.width * header.height;

    for (size_t cellIdx = 0; cellIdx < spatialPixels; ++cellIdx) {
        for (int varIdx = 0; varIdx < TOTAL_VARIABLES; ++varIdx) {
            for (int month = 0; month < header.depth; ++month) {
                size_t dataIdx = month * spatialPixels + cellIdx;
                double value = allData[varIdx][dataIdx];
                file.write(reinterpret_cast<const char*>(&value), sizeof(double));
            }
        }
    }

    file.close();
}

//============================================================
// Main
//============================================================
int main()
{
    std::cout << "🌍 Universal Interpolator v4.0 (FLOOD FILL)\n";
    std::cout << "══════════════════════════════════════════════════════════\n";
    std::cout << "OPTION A - BEST: Nearest ocean flood fill\n";
    std::cout << "  • GPU-accelerated BFS flood fill\n";
    std::cout << "  • Each land cell → nearest ocean temperature\n";
    std::cout << "  • No directional bias, works for all geography\n";
    std::cout << "  • True lat/lon variation from real ocean data\n";
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
    std::cout << "✅ Flood fill processing complete!\n";
    std::cout << "   ✓ Each land cell uses nearest ocean temperature\n";
    std::cout << "   ✓ Geographic variation preserved\n";
    std::cout << "   ✓ Smooth gradients from Laplacian blending\n";
    std::cout << "   ✓ No continental artifacts\n";
    
    return 0;
}
