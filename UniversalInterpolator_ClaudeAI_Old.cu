//============================================================
// GPU_BilinearUpscaleInterpolate_FIXED.cu
// Complete rewrite with ChatGPT's fixes applied
// ------------------------------------------------------------
// FIXES APPLIED:
// 1. Per-month spatial processing (no month mixing)
// 2. Masked bilinear (no sentinel contamination)
// 3. Proper zero handling (zeros are valid data)
// 4. Temporal cosine interpolation
// 5. Consolidated OceanWorld_Climate.bin output
// 6. Per-dataset sentinel values
//
// Author: Mark Devereux (rewritten)
// Date: 2025-10-17
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

constexpr int OUTPUT_WIDTH = 3600;
constexpr int OUTPUT_HEIGHT = 1800;
constexpr int MONTHS = 12;
constexpr int THREADS_PER_BLOCK = 256;
constexpr int MAX_SEARCH_RADIUS = 900;   // Increased for continent-spanning blends
constexpr int RADIUS_INCREMENT = 20;     // Larger steps for faster convergence
constexpr int TOTAL_VARIABLES = 14;      // 13 source + 1 derived (AirTemp)
constexpr double M_PI = 3.14159265358979323846;

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
};

//============================================================
// Device: Check if value is missing/sentinel
//============================================================
__device__ __forceinline__ bool isMissing(double v, double sentinel) {
    return !isfinite(v) || fabs(v - sentinel) < 1e-9;
}

//============================================================
// Device: Masked Bilinear Sampling (FIX #2)
// Avoids sentinel contamination by ignoring invalid neighbors
//============================================================
__device__ double bilinearSampleMasked(const double* __restrict__ input,
                                       int inWidth, int inHeight,
                                       double x, double y, double sentinel)
{
    x = fmax(0.0, fmin(x, inWidth - 1.001));
    y = fmax(0.0, fmin(y, inHeight - 1.001));
    
    int x0 = static_cast<int>(x);
    int y0 = static_cast<int>(y);
    int x1 = min(x0 + 1, inWidth - 1);
    int y1 = min(y0 + 1, inHeight - 1);
    
    double fx = x - x0;
    double fy = y - y0;
    
    // Fetch corner values
    double v00 = input[y0 * inWidth + x0];
    double v10 = input[y0 * inWidth + x1];
    double v01 = input[y1 * inWidth + x0];
    double v11 = input[y1 * inWidth + x1];
    
    // Mark sentinels as NaN
    if (isMissing(v00, sentinel)) v00 = NAN;
    if (isMissing(v10, sentinel)) v10 = NAN;
    if (isMissing(v01, sentinel)) v01 = NAN;
    if (isMissing(v11, sentinel)) v11 = NAN;
    
    // Bilinear weights
    double w00 = (1.0 - fx) * (1.0 - fy);
    double w10 = fx * (1.0 - fy);
    double w01 = (1.0 - fx) * fy;
    double w11 = fx * fy;
    
    // Accumulate only valid samples
    double sum = 0.0;
    double weightSum = 0.0;
    
    if (isfinite(v00)) { sum += v00 * w00; weightSum += w00; }
    if (isfinite(v10)) { sum += v10 * w10; weightSum += w10; }
    if (isfinite(v01)) { sum += v01 * w01; weightSum += w01; }
    if (isfinite(v11)) { sum += v11 * w11; weightSum += w11; }
    
    return (weightSum > 0.0) ? (sum / weightSum) : NAN;
}

//============================================================
// Kernel 1: Bilinear Upscaling with Masked Sampling
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
    
    // Use masked bilinear (FIX #2)
    double value = bilinearSampleMasked(input, inWidth, inHeight, inX, inY, sentinel);
    
    output[idx] = value;
}

//============================================================
// Kernel 2: Initialize Mask (FIX #3 - No zero rejection)
//============================================================
__global__ void initMaskKernel(const double* __restrict__ data,
                               unsigned char* __restrict__ mask,
                               int totalPixels)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= totalPixels) return;
    
    double val = data[idx];
    // FIX: Only reject NaN, not zeros
    mask[idx] = isfinite(val) ? 1 : 0;
}

//============================================================
// Kernel 3: Spatial Fill with Cosine Weighting (FIX #1 - Per Month)
//============================================================
__global__ void fillLandPixelsKernel(double* __restrict__ data,
                                     unsigned char* __restrict__ mask,
                                     int width, int height,
                                     int searchRadius,
                                     int* __restrict__ changeCount)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalPixels = width * height;
    
    if (idx >= totalPixels) return;
    if (mask[idx] == 1) return;  // Already valid
    
    int x = idx % width;
    int y = idx / width;
    
    double sumValue = 0.0;
    double weightSum = 0.0;
    
    for (int dy = -searchRadius; dy <= searchRadius; ++dy) {
        for (int dx = -searchRadius; dx <= searchRadius; ++dx) {
            if (dx == 0 && dy == 0) continue;
            
            int nx = (x + dx + width) % width;  // Wrap longitude
            int ny = y + dy;
            
            if (ny < 0 || ny >= height) continue;  // Clamp latitude
            
            int nIdx = ny * width + nx;
            
            if (mask[nIdx] == 1) {
                double dist = sqrt(static_cast<double>(dx*dx + dy*dy));
                
                // Cosine weighting for smooth transitions
                double t = fmin(dist / searchRadius, 1.0);
                double weight;
                
                if (dist < 2.0) {
                    weight = 1.0;  // Full weight for immediate neighbors
                } else {
                    // Cosine easing: smooth S-curve
                    weight = 0.5 * (1.0 - cos(M_PI * (1.0 - t)));
                }
                
                sumValue += data[nIdx] * weight;
                weightSum += weight;
            }
        }
    }
    
    if (weightSum > 0.0) {
        data[idx] = sumValue / weightSum;
        mask[idx] = 1;
        atomicAdd(changeCount, 1);
    }
}

//============================================================
// Kernel 4: Temporal Cosine Smoothing (FIX #4)
//============================================================
__global__ void temporalSmoothKernel(double* __restrict__ data,
                                     int width, int height,
                                     int months,
                                     double strength)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int spatialPixels = width * height;
    
    if (idx >= spatialPixels) return;
    
    // For each spatial location, smooth across months
    for (int m = 0; m < months; ++m) {
        int prevMonth = (m + months - 1) % months;
        int nextMonth = (m + 1) % months;
        
        double curr = data[m * spatialPixels + idx];
        double prev = data[prevMonth * spatialPixels + idx];
        double next = data[nextMonth * spatialPixels + idx];
        
        if (isfinite(curr) && isfinite(prev) && isfinite(next)) {
            // Cosine blend with neighbors
            double blended = curr * (1.0 - strength) + 
                           (prev + next) * 0.5 * strength;
            data[m * spatialPixels + idx] = blended;
        }
    }
}

//============================================================
// Kernel 5: Normalize to [0,1]
//============================================================
__global__ void normalizeKernel(double* __restrict__ data,
                                int totalPixels,
                                double minVal, double maxVal)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= totalPixels) return;
    
    double val = data[idx];
    double range = maxVal - minVal;
    
    if (range > 0.0 && isfinite(val)) {
        val = (val - minVal) / range;
        val = fmax(0.0, fmin(1.0, val));
        data[idx] = val;
    } else if (isfinite(val)) {
        data[idx] = 0.5;  // Fallback
    }
    // Leave NaN as NaN for now
}

//============================================================
// Host: Load CSV
//============================================================
std::vector<double> loadCSV(const std::string& path, int& rows, int& cols)
{
    std::ifstream file(path);
    if (!file) {
        std::cerr << "❌ Failed to open: " << path << "\n";
        return {};
    }
    
    std::vector<double> data;
    std::string line;
    rows = cols = 0;
    
    while (getline(file, line)) {
        std::stringstream ss(line);
        std::string val;
        int colCount = 0;
        
        while (getline(ss, val, ',')) {
            try {
                data.push_back(std::stod(val));
                colCount++;
            } catch (...) {
                data.push_back(NAN);
                colCount++;
            }
        }
        
        if (rows == 0) cols = colCount;
        rows++;
    }
    
    return data;
}

//============================================================
// Host: Process Single Dataset
//============================================================
void processDataset(const DatasetConfig& config, std::vector<double>& outputData)
{
    std::cout << "📊 Processing: " << config.name << "\n";
    
    size_t spatialPixels = OUTPUT_WIDTH * OUTPUT_HEIGHT;
    size_t totalPixels = spatialPixels * MONTHS;
    outputData.resize(totalPixels);
    
    // Allocate device memory
    double* d_data = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, totalPixels * sizeof(double)));
    
    // Process each month
    for (int month = 0; month < MONTHS; ++month) {
        std::string filename = config.filenamePattern;
        
        size_t namePos = filename.find("{NAME}");
        if (namePos != std::string::npos) {
            filename.replace(namePos, 6, config.name);
        }
        
        std::stringstream monthStr;
        monthStr << std::setfill('0') << std::setw(2) << (month + 1);
        size_t mmPos = filename.find("{MM}");
        if (mmPos != std::string::npos) {
            filename.replace(mmPos, 4, monthStr.str());
        }
        
        std::string fullPath = "./input/" + filename;
        
        int inRows, inCols;
        std::vector<double> hostInput = loadCSV(fullPath, inRows, inCols);
        
        if (hostInput.empty()) {
            std::cerr << "⚠️  Skipping month " << (month+1) << "\n";
            continue;
        }
        
        std::cout << "   Month " << std::setw(2) << (month+1) << ": " 
                  << inCols << "×" << inRows;
        
        // Upload to device
        double* d_input = nullptr;
        CUDA_CHECK(cudaMalloc(&d_input, hostInput.size() * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_input, hostInput.data(), 
                             hostInput.size() * sizeof(double), 
                             cudaMemcpyHostToDevice));
        
        // Pointer to this month's output
        double* d_monthOut = d_data + month * spatialPixels;
        
        int blocks = (spatialPixels + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        
        if (config.needsUpscale && (inCols != OUTPUT_WIDTH || inRows != OUTPUT_HEIGHT)) {
            bilinearUpscaleKernel<<<blocks, THREADS_PER_BLOCK>>>(
                d_input, d_monthOut, inCols, inRows, 
                OUTPUT_WIDTH, OUTPUT_HEIGHT, config.sentinel);
            std::cout << " → Upscaled to " << OUTPUT_WIDTH << "×" << OUTPUT_HEIGHT;
        } else {
            CUDA_CHECK(cudaMemcpy(d_monthOut, d_input, 
                                 spatialPixels * sizeof(double), 
                                 cudaMemcpyDeviceToDevice));
            std::cout << " → Direct copy";
        }
        std::cout << "\n";
        
        CUDA_CHECK(cudaFree(d_input));
    }
    
    // FIX #1: Process spatial fill PER MONTH
    std::cout << "   🌊 Filling land pixels (per-month processing)...\n";
    
    unsigned char* d_mask = nullptr;
    int* d_changeCount = nullptr;
    CUDA_CHECK(cudaMalloc(&d_mask, totalPixels * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_changeCount, sizeof(int)));
    
    int blocks2D = (spatialPixels + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    int totalFilledAllMonths = 0;
    
    for (int month = 0; month < MONTHS; ++month) {
        double* d_monthData = d_data + month * spatialPixels;
        unsigned char* d_monthMask = d_mask + month * spatialPixels;
        
        // Initialize mask for this month
        initMaskKernel<<<blocks2D, THREADS_PER_BLOCK>>>(
            d_monthData, d_monthMask, spatialPixels);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Fill land pixels for this month
        int monthFilled = 0;
        for (int radius = 1; radius <= MAX_SEARCH_RADIUS; radius += RADIUS_INCREMENT) {
            int h_change = 0;
            CUDA_CHECK(cudaMemcpy(d_changeCount, &h_change, sizeof(int), 
                                 cudaMemcpyHostToDevice));
            
            fillLandPixelsKernel<<<blocks2D, THREADS_PER_BLOCK>>>(
                d_monthData, d_monthMask, OUTPUT_WIDTH, OUTPUT_HEIGHT, 
                radius, d_changeCount);
            
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(&h_change, d_changeCount, sizeof(int), 
                                 cudaMemcpyDeviceToHost));
            
            monthFilled += h_change;
            if (h_change == 0 && radius > 20) break;
        }
        
        totalFilledAllMonths += monthFilled;
    }
    
    std::cout << "      ✅ Filled " << totalFilledAllMonths << " total pixels across all months\n";
    
    CUDA_CHECK(cudaFree(d_mask));
    CUDA_CHECK(cudaFree(d_changeCount));
    
    // FIX #4: Temporal smoothing
    std::cout << "   🔄 Applying temporal cosine smoothing...\n";
    temporalSmoothKernel<<<blocks2D, THREADS_PER_BLOCK>>>(
        d_data, OUTPUT_WIDTH, OUTPUT_HEIGHT, MONTHS, 0.15);  // 15% smoothing
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Normalize
    std::cout << "   📏 Normalizing to [0,1]...\n";
    int blocksAll = (totalPixels + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    normalizeKernel<<<blocksAll, THREADS_PER_BLOCK>>>(
        d_data, totalPixels, config.minValue, config.maxValue);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy back
    CUDA_CHECK(cudaMemcpy(outputData.data(), d_data, 
                         totalPixels * sizeof(double), 
                         cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaFree(d_data));
    
    std::cout << "   ✅ Dataset complete\n\n";
}

//============================================================
// Host: Write Consolidated Binary (FIX #5)
//============================================================
void writeConsolidatedBinary(const std::string& outputPath,
                             const std::vector<DatasetConfig>& datasets,
                             const std::vector<std::vector<double>>& allData)
{
    std::cout << "💾 Writing consolidated OceanWorld_Climate.bin...\n";
    
    std::ofstream file(outputPath, std::ios::binary);
    if (!file) {
        std::cerr << "❌ Failed to create output file\n";
        return;
    }
    
    // Write header matching OceanWorldGenerator expectations
    int depth = MONTHS;
    int height = OUTPUT_HEIGHT;
    int width = OUTPUT_WIDTH;
    int varCount = datasets.size();
    
    file.write(reinterpret_cast<const char*>(&depth), sizeof(int));
    file.write(reinterpret_cast<const char*>(&height), sizeof(int));
    file.write(reinterpret_cast<const char*>(&width), sizeof(int));
    
    std::cout << "   Dimensions: " << depth << " × " << height << " × " << width << "\n";
    std::cout << "   Variables: " << varCount << "\n";
    
    // Write data in layout: [cell][variable][month]
    size_t spatialPixels = width * height;
    
    for (size_t cellIdx = 0; cellIdx < spatialPixels; ++cellIdx) {
        for (int varIdx = 0; varIdx < varCount; ++varIdx) {
            for (int month = 0; month < depth; ++month) {
                size_t dataIdx = month * spatialPixels + cellIdx;
                double value = allData[varIdx][dataIdx];
                file.write(reinterpret_cast<const char*>(&value), sizeof(double));
            }
        }
        
        if (cellIdx % 100000 == 0) {
            std::cout << "      Progress: " << (cellIdx * 100 / spatialPixels) << "%\r" << std::flush;
        }
    }
    
    file.close();
    
    size_t fileSize = spatialPixels * varCount * depth * sizeof(double);
    std::cout << "\n   ✅ Wrote " << (fileSize / (1024.0*1024.0*1024.0)) 
              << " GB\n";
}

//============================================================
// Main
//============================================================
int main()
{
    std::cout << "🌍 GPU Climate Processor (FIXED - All ChatGPT Recommendations)\n";
    std::cout << "═══════════════════════════════════════════════════════\n\n";
    
    std::vector<DatasetConfig> datasets = {
        // 3600x1800 files
        {"GPM_3IMERGM", 0.0, 300.0, 99999.0, false, "{NAME}_2024-{MM}-01_rgb_3600x1800.CSV"},
        {"MYD28M", -2.0, 35.0, 99999.0, false, "{NAME}_2024-{MM}-01_rgb_3600x1800.CSV"},
        {"MOD_LSTD_M", -50.0, 60.0, 99999.0, false, "{NAME}_2024-{MM}-01_rgb_3600x1800.CSV"},
        {"MOD_LSTN_M", -50.0, 60.0, 99999.0, false, "{NAME}_2024-{MM}-01_rgb_3600x1800.CSV"},
        {"MCD43C3_M_BSA", 0.0, 1.0, 99999.0, false, "{NAME}_2024-{MM}-01_rgb_3600x1800.CSV"},
        
        // 1440x720 files (CERES)
        {"CERES_NETFLUX_M", 0.0, 500.0, 99999.0, true, "{NAME}_2024-{MM}-01_rgb_1440x720.CSV"},
        {"CERES_SWFLUX_M", 0.0, 600.0, 99999.0, true, "{NAME}_2024-{MM}-01_rgb_1440x720.CSV"},
        {"CERES_LWFLUX_M", 0.0, 400.0, 99999.0, true, "{NAME}_2024-{MM}-01_rgb_1440x720.CSV"},
        {"CERES_INSOL_M", 0.0, 500.0, 99999.0, true, "{NAME}_2024-{MM}-01_rgb_1440x720.CSV"},
        
        // 3600x1800 files (continued)
        {"MOD15A2_M_LAI", 0.0, 7.0, 32767.0, false, "{NAME}_2024-{MM}-01_rgb_3600x1800.CSV"},
        {"NISE_D", 0.0, 1.0, 255.0, false, "{NAME}_2024-{MM}-01_rgb_3600x1800.CSV"},
        {"MOD10C1_M_SNOW", 0.0, 1.0, 255.0, false, "{NAME}_2024-{MM}-01_rgb_3600x1800.CSV"},
        {"MOD_NDVI_M", -0.2, 1.0, 32767.0, false, "{NAME}_2024-{MM}-01_rgb_3600x1800.CSV"}
    };
    
    // Process all 13 source datasets
    std::vector<std::vector<double>> allData(TOTAL_VARIABLES);  // 14 total (13 + AirTemp)
    
    for (size_t i = 0; i < datasets.size(); ++i) {
        processDataset(datasets[i], allData[i]);
    }
    
    // Derive AirTemp from SST (variable index 1)
    std::cout << "🌡️  Deriving Air Temperature from SST...\n";
    size_t sstIndex = 1;  // MYD28M is at index 1
    size_t totalPixels = OUTPUT_WIDTH * OUTPUT_HEIGHT * MONTHS;
    
    allData[13].resize(totalPixels);  // AirTemp goes at index 13
    
    for (size_t i = 0; i < totalPixels; ++i) {
        double sst = allData[sstIndex][i];
        if (isfinite(sst)) {
            // AirTemp ≈ SST - 2°C, then normalize to [0,1] using same range as SST
            double sstCelsius = sst * (35.0 - (-2.0)) + (-2.0);  // Denormalize
            double airTempCelsius = sstCelsius - 2.0;             // Derive
            double airTempNorm = (airTempCelsius - (-4.0)) / (33.0 - (-4.0));  // Normalize
            allData[13][i] = fmax(0.0, fmin(1.0, airTempNorm));
        } else {
            allData[13][i] = sst;  // Copy NaN if SST is invalid
        }
    }
    std::cout << "   ✅ Air Temperature derived\n\n";
    
    // Write consolidated output
    writeConsolidatedBinary("./output/OceanWorld_Climate.bin", datasets, allData);
    
    std::cout << "\n═══════════════════════════════════════════════════════\n";
    std::cout << "✅ All processing complete!\n";
    std::cout << "   Output: ./output/OceanWorld_Climate.bin\n";
    
    return 0;
}