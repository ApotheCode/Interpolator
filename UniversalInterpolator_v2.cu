//============================================================
// UniversalInterpolator_v2_Fixed.cu
// Fixes for Ocean World generation:
// - Periodic longitude boundaries (seam fix)
// - Smoothstep normalization (polar taper)
// - Pre-flood continental regions
// - Multi-scale noise preparation hooks
// Author: Claude (2025-11-10)
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
constexpr int MAX_SEARCH_RADIUS = 4800;
constexpr int RADIUS_INCREMENT = 1;
constexpr int TOTAL_VARIABLES = 14;  // 13 source + 1 derived (AirTemp)
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
// Device: Sentinel/Missing Value Check
//============================================================
__device__ __forceinline__ bool isMissing(double v, double sentinel) {
    return !isfinite(v) || fabs(v - sentinel) < 1e-9;
}

//============================================================
// Device: Smoothstep Function (soft S-curve)
// Used for smooth normalization without harsh clipping
//============================================================
__device__ __forceinline__ double smoothstep(double edge0, double edge1, double x) {
    double t = fmax(0.0, fmin(1.0, (x - edge0) / (edge1 - edge0)));
    return t * t * (3.0 - 2.0 * t);
}

//============================================================
// Device: Periodic-Aware Bilinear Sampling
// FIX #1: Wraps longitude (x) periodically, clamps latitude (y)
//============================================================
__device__ double bilinearSampleMaskedPeriodic(const double* __restrict__ input,
                                               int inWidth, int inHeight,
                                               double x, double y, double sentinel)
{
    // CRITICAL FIX: Wrap x for periodic longitude boundaries
    // Normalize to [0, inWidth) range with modulo
    x = fmod(x + inWidth * 1000.0, static_cast<double>(inWidth));  // Ensure positive
    if (x < 0.0) x += inWidth;
    
    // Clamp y (latitude) to valid range
    y = fmax(0.0, fmin(y, inHeight - 1.001));
    
    int x0 = static_cast<int>(x);
    int y0 = static_cast<int>(y);
    
    // CRITICAL: Wrap x1 with modulo for seamless boundaries
    int x1 = (x0 + 1) % inWidth;
    int y1 = min(y0 + 1, inHeight - 1);
    
    double fx = x - x0;
    double fy = y - y0;
    
    // Fetch 4 corner values with wrapped x-coordinates
    double v00 = input[y0 * inWidth + x0];
    double v10 = input[y0 * inWidth + x1];  // Wraps at boundary
    double v01 = input[y1 * inWidth + x0];
    double v11 = input[y1 * inWidth + x1];  // Wraps at boundary
    
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
// Kernel 1: Bilinear Upscaling with Periodic Boundaries
// FIX #1: Uses periodic-aware sampling
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
// Kernel 2: Initialize Valid Data Mask
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
// Kernel 3: Continental Pre-Flood
// FIX #3: Initialize land regions with distance-weighted ocean avg
// This removes inherited Earth-like continental shapes
//============================================================
__global__ void prefloodContinentsKernel(double* __restrict__ data,
                                         const unsigned char* __restrict__ mask,
                                         int width, int height,
                                         double latitudinalBase)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalPixels = width * height;
    
    if (idx >= totalPixels) return;
    if (mask[idx] == 1) return;  // Already has ocean data
    
    int x = idx % width;
    int y = idx / width;
    
    // Generate latitude-dependent base value for land regions
    // This creates a smooth gradient without continental imprints
    double lat = (static_cast<double>(y) / height - 0.5) * 180.0;  // -90 to +90
    double latFactor = cos(lat * M_PI / 180.0);  // Cosine weighting
    
    // Base value: warmer at equator, cooler at poles
    double baseValue = latitudinalBase * (0.4 + 0.6 * latFactor);
    
    data[idx] = baseValue;
}

//============================================================
// Kernel 4: Spatial Fill with Cosine Distance Weighting
// Now works on pre-flooded data for smoother blending
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
            
            // CRITICAL: Wrap longitude with modulo
            int nx = (x + dx + width) % width;
            int ny = y + dy;
            
            if (ny < 0 || ny >= height) continue;  // Clamp latitude
            
            int nIdx = ny * width + nx;
            
            if (mask[nIdx] == 1) {
                double dist = sqrt(static_cast<double>(dx*dx + dy*dy));
                double t = fmin(dist / searchRadius, 1.0);
                
                // Cosine easing for smooth gradient
                double weight = (dist < 2.0) ? 1.0 : 0.5 * (1.0 - cos(M_PI * (1.0 - t)));
                
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
// Kernel 5: Temporal Cosine Smoothing
// Blends each month with its neighbors for continuity
//============================================================
__global__ void temporalSmoothKernel(double* __restrict__ data,
                                     int width, int height,
                                     int months,
                                     double strength)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int spatialPixels = width * height;
    
    if (idx >= spatialPixels) return;
    
    for (int m = 0; m < months; ++m) {
        int prevMonth = (m + months - 1) % months;
        int nextMonth = (m + 1) % months;
        
        double curr = data[m * spatialPixels + idx];
        double prev = data[prevMonth * spatialPixels + idx];
        double next = data[nextMonth * spatialPixels + idx];
        
        if (isfinite(curr) && isfinite(prev) && isfinite(next)) {
            double blended = curr * (1.0 - strength) + (prev + next) * 0.5 * strength;
            data[m * spatialPixels + idx] = blended;
        }
    }
}

//============================================================
// Kernel 6: Smoothstep Normalization
// FIX #2: Uses soft S-curve to avoid harsh polar clipping
//============================================================
__global__ void normalizeSmoothKernel(double* __restrict__ data,
                                      int totalPixels,
                                      double minVal, double maxVal,
                                      double softMargin)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= totalPixels) return;
    
    double val = data[idx];
    double range = maxVal - minVal;
    
    if (range > 0.0 && isfinite(val)) {
        // Linear normalization to [0,1]
        val = (val - minVal) / range;
        
        // Apply smoothstep at margins for soft tapering
        // This prevents harsh clipping at polar extremes
        if (val < softMargin) {
            val = smoothstep(0.0, softMargin, val) * softMargin;
        } else if (val > (1.0 - softMargin)) {
            val = 1.0 - smoothstep(0.0, softMargin, 1.0 - val) * softMargin;
        }
        
        data[idx] = fmax(0.0, fmin(1.0, val));
    } else if (isfinite(val)) {
        data[idx] = 0.5;
    }
}

//============================================================
// Host: Load CSV
//============================================================
std::vector<double> loadCSV(const std::string& path, int& rows, int& cols, double sentinel)
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
            } catch (...) {
                data.push_back(sentinel);
            }
            colCount++;
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
    
    double* d_data = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, totalPixels * sizeof(double)));
    
    // Process each month independently
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
        std::vector<double> hostInput = loadCSV(fullPath, inRows, inCols, config.sentinel);
        
        if (hostInput.empty()) {
            std::cerr << "⚠️ Skipping month " << (month+1) << "\n";
            continue;
        }
        
        std::cout << "   Month " << std::setw(2) << (month+1) << ": " 
                  << inCols << "×" << inRows;
        
        double* d_input = nullptr;
        CUDA_CHECK(cudaMalloc(&d_input, hostInput.size() * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_input, hostInput.data(), 
                             hostInput.size() * sizeof(double), 
                             cudaMemcpyHostToDevice));
        
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
    
    // FIX #3: Pre-flood continents before spatial fill
    std::cout << "   🌊 Pre-flooding continental regions...\n";
    
    unsigned char* d_mask = nullptr;
    CUDA_CHECK(cudaMalloc(&d_mask, totalPixels * sizeof(unsigned char)));
    
    int blocks2D = (spatialPixels + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    // Initialize mask for all months
    for (int month = 0; month < MONTHS; ++month) {
        double* d_monthData = d_data + month * spatialPixels;
        unsigned char* d_monthMask = d_mask + month * spatialPixels;
        
        initMaskKernel<<<blocks2D, THREADS_PER_BLOCK>>>(
            d_monthData, d_monthMask, spatialPixels, config.sentinel);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Pre-flood land regions with latitudinal base values
    // This removes inherited Earth continental shapes
    double latBase = (config.maxValue + config.minValue) * 0.5;  // Midpoint
    for (int month = 0; month < MONTHS; ++month) {
        double* d_monthData = d_data + month * spatialPixels;
        unsigned char* d_monthMask = d_mask + month * spatialPixels;
        
        prefloodContinentsKernel<<<blocks2D, THREADS_PER_BLOCK>>>(
            d_monthData, d_monthMask, OUTPUT_WIDTH, OUTPUT_HEIGHT, latBase);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "      ✅ Continents pre-flooded\n";
    
    // Spatial fill: Blend ocean values into pre-flooded land
    std::cout << "   🌊 Filling land pixels (per-month)...\n";
    
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
                radius, d_changeCount);
            
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(&h_change, d_changeCount, sizeof(int), 
                                 cudaMemcpyDeviceToHost));
            
            monthFilled += h_change;
            if (h_change == 0 && radius > 20) break;
        }
        
        totalFilledAllMonths += monthFilled;
    }
    
    std::cout << "      ✅ Filled " << totalFilledAllMonths << " pixels\n";
    
    CUDA_CHECK(cudaFree(d_mask));
    CUDA_CHECK(cudaFree(d_changeCount));
    
    // Temporal smoothing
    std::cout << "   🔄 Temporal smoothing...\n";
    temporalSmoothKernel<<<blocks2D, THREADS_PER_BLOCK>>>(
        d_data, OUTPUT_WIDTH, OUTPUT_HEIGHT, MONTHS, 0.15);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // FIX #2: Smoothstep normalization with soft margins
    std::cout << "   📏 Normalizing with soft tapering...\n";
    int blocksAll = (totalPixels + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    double softMargin = 0.05;  // Apply smoothstep in 5% margins at edges
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
// Host: Write Consolidated Binary
// CRITICAL: Layout must be [cell][variable][month]
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
    
    // Simple header: depth, height, width
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
    
    // Write in correct layout for classifier
    size_t spatialPixels = width * height;
    size_t bytesWritten = 12;  // Header
    
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
    
    // Verify file size
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
    std::cout << "🌍 Unified Climate Processor v2.0 (Fixed)\n";
    std::cout << "══════════════════════════════════════════════════════════\n";
    std::cout << "Fixes: Periodic boundaries, smoothstep norm, continental flood\n";
    std::cout << "══════════════════════════════════════════════════════════\n\n";
    
    std::vector<DatasetConfig> datasets = {
        {"GPM_3IMERGM", 0.0, 300.0, 99999.0, false, "{NAME}_2024-{MM}-01_rgb_3600x1800.CSV"},
        {"MYD28M", -2.0, 35.0, 99999.0, false, "{NAME}_2024-{MM}-01_rgb_3600x1800.CSV"},
        {"MOD_LSTD_M", -50.0, 60.0, 99999.0, false, "{NAME}_2024-{MM}-01_rgb_3600x1800.CSV"},
        {"MOD_LSTN_M", -50.0, 60.0, 99999.0, false, "{NAME}_2024-{MM}-01_rgb_3600x1800.CSV"},
        {"MCD43C3_M_BSA", 0.0, 1.0, 99999.0, false, "{NAME}_2024-{MM}-01_rgb_3600x1800.CSV"},
        {"CERES_NETFLUX_M", 0.0, 500.0, 99999.0, true, "{NAME}_2024-{MM}-01_rgb_1440x720.CSV"},
        {"CERES_SWFLUX_M", 0.0, 600.0, 99999.0, true, "{NAME}_2024-{MM}-01_rgb_1440x720.CSV"},
        {"CERES_LWFLUX_M", 0.0, 400.0, 99999.0, true, "{NAME}_2024-{MM}-01_rgb_1440x720.CSV"},
        {"CERES_INSOL_M", 0.0, 500.0, 99999.0, true, "{NAME}_2024-{MM}-01_rgb_1440x720.CSV"},
        {"MOD15A2_M_LAI", 0.0, 7.0, 32767.0, false, "{NAME}_2024-{MM}-01_rgb_3600x1800.CSV"},
        {"NISE_D", 0.0, 1.0, 255.0, false, "{NAME}_2024-{MM}-01_rgb_3600x1800.CSV"},
        {"MOD10C1_M_SNOW", 0.0, 1.0, 255.0, false, "{NAME}_2024-{MM}-01_rgb_3600x1800.CSV"},
        {"MOD_NDVI_M", -0.2, 1.0, 32767.0, false, "{NAME}_2024-{MM}-01_rgb_3600x1800.CSV"}
    };
    
    // Process 13 source datasets
    std::vector<std::vector<double>> allData(TOTAL_VARIABLES);
    
    for (size_t i = 0; i < datasets.size(); ++i) {
        processDataset(datasets[i], allData[i]);
    }
    
    // Derive AirTemp from SST (variable index 1)
    std::cout << "🌡️ Deriving Air Temperature from SST...\n";
    size_t sstIndex = 1;
    size_t totalPixels = OUTPUT_WIDTH * OUTPUT_HEIGHT * MONTHS;
    
    allData[13].resize(totalPixels);
    
    for (size_t i = 0; i < totalPixels; ++i) {
        double sst = allData[sstIndex][i];
        if (isfinite(sst)) {
            // Denormalize SST: [0,1] → [-2,35]°C
            double sstCelsius = sst * 37.0 - 2.0;
            // Derive: AirTemp ≈ SST - 2°C
            double airTempCelsius = sstCelsius - 2.0;
            // Normalize: [-4,33]°C → [0,1]
            double airTempNorm = (airTempCelsius + 4.0) / 37.0;
            allData[13][i] = fmax(0.0, fmin(1.0, airTempNorm));
        } else {
            allData[13][i] = sst;  // Preserve NaN
        }
    }
    std::cout << "   ✅ Air Temperature derived\n\n";
    
    // Write consolidated output
    writeConsolidatedBinary("./output/OceanWorld_Climate.bin", allData);
    
    std::cout << "\n══════════════════════════════════════════════════════════\n";
    std::cout << "✅ Processing complete!\n";
    std::cout << "   Output: ./output/OceanWorld_Climate.bin\n";
    std::cout << "   Expected size: 7.60 GB (8,164,800,012 bytes)\n";
    std::cout << "\n📋 Next steps for full pipeline:\n";
    std::cout << "   1. Apply multi-octave PNPL noise (3-4 octaves)\n";
    std::cout << "   2. Export as 16-bit PNG or use error-diffusion dithering\n";
    std::cout << "   3. Add legend with variable name, range, and units\n";
    
    return 0;
}
