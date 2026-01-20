
// Multithreaded Universal Interpolator for Climate Datasets
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <filesystem>
#include <algorithm>
#include <thread>

const int TARGET_WIDTH = 3600;
const int TARGET_HEIGHT = 1800;
const int MONTHS = 12;
const float M_PI = 3.14159265358979323846;
const int THREAD_COUNT = std::thread::hardware_concurrency();

struct DatasetConfig {
    std::string name;
    double minValue = 0.0;
    double maxValue = 1.0;
    double placeholder = 99999.0;
    bool normalize = true;
};

std::vector<DatasetConfig> predefinedDatasets = {
    {"GPM_3IMERGM", 0.0, 300.0, 99999.0, true}, /** Rainfall */
    {"MYD28M", -2.0, 35.0, 99999.0, true}, /** SST */
    {"MOD_LSTD_M", -50.0, 60.0, 99999.0, true}, /** LST Day Time */
    {"MOD_LSTN_M", -50.0, 60.0, 99999.0, true}, /** LST Night Time */
    {"MCD43C3_M_BSA", 0.0, 1.0, 99999.0, true}, /** Albedo */
    {"CERES_NETFLUX_M", 0.0, 500.0, 99999.0, true}, /** Net Radiation Flux */
    {"CERES_SWFLUX_M", 0.0, 600.0, 99999.0, true}, /** Shortwave Radiation Flux */
    {"CERES_LWFLUX_M", 0.0, 400.0, 99999.0, true}, /** Longwave Radiation Flux */
    {"CERES_INSOL_M_", 0.0, 500.0, 99999.0, true} /** Solar Insolation */
};

DatasetConfig selectDataset() {
    std::cout << "Available datasets:\n";
    for (size_t i = 0; i < predefinedDatasets.size(); ++i) {
        std::cout << "  [" << i + 1 << "] " << predefinedDatasets[i].name << "\n";
    }
    int choice;
    std::cout << "Select dataset type: ";
    std::cin >> choice;
    return predefinedDatasets[std::max(0, std::min((int)predefinedDatasets.size(), choice - 1))];
}

using Grid = std::vector<std::vector<double>>;

Grid upscaleBilinear(const Grid& input, int targetWidth, int targetHeight) {
    int srcHeight = input.size();
    int srcWidth = input[0].size();
    Grid output(targetHeight, std::vector<double>(targetWidth));

    for (int y = 0; y < targetHeight; ++y) {
        for (int x = 0; x < targetWidth; ++x) {
            double gy = (double)(y * (srcHeight - 1)) / (targetHeight - 1);
            double gx = (double)(x * (srcWidth - 1)) / (targetWidth - 1);
            int y0 = (int)gy, x0 = (int)gx;
            int y1 = std::min(y0 + 1, srcHeight - 1);
            int x1 = std::min(x0 + 1, srcWidth - 1);
            double dy = gy - y0, dx = gx - x0;

            double top = (1 - dx) * input[y0][x0] + dx * input[y0][x1];
            double bottom = (1 - dx) * input[y1][x0] + dx * input[y1][x1];
            output[y][x] = (1 - dy) * top + dy * bottom;
        }
    }
    return output;
}

Grid loadCSV(const std::string& filepath, double placeholder, int& outRows, int& outCols) {
    Grid grid;
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "❌ Failed to open: " << filepath << "\n";
        return grid;
    }
    std::string line;
    while (getline(file, line)) {
        std::stringstream ss(line);
        std::string val;
        std::vector<double> row;
        while (getline(ss, val, ',')) {
            try {
                row.push_back(std::stod(val));
            } catch (...) {
                row.push_back(placeholder);
            }
        }
        grid.push_back(row);
    }
    outRows = grid.size();
    outCols = grid.empty() ? 0 : grid[0].size();
    return grid;
}

void normalize(Grid& grid, double minVal, double maxVal, double placeholder) {
    double range = maxVal - minVal;
    int height = grid.size();
    int width = grid[0].size();
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            if (grid[i][j] == placeholder) {
                grid[i][j] = std::numeric_limits<double>::quiet_NaN();
            }
        }
    }
}

void interpolateWorker(const std::vector<Grid>& data, std::vector<Grid>& result, double placeholder, int startRow, int endRow) {
    int width = result[0][0].size();
    for (int i = startRow; i < endRow; ++i) {
        for (int j = 0; j < width; ++j) {
            std::vector<double> monthly(MONTHS, placeholder);
            for (int m = 0; m < MONTHS; ++m) monthly[m] = data[m][i][j];
            for (int m = 0; m < MONTHS; ++m) {
                if (std::isnan(monthly[m])) {
                    int prev = m, next = m;
                    while (std::isnan(monthly[prev])) prev = (prev - 1 + MONTHS) % MONTHS;
                    while (std::isnan(monthly[next])) next = (next + 1) % MONTHS;
                    int d = (next - prev + MONTHS) % MONTHS;
                    int interpPos = (m - prev + MONTHS) % MONTHS;
                    double t = (double)interpPos / d;
                    double sinT = 0.5 * (1 - cos(t * M_PI));
                    monthly[m] = monthly[prev] * (1 - sinT) + monthly[next] * sinT;
                }
                result[m][i][j] = monthly[m];
            }
        }
    }
}

std::vector<Grid> interpolateTime(const std::vector<Grid>& data, double placeholder) {
    int height = data[0].size();
    int width = data[0][0].size();

    std::vector<Grid> result(MONTHS, Grid(height, std::vector<double>(width, placeholder)));
    std::vector<std::thread> threads;
    int rowsPerThread = height / THREAD_COUNT;
    for (int t = 0; t < THREAD_COUNT; ++t) {
        int startRow = t * rowsPerThread;
        int endRow = (t == THREAD_COUNT - 1) ? height : startRow + rowsPerThread;
        threads.emplace_back(interpolateWorker, std::ref(data), std::ref(result), placeholder, startRow, endRow);
    }
    for (auto& thread : threads) thread.join();
    return result;
}

void saveCSV(const Grid& grid, const std::string& path) {
    std::ofstream file(path);
    for (const auto& row : grid) {
        for (size_t j = 0; j < row.size(); ++j) {
            file << std::fixed << std::setprecision(6) << row[j];
            if (j < row.size() - 1) file << ",";
        }
        file << "\n";
    }
}

void saveBIN(const std::vector<Grid>& data, const std::string& path) {
    std::ofstream file(path, std::ios::binary);
    for (const auto& grid : data)
        for (const auto& row : grid)
            for (double val : row)
                file.write(reinterpret_cast<const char*>(&val), sizeof(double));
}

int main() {
    DatasetConfig config = selectDataset();
    std::vector<Grid> monthlyData;

    for (const DatasetConfig& config : predefinedDatasets) {
        std::cout << "\n========================================\n";
        std::cout << "🛰️  Processing dataset: " << config.name << "\n";

        std::vector<Grid> monthlyData;

        for (int month = 0; month < MONTHS; ++month) {
            std::stringstream baseName;
            baseName << "./input/" << config.name << "_2024-"
                     << std::setfill('0') << std::setw(2) << (month + 1)
                     << "-01";

            std::string path3600 = baseName.str() + "_rgb_3600x1800.csv";
            std::string path1440 = baseName.str() + "_rgb_1440x720.csv";

            std::ifstream test3600(path3600);
            std::ifstream test1440(path1440);

            std::string selectedPath;
            if (test3600.is_open()) {
                selectedPath = path3600;
                std::cout << "→ Found 3600x1800: " << selectedPath << "\n";
            } else if (test1440.is_open()) {
                selectedPath = path1440;
                std::cout << "→ Found 1440x720: " << selectedPath << "\n";
            } else {
                std::cerr << "❌ Neither resolution found for month " << (month + 1) << "\n";
                continue;
            }

            int rows, cols;
            Grid grid = loadCSV(selectedPath, config.placeholder, rows, cols);

            if (rows == 0 || cols == 0) {
                std::cerr << "❌ Skipping invalid grid: " << selectedPath << "\n";
                continue;
            }

            if (rows != TARGET_HEIGHT || cols != TARGET_WIDTH) {
                std::cout << "⬆️  Upscaling " << rows << "x" << cols << " to " << TARGET_WIDTH << "x" << TARGET_HEIGHT << "\n";
                grid = upscaleBilinear(grid, TARGET_WIDTH, TARGET_HEIGHT);
            }

            if (config.normalize) {
                normalize(grid, config.minValue, config.maxValue, config.placeholder);
            }

            monthlyData.push_back(grid);
        }

        if (monthlyData.size() != 12) {
            std::cerr << "⚠️ Skipping dataset due to incomplete months: " << config.name << "\n";
            continue;
        }

        std::cout << "⏳ Interpolating missing values...\n";
        auto interpolated = interpolateTime(monthlyData, config.placeholder);

        for (int month = 0; month < MONTHS; ++month) {
            std::stringstream filename;
            filename << "./output/" << config.name << "_interpolated_"
                     << std::setfill('0') << std::setw(2) << (month + 1) << ".csv";
            saveCSV(interpolated[month], filename.str());
        }

        std::cout << "💾 Saving binary stack...\n";
        saveBIN(interpolated, "./output/" + config.name + "_interpolated.bin");
        std::cout << "✅ Finished: " << config.name << "\n";
    }

    std::cout << "\n🏁 All datasets completed!\n";
    return 0;
}
