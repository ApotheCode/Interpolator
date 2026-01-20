#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <limits>
#include <string>

const int WIDTH = 3600;
const int HEIGHT = 1800;
const int MONTHS = 12;

const float M_PI = 3.14159265358979323846;

const std::vector<double> PLACEHOLDERS = {99999, -9999, 9999, 1e20, -1e20};

bool isPlaceholder(double val) {
    for (double ph : PLACEHOLDERS) {
        if (std::abs(val - ph) < 1e-6) return true;
    }
    return false;
}

void loadCSV(const std::string& filename, std::vector<std::vector<double>>& grid) {
    std::ifstream file(filename);
    std::string line;
    int row = 0;

    while (std::getline(file, line) && row < HEIGHT) {
        std::stringstream ss(line);
        std::string val;
        int col = 0;

        while (std::getline(ss, val, ',') && col < WIDTH) {
            double num = std::stod(val);
            if (isPlaceholder(num)) num = std::numeric_limits<double>::quiet_NaN();
            grid[row][col] = num;
            col++;
        }
        row++;
    }
    file.close();
}

void normalizeGrid(std::vector<std::vector<double>>& grid) {
    double minVal = std::numeric_limits<double>::max();
    double maxVal = std::numeric_limits<double>::lowest();

    for (const auto& row : grid) {
        for (double val : row) {
            if (!std::isnan(val)) {
                minVal = std::min(minVal, val);
                maxVal = std::max(maxVal, val);
            }
        }
    }

    double range = maxVal - minVal;
    for (auto& row : grid) {
        for (double& val : row) {
            if (!std::isnan(val)) {
                val = (val - minVal) / range;
            }
        }
    }
}

// Sinusoidal interpolation over time for each (lat, lon)
void interpolateTime(std::vector<std::vector<std::vector<double>>>& data) {
    for (int i = 0; i < HEIGHT; ++i) {
        for (int j = 0; j < WIDTH; ++j) {
            std::vector<double> timeSeries(MONTHS);
            for (int t = 0; t < MONTHS; ++t)
                timeSeries[t] = data[t][i][j];

            for (int t = 0; t < MONTHS; ++t) {
                if (std::isnan(timeSeries[t])) {
                    int prev = t - 1;
                    while (prev >= 0 && std::isnan(timeSeries[prev])) --prev;
                    int next = t + 1;
                    while (next < MONTHS && std::isnan(timeSeries[next])) ++next;

                    if (prev >= 0 && next < MONTHS) {
                        double A = timeSeries[prev];
                        double B = timeSeries[next];
                        double x = double(t - prev) / (next - prev);
                        timeSeries[t] = A + (B - A) * (1 - std::cos(M_PI * x)) / 2.0;
                    } else if (prev >= 0) {
                        timeSeries[t] = timeSeries[prev];
                    } else if (next < MONTHS) {
                        timeSeries[t] = timeSeries[next];
                    } else {
                        timeSeries[t] = 0.0;
                    }
                }
            }

            for (int t = 0; t < MONTHS; ++t)
                data[t][i][j] = timeSeries[t];
        }
    }
}

void saveCSV(const std::string& filename, const std::vector<std::vector<double>>& grid) {
    std::ofstream file(filename);
    for (const auto& row : grid) {
        for (int j = 0; j < WIDTH; ++j) {
            file << row[j];
            if (j != WIDTH - 1) file << ",";
        }
        file << "\n";
    }
    file.close();
}

void saveBIN(const std::string& filename, const std::vector<std::vector<std::vector<double>>>& data) {
    std::ofstream file(filename, std::ios::binary);
    for (int t = 0; t < MONTHS; ++t)
        for (int i = 0; i < HEIGHT; ++i)
            for (int j = 0; j < WIDTH; ++j)
                file.write(reinterpret_cast<const char*>(&data[t][i][j]), sizeof(double));
    file.close();
}

int main() {
    std::string datasetName;
    std::cout << "Enter the climate variable name (e.g., rainfall, temperature, sst): ";
    std::getline(std::cin, datasetName);

    std::vector<std::vector<std::vector<double>>> dataGrid(
        MONTHS, std::vector<std::vector<double>>(HEIGHT, std::vector<double>(WIDTH))
    );

    for (int month = 0; month < MONTHS; ++month) {
        std::string filename = std::string("GPM_3IMERGM_2024-") + (month + 1 < 10 ? "0" : "") + std::to_string(month + 1) + "-01_rgb_3600x1800.CSV";
        loadCSV(filename, dataGrid[month]);
        normalizeGrid(dataGrid[month]);
        std::cout << "Loaded and normalized " << filename << "\n";
    }

    interpolateTime(dataGrid);
    std::cout << "Sinusoidal interpolation complete.\n";

    for (int month = 0; month < MONTHS; ++month) {
        std::string outname = datasetName + "_interpolated_" + (month + 1 < 10 ? "0" : "") + std::to_string(month + 1) + ".csv";
        saveCSV(outname, dataGrid[month]);
    }

    saveBIN(datasetName + "_interpolated_all.bin", dataGrid);
    std::cout << "Saved all outputs for dataset: " << datasetName << "\n";

    return 0;
}
