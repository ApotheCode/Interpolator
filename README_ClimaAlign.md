# ClimaAlign

**ClimaAlign** is a high-performance C++ tool designed to normalize, interpolate, and upscale large-scale gridded climate datasets for use in machine learning, climate modeling, and biome generation pipelines. It ensures consistency across time and resolution, preparing complex environmental data for analytical or generative applications such as Earth-like planet simulations.

---

## 🌍 Features

- ✅ Supports multi-format climate datasets (CSV input)
- ✅ Detects and handles placeholder/missing values
- ✅ Normalizes values for machine learning compatibility
- ✅ Performs monthly interpolation with sinusoidal smoothing
- ✅ Automatically upscales lower-resolution datasets to 3600×1800
- ✅ Outputs interpolated `.csv` files and a combined `.bin` stack

---

## 🧠 Use Case

Originally built for the **Biome Map Generator** project, ClimaAlign is suitable for:
- Climate-driven biome simulation
- Planetary surface modeling
- Preprocessing climate data for ML pipelines
- Training AI systems on normalized environmental grids

---

## 📦 Supported Datasets

| Dataset Name         | Description                  | Example Value Range | Requires Interpolation |
|----------------------|------------------------------|----------------------|-------------------------|
| `GPM_3IMERGM`        | Rainfall                     | 0–300 mm             | Yes                     |
| `MYD28M`             | Sea Surface Temp (SST)       | -2°C to 35°C         | Yes                     |
| `MOD_LSTD_M`         | Land Surface Temp (Day)      | -50°C to 60°C        | Yes                     |
| `MOD_LSTN_M`         | Land Surface Temp (Night)    | -50°C to 60°C        | Yes                     |
| `MCD43C3_M_BSA`      | Albedo                       | 0.0 to 1.0           | Optional                |
| `NetRadiation`       | Net surface radiation        | 0 to 500 W/m²        | Optional                |
| `ShortwaveRadiation`| Incoming shortwave radiation | 0 to 600 W/m²        | Optional                |
| `LongwaveRadiation` | Outgoing longwave radiation  | 0 to 400 W/m²        | Optional                |
| `CERES_INSOL_M`      | Solar Insolation             | 0 to 500 W/m²        | Optional (1440×720)     |

---

## 🛠️ Build & Run

```bash
g++ -std=c++17 -O3 -o climaalign UniversalInterpolator_Updated.cpp
./climaalign
```

### 📁 Folder Structure

- `./input/` — Monthly input CSV files (one per month)
- `./output/` — Normalized & interpolated `.csv` and `.bin` files

---

## 📈 Output

- `output/<DatasetName>_interpolated_01.csv` → January
- ...
- `output/<DatasetName>_interpolated_12.csv` → December
- `output/<DatasetName>_interpolated.bin` → Combined 3D matrix for fast access

---

## ⚙️ Interpolation Method

- **Sinusoidal Smoothing** across missing months
- Handles wrap-around (e.g., December → January)
- Preserves realistic environmental transitions

---

## 🧪 Example Use Cases

- Training climate-aware AI models
- Building simulations of "alternate Earths"
- Generating procedural biomes for games or research
- Filling in sparse satellite records with inferred values

---

## 👨‍💻 Author

Developed by **Mark Devereux** as part of the *Biome Map Generator* project for planetary science and visualization.

---

## 📄 License

MIT License (insert your actual license if different).