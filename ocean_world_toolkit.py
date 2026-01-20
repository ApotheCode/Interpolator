#!/usr/bin/env python3
"""
Ocean World Validation & Post-Processing Toolkit
Companion tool for UniversalInterpolator_v2_Fixed.cu

Features:
1. Validate seam continuity, color depth, continental ghosts
2. Apply multi-octave PNPL with latitude weighting
3. Export as 16-bit PNG with proper colormap
4. Add scientific legend with metadata

Usage:
    python ocean_world_toolkit.py validate output.npy
    python ocean_world_toolkit.py process output.npy --pnpl-octaves 3
    python ocean_world_toolkit.py export output.npy --format png16
"""

import numpy as np
import argparse
import sys
from pathlib import Path
from typing import Tuple, Optional
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image, ImageEnhance
from scipy import ndimage
from noise import pnoise2  # pip install noise

class OceanWorldValidator:
    """Validates Ocean World baseline against acceptance criteria"""
    
    def __init__(self, data: np.ndarray):
        """
        Args:
            data: [height, width] or [height, width, channels] array
        """
        if data.ndim == 3:
            # Convert RGB to grayscale using standard weights
            data = 0.299 * data[:,:,0] + 0.587 * data[:,:,1] + 0.114 * data[:,:,2]
        
        self.data = data
        self.height, self.width = data.shape
        self.passed_tests = []
        self.failed_tests = []
    
    def test_seam_continuity(self, threshold: float = 1/255) -> bool:
        """Test for equirectangular seam discontinuity"""
        col_left = self.data[:, 0]
        col_right = self.data[:, -1]
        
        abs_diff = np.abs(col_left - col_right)
        max_diff = abs_diff.max()
        mean_diff = abs_diff.mean()
        
        passed = max_diff < threshold
        
        print(f"\n{'='*60}")
        print(f"TEST 1: Seam Continuity")
        print(f"{'='*60}")
        print(f"Max difference:  {max_diff:.6f} (threshold: {threshold:.6f})")
        print(f"Mean difference: {mean_diff:.6f}")
        print(f"Affected rows:   {np.sum(abs_diff > threshold)}/{self.height}")
        print(f"Status: {'✅ PASS' if passed else '❌ FAIL'}")
        
        if passed:
            self.passed_tests.append("Seam Continuity")
        else:
            self.failed_tests.append("Seam Continuity")
            print(f"\n⚠️  Top 5 problematic latitudes:")
            worst_rows = np.argsort(abs_diff)[-5:][::-1]
            for row in worst_rows:
                lat = (row / self.height - 0.5) * 180
                print(f"   Row {row} (lat {lat:.1f}°): diff = {abs_diff[row]:.6f}")
        
        return passed
    
    def test_color_richness(self, min_unique: int = 1024) -> bool:
        """Test for sufficient color depth (anti-posterization)"""
        # Quantize to 16-bit for counting
        quantized = (self.data * 65535).astype(np.uint16)
        unique_values = len(np.unique(quantized))
        
        passed = unique_values >= min_unique
        
        print(f"\n{'='*60}")
        print(f"TEST 2: Color Richness")
        print(f"{'='*60}")
        print(f"Unique values: {unique_values} (min: {min_unique})")
        print(f"Bit depth:     {np.log2(unique_values):.1f} effective bits")
        print(f"Status: {'✅ PASS' if passed else '❌ FAIL'}")
        
        if passed:
            self.passed_tests.append("Color Richness")
        else:
            self.failed_tests.append("Color Richness")
            print(f"\n⚠️  Only {unique_values} colors detected")
            print(f"   Recommendation: Export as 16-bit PNG or apply dithering")
        
        return passed
    
    def test_continental_ghosts(self, contrast_factor: float = 5.0) -> bool:
        """Test for residual Earth-like continental patterns"""
        # Enhance contrast to reveal subtle features
        normalized = (self.data - self.data.min()) / (self.data.max() - self.data.min())
        enhanced = np.clip((normalized - 0.5) * contrast_factor + 0.5, 0, 1)
        
        # Compute spatial frequency content
        fft2d = np.fft.fft2(self.data)
        power_spectrum = np.abs(fft2d)
        
        # Check for suspicious peaks in medium frequencies (continental scales)
        mid_freq_slice = power_spectrum[self.height//4:self.height//2, 
                                       self.width//4:self.width//2]
        mid_freq_power = np.mean(mid_freq_slice)
        
        # Compare to expected smooth gradient power
        low_freq_slice = power_spectrum[:self.height//8, :self.width//8]
        low_freq_power = np.mean(low_freq_slice)
        
        ratio = mid_freq_power / low_freq_power
        
        # Heuristic: ratio should be < 0.1 for smooth ocean world
        passed = ratio < 0.1
        
        print(f"\n{'='*60}")
        print(f"TEST 3: Continental Ghost Detection")
        print(f"{'='*60}")
        print(f"Mid/Low freq ratio: {ratio:.4f} (threshold: 0.1)")
        print(f"Status: {'✅ PASS' if passed else '❌ FAIL'}")
        
        if not passed:
            self.failed_tests.append("Continental Ghosts")
            print(f"\n⚠️  Possible Earth-like patterns detected")
            print(f"   Recommendation: Re-run with pre-flooding enabled")
            
            # Save high-contrast image for manual inspection
            enhanced_img = (enhanced * 255).astype(np.uint8)
            Image.fromarray(enhanced_img).save("ghost_detection.png")
            print(f"   Saved: ghost_detection.png (manual inspection)")
        else:
            self.passed_tests.append("Continental Ghosts")
        
        return passed
    
    def test_multiscale_texture(self) -> bool:
        """Test for presence of multi-scale variability"""
        # Compute power spectrum
        fft2d = np.fft.fft2(self.data)
        power_spectrum = np.abs(fft2d)
        
        # Sample power at different scales
        scales = [2, 4, 8, 16, 32, 64]
        powers = []
        
        for scale in scales:
            if scale < min(self.height, self.width) // 2:
                idx_y = min(scale, self.height // 2 - 1)
                idx_x = min(scale, self.width // 2 - 1)
                powers.append(power_spectrum[idx_y, idx_x])
            else:
                powers.append(0)
        
        # Check for non-monotonic power (indicates multi-scale content)
        diffs = np.diff(powers)
        has_variation = np.any(diffs > 0) and np.any(diffs < 0)
        
        print(f"\n{'='*60}")
        print(f"TEST 4: Multi-Scale Texture")
        print(f"{'='*60}")
        print(f"Power at different scales:")
        for scale, power in zip(scales, powers):
            print(f"  {scale:3d} px: {power:.3e}")
        
        print(f"Status: {'✅ PASS' if has_variation else '⚠️  WEAK'}")
        
        if has_variation:
            self.passed_tests.append("Multi-Scale Texture")
        else:
            print(f"\n⚠️  Texture appears over-smoothed")
            print(f"   Recommendation: Apply 3-4 octave PNPL")
        
        return has_variation
    
    def generate_report(self) -> None:
        """Print summary report"""
        total_tests = len(self.passed_tests) + len(self.failed_tests)
        
        print(f"\n{'='*60}")
        print(f"VALIDATION REPORT")
        print(f"{'='*60}")
        print(f"Passed: {len(self.passed_tests)}/{total_tests}")
        print(f"Failed: {len(self.failed_tests)}/{total_tests}")
        
        if self.failed_tests:
            print(f"\n❌ Failed Tests:")
            for test in self.failed_tests:
                print(f"   • {test}")
        
        if self.passed_tests:
            print(f"\n✅ Passed Tests:")
            for test in self.passed_tests:
                print(f"   • {test}")
        
        if not self.failed_tests:
            print(f"\n🎉 All validation tests passed!")
            print(f"   Ocean World baseline is production-ready.")
        else:
            print(f"\n⚠️  Some tests failed. Review recommendations above.")


class MultiOctavePNPL:
    """Apply multi-octave PNPL with latitude-dependent amplitude"""
    
    def __init__(self, width: int, height: int, num_octaves: int = 3):
        self.width = width
        self.height = height
        self.num_octaves = num_octaves
    
    def generate_latitude_weight(self) -> np.ndarray:
        """Generate latitude-dependent amplitude mask"""
        y = np.arange(self.height)
        lat = (y / self.height - 0.5) * 180  # -90 to +90
        
        # Storm track emphasis: peak at 40-60° latitude
        storm_track = np.exp(-((np.abs(lat) - 50) / 20) ** 2)
        
        # Base amplitude profile
        # High at storm tracks, moderate at subtropics, low at equator/poles
        amplitude = 0.05 + 0.10 * storm_track
        
        # Broadcast to 2D
        return amplitude[:, np.newaxis] * np.ones((1, self.width))
    
    def generate_octave(self, octave: int, scale: float = 100.0) -> np.ndarray:
        """Generate single Perlin noise octave"""
        freq = 2 ** octave
        noise = np.zeros((self.height, self.width))
        
        for y in range(self.height):
            for x in range(self.width):
                # Perlin noise with frequency scaling
                # Add anisotropy: 1.5x stretch in longitude (Coriolis effect)
                nx = x / scale * freq
                ny = y / scale * freq / 1.5
                
                noise[y, x] = pnoise2(nx, ny, octaves=1, 
                                      persistence=0.5,
                                      lacunarity=2.0,
                                      repeatx=self.width,
                                      repeaty=None,  # Don't repeat latitude
                                      base=octave)
        
        return noise
    
    def apply(self, base_field: np.ndarray, 
              strength: float = 0.15) -> np.ndarray:
        """
        Apply multi-octave PNPL to base field
        
        Args:
            base_field: [height, width] normalized [0,1] ocean field
            strength: Overall noise strength (0-1)
        
        Returns:
            Enhanced field with multi-scale texture
        """
        print(f"\nApplying {self.num_octaves}-octave PNPL...")
        
        lat_weight = self.generate_latitude_weight()
        accumulated_noise = np.zeros_like(base_field)
        
        for octave in range(self.num_octaves):
            print(f"  Generating octave {octave+1}/{self.num_octaves}...")
            
            octave_noise = self.generate_octave(octave, scale=100.0)
            amplitude = 0.5 ** octave  # Halving per octave
            
            # Apply latitude weighting
            weighted_noise = octave_noise * lat_weight * amplitude
            accumulated_noise += weighted_noise
        
        # Normalize accumulated noise
        if accumulated_noise.std() > 0:
            accumulated_noise = (accumulated_noise - accumulated_noise.mean()) / accumulated_noise.std()
            accumulated_noise *= strength
        
        # Blend with base field
        result = base_field + accumulated_noise
        result = np.clip(result, 0, 1)
        
        print(f"  ✅ PNPL applied (amplitude: {strength})")
        
        return result


class OceanWorldExporter:
    """Export ocean world with proper colormaps and legends"""
    
    @staticmethod
    def create_colormap(name: str = "ocean_thermal") -> np.ndarray:
        """
        Create perceptually uniform colormap
        
        Returns:
            [256, 3] RGB colormap array
        """
        if name == "ocean_thermal":
            # Blue (cold) to White (warm) gradient
            t = np.linspace(0, 1, 256)
            
            # Perceptual blue-white with slight gamma
            r = np.power(t, 0.8)
            g = t
            b = np.ones(256)
            
            return np.stack([r, g, b], axis=1)
        
        else:
            raise ValueError(f"Unknown colormap: {name}")
    
    @staticmethod
    def export_16bit(data: np.ndarray, 
                     filename: str,
                     colormap: Optional[np.ndarray] = None) -> None:
        """
        Export as 16-bit PNG
        
        Args:
            data: [height, width] normalized [0,1] array
            filename: Output path
            colormap: [256, 3] RGB colormap (optional)
        """
        if colormap is None:
            colormap = OceanWorldExporter.create_colormap()
        
        # Map to 16-bit
        indices = (data * 65535).astype(np.uint16)
        
        # Apply colormap
        # Downsample to 8-bit for colormap lookup
        indices_8bit = (indices // 256).astype(np.uint8)
        rgb = colormap[indices_8bit]
        
        # Scale back to 16-bit
        rgb_16bit = (rgb * 65535).astype(np.uint16)
        
        # Save using PIL (limited to 8-bit) or use imageio for true 16-bit
        try:
            import imageio
            imageio.imwrite(filename, rgb_16bit)
            print(f"✅ Exported 16-bit PNG: {filename}")
        except ImportError:
            # Fallback to 8-bit with dithering
            print("⚠️  imageio not available, falling back to 8-bit with dithering")
            OceanWorldExporter.export_8bit_dithered(data, filename)
    
    @staticmethod
    def export_8bit_dithered(data: np.ndarray, filename: str) -> None:
        """Export with Floyd-Steinberg dithering"""
        # Convert to 8-bit
        img_8bit = (data * 255).astype(np.float64)
        
        # Apply Floyd-Steinberg dithering
        height, width = img_8bit.shape
        
        for y in range(height - 1):
            for x in range(1, width - 1):
                old_pixel = img_8bit[y, x]
                new_pixel = round(old_pixel)
                img_8bit[y, x] = new_pixel
                
                error = old_pixel - new_pixel
                
                # Distribute error
                img_8bit[y, x + 1]     += error * 7 / 16
                img_8bit[y + 1, x - 1] += error * 3 / 16
                img_8bit[y + 1, x]     += error * 5 / 16
                img_8bit[y + 1, x + 1] += error * 1 / 16
        
        # Clip and convert
        img_8bit = np.clip(img_8bit, 0, 255).astype(np.uint8)
        
        # Apply colormap
        colormap = OceanWorldExporter.create_colormap()
        rgb = colormap[img_8bit]
        rgb_8bit = (rgb * 255).astype(np.uint8)
        
        Image.fromarray(rgb_8bit).save(filename)
        print(f"✅ Exported 8-bit dithered PNG: {filename}")
    
    @staticmethod
    def add_legend(img_path: str, 
                   variable_name: str,
                   value_range: Tuple[float, float],
                   units: str,
                   output_path: str) -> None:
        """Add scientific legend to image"""
        img = Image.open(img_path)
        img_array = np.array(img)
        
        fig, ax = plt.subplots(figsize=(14, 7), dpi=200)
        ax.imshow(img_array, aspect='equal')
        ax.axis('off')
        
        # Colorbar
        gradient = np.linspace(0, 1, 256).reshape(1, -1)
        cax = fig.add_axes([0.15, 0.08, 0.7, 0.03])
        
        # Apply ocean thermal colormap
        colormap = OceanWorldExporter.create_colormap()
        gradient_rgb = colormap[np.linspace(0, 255, 256).astype(int)]
        cax.imshow([gradient_rgb], aspect='auto')
        
        # Colorbar ticks
        tick_positions = [0, 64, 128, 192, 255]
        tick_values = np.linspace(value_range[0], value_range[1], 5)
        cax.set_xticks(tick_positions)
        cax.set_xticklabels([f"{v:.1f}" for v in tick_values])
        cax.set_yticks([])
        cax.set_xlabel(f"{variable_name} ({units})", fontsize=10, weight='bold')
        
        # Metadata box
        metadata = f"""Ocean World Baseline
Variable: {variable_name}
Range: {value_range[0]:.2f} to {value_range[1]:.2f} {units}
Projection: Equirectangular
Processing: Interpolated, Normalized, PNPL-Enhanced
Generator: UniversalInterpolator v2.0"""
        
        ax.text(0.02, 0.98, metadata, transform=ax.transAxes,
                fontsize=9, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Added legend: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Ocean World Validation & Post-Processing Toolkit"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', 
                                           help='Run validation tests')
    validate_parser.add_argument('input', type=str, 
                                help='Input file (.npy or .png)')
    
    # Process command (PNPL)
    process_parser = subparsers.add_parser('process',
                                          help='Apply multi-octave PNPL')
    process_parser.add_argument('input', type=str)
    process_parser.add_argument('--output', type=str, default='ocean_pnpl.npy')
    process_parser.add_argument('--octaves', type=int, default=3)
    process_parser.add_argument('--strength', type=float, default=0.15)
    
    # Export command
    export_parser = subparsers.add_parser('export',
                                         help='Export with colormap')
    export_parser.add_argument('input', type=str)
    export_parser.add_argument('--output', type=str, default='ocean_final.png')
    export_parser.add_argument('--format', type=str, 
                              choices=['png8', 'png16'],
                              default='png16')
    export_parser.add_argument('--variable', type=str, default='SST')
    export_parser.add_argument('--range', type=float, nargs=2, 
                              default=[-2.0, 35.0])
    export_parser.add_argument('--units', type=str, default='°C')
    export_parser.add_argument('--add-legend', action='store_true')
    
    args = parser.parse_args()
    
    if args.command == 'validate':
        # Load data
        if args.input.endswith('.npy'):
            data = np.load(args.input)
        else:
            data = np.array(Image.open(args.input))
        
        # Run validation
        validator = OceanWorldValidator(data)
        validator.test_seam_continuity()
        validator.test_color_richness()
        validator.test_continental_ghosts()
        validator.test_multiscale_texture()
        validator.generate_report()
    
    elif args.command == 'process':
        # Load data
        data = np.load(args.input)
        
        # Apply PNPL
        pnpl = MultiOctavePNPL(data.shape[1], data.shape[0], args.octaves)
        enhanced = pnpl.apply(data, strength=args.strength)
        
        # Save
        np.save(args.output, enhanced)
        print(f"\n✅ Saved enhanced data: {args.output}")
    
    elif args.command == 'export':
        # Load data
        if args.input.endswith('.npy'):
            data = np.load(args.input)
        else:
            data = np.array(Image.open(args.input))
            if data.ndim == 3:
                data = data.mean(axis=2) / 255.0
        
        # Export
        exporter = OceanWorldExporter()
        
        if args.format == 'png16':
            exporter.export_16bit(data, args.output)
        else:
            exporter.export_8bit_dithered(data, args.output)
        
        # Add legend if requested
        if args.add_legend:
            legend_path = args.output.replace('.png', '_labeled.png')
            exporter.add_legend(
                args.output,
                variable_name=args.variable,
                value_range=tuple(args.range),
                units=args.units,
                output_path=legend_path
            )
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
