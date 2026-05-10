"""
IDM-VTON Optimization Settings Module

Provides tunable parameters for local inference optimization.
Environment-configurable for flexibility across different GPU hardware.

Usage in fits_app.py:
    from idm_vton_settings import get_inference_steps, get_optimization_mode
    steps = get_inference_steps("hq")  # or "preview"
"""

import os
from enum import Enum


class InferenceMode(Enum):
    """Inference quality modes."""
    PREVIEW = "preview"      # Fast iteration, 8 steps, ~30 seconds
    BALANCED = "balanced"    # Medium quality, 12 steps, ~45 seconds
    HQ = "hq"                # High quality, 20 steps, ~90 seconds
    ULTRA = "ultra"          # Maximum quality, 30 steps, ~135 seconds (original)


class OptimizationSettings:
    """Centralized optimization settings with environment override support."""
    
    # Inference steps per mode
    STEPS_PREVIEW = int(os.getenv("IDMVTON_STEPS_PREVIEW", "8"))
    STEPS_BALANCED = int(os.getenv("IDMVTON_STEPS_BALANCED", "12"))
    STEPS_HQ = int(os.getenv("IDMVTON_STEPS_HQ", "20"))
    STEPS_ULTRA = int(os.getenv("IDMVTON_STEPS_ULTRA", "30"))
    
    # Memory optimizations
    ENABLE_ATTENTION_SLICING = os.getenv("IDMVTON_ENABLE_ATTENTION_SLICING", "1").strip().lower() in {"1", "true", "yes"}
    ENABLE_VAE_TILING = os.getenv("IDMVTON_ENABLE_VAE_TILING", "1").strip().lower() in {"1", "true", "yes"}
    USE_XFORMERS = os.getenv("IDMVTON_USE_XFORMERS", "1").strip().lower() in {"1", "true", "yes"}
    
    # Output resolution constraints (for different GPU VRAM)
    MAX_WIDTH = int(os.getenv("IDMVTON_MAX_WIDTH", "768"))
    MAX_HEIGHT = int(os.getenv("IDMVTON_MAX_HEIGHT", "1024"))
    MIN_WIDTH = int(os.getenv("IDMVTON_MIN_WIDTH", "384"))
    MIN_HEIGHT = int(os.getenv("IDMVTON_MIN_HEIGHT", "512"))
    
    # Guidance scale (from paper: 2.0 is recommended for realism)
    GUIDANCE_SCALE = float(os.getenv("IDMVTON_GUIDANCE_SCALE", "2.0"))
    
    # Seed for reproducibility
    DEFAULT_SEED = int(os.getenv("IDMVTON_DEFAULT_SEED", "42"))
    
    # Enable/disable features
    ENABLE_PREPROCESSING_CACHE = os.getenv("IDMVTON_ENABLE_PREPROCESSING_CACHE", "1").strip().lower() in {"1", "true", "yes"}
    
    @classmethod
    def get_steps_for_mode(cls, mode: str) -> int:
        """Get inference steps for a given mode."""
        mode_lower = mode.strip().lower()
        
        if mode_lower in {"preview", "p"}:
            return cls.STEPS_PREVIEW
        elif mode_lower in {"balanced", "b", "medium", "m"}:
            return cls.STEPS_BALANCED
        elif mode_lower in {"hq", "h", "high", "quality"}:
            return cls.STEPS_HQ
        elif mode_lower in {"ultra", "u", "maximum", "max", "best"}:
            return cls.STEPS_ULTRA
        else:
            # Default to HQ if unknown mode
            return cls.STEPS_HQ
    
    @classmethod
    def get_estimated_time(cls, mode: str, resolution: str = "standard") -> str:
        """Get estimated inference time for a given mode and resolution.
        
        Estimates are for RTX 2000 Ada (20-40 TFLOPS).
        Actual times vary with GPU and system load.
        """
        steps = cls.get_steps_for_mode(mode)
        
        # Base times per mode (including preprocessing ~30s)
        base_times = {
            8: 60,      # ~30s preprocessing + ~30s inference
            12: 75,     # ~30s preprocessing + ~45s inference
            20: 105,    # ~30s preprocessing + ~75s inference
            30: 165,    # ~30s preprocessing + ~135s inference
        }
        
        base_time = base_times.get(steps, steps * 5 + 30)
        
        # Resolution multiplier
        res_multipliers = {
            "preview": 0.8,      # 512x768 or smaller
            "standard": 1.0,     # 768x1024 (default)
            "high": 1.3,         # 1024x1536
            "ultra": 1.7,        # 1280x1920
        }
        
        multiplier = res_multipliers.get(resolution.lower(), 1.0)
        estimated_seconds = int(base_time * multiplier)
        
        minutes = estimated_seconds // 60
        seconds = estimated_seconds % 60
        
        if minutes > 0:
            return f"~{minutes}m {seconds}s"
        else:
            return f"~{seconds}s"
    
    @classmethod
    def get_all_settings(cls) -> dict:
        """Get all current optimization settings as a dictionary."""
        return {
            "steps_preview": cls.STEPS_PREVIEW,
            "steps_balanced": cls.STEPS_BALANCED,
            "steps_hq": cls.STEPS_HQ,
            "steps_ultra": cls.STEPS_ULTRA,
            "enable_attention_slicing": cls.ENABLE_ATTENTION_SLICING,
            "enable_vae_tiling": cls.ENABLE_VAE_TILING,
            "use_xformers": cls.USE_XFORMERS,
            "max_width": cls.MAX_WIDTH,
            "max_height": cls.MAX_HEIGHT,
            "guidance_scale": cls.GUIDANCE_SCALE,
            "preprocessing_cache": cls.ENABLE_PREPROCESSING_CACHE,
        }


def get_inference_steps(mode: str) -> int:
    """Convenience function: Get inference steps for a mode."""
    return OptimizationSettings.get_steps_for_mode(mode)


def get_estimated_time(mode: str, resolution: str = "standard") -> str:
    """Convenience function: Get estimated inference time."""
    return OptimizationSettings.get_estimated_time(mode, resolution)


if __name__ == "__main__":
    # Display all settings when run as main
    print("IDM-VTON Optimization Settings")
    print("=" * 50)
    for key, value in OptimizationSettings.get_all_settings().items():
        print(f"  {key}: {value}")
    
    print("\nEstimated inference times (RTX 2000 Ada):")
    for mode in ["preview", "balanced", "hq", "ultra"]:
        print(f"  {mode.upper():8} → {get_estimated_time(mode)}")
