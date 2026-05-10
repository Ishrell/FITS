"""
Quick validation script to measure inference speedup after optimizations.

Usage:
    python measure_speedup.py --person path/to/person.jpg --cloth path/to/cloth.jpg
"""

import argparse
import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime

def run_inference(person_path, cloth_path, steps, output_file):
    """Run a single inference and return timing info."""
    start_time = datetime.now()
    
    try:
        result = subprocess.run([
            sys.executable,
            "fits_single_infer.py",
            "--person", str(person_path),
            "--cloth", str(cloth_path),
            "--output", str(output_file),
            "--steps", str(steps),
        ], capture_output=True, text=True, timeout=600)
        
        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()
        
        if result.returncode == 0 and os.path.isfile(output_file):
            return {
                "success": True,
                "elapsed_seconds": elapsed,
                "steps": steps,
                "output_file": str(output_file),
            }
        else:
            return {
                "success": False,
                "error": result.stderr or result.stdout,
                "steps": steps,
            }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": "Timeout after 600 seconds",
            "steps": steps,
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "steps": steps,
        }


def main():
    parser = argparse.ArgumentParser(description="Measure IDM-VTON inference speedup")
    parser.add_argument("--person", required=True, help="Path to person image")
    parser.add_argument("--cloth", required=True, help="Path to cloth image")
    parser.add_argument("--output-dir", default="./speedup_results", help="Output directory for results")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Test different step counts
    step_configs = [
        ("preview", 8),
        ("balanced", 12),
        ("hq", 20),
        ("ultra", 30),
    ]
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "person_image": args.person,
        "cloth_image": args.cloth,
        "measurements": [],
    }
    
    print("IDM-VTON Inference Speedup Measurement")
    print("=" * 60)
    print(f"Person: {args.person}")
    print(f"Cloth: {args.cloth}\n")
    
    for mode_name, steps in step_configs:
        output_file = output_dir / f"result_{mode_name}.jpg"
        print(f"Running {mode_name.upper()} mode ({steps} steps)...", end=" ", flush=True)
        
        measurement = run_inference(args.person, args.cloth, steps, output_file)
        results["measurements"].append({
            "mode": mode_name,
            **measurement,
        })
        
        if measurement["success"]:
            print(f"✓ {measurement['elapsed_seconds']:.1f}s")
        else:
            print(f"✗ {measurement['error']}")
    
    # Save results
    results_file = output_dir / "measurements.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print("=" * 60)
    
    successful = [m for m in results["measurements"] if m["success"]]
    if successful:
        # Find fastest and slowest
        fastest = min(successful, key=lambda x: x["elapsed_seconds"])
        slowest = max(successful, key=lambda x: x["elapsed_seconds"])
        
        print(f"\nFastest: {fastest['mode'].upper()} ({fastest['steps']} steps) - {fastest['elapsed_seconds']:.1f}s")
        print(f"Slowest: {slowest['mode'].upper()} ({slowest['steps']} steps) - {slowest['elapsed_seconds']:.1f}s")
        print(f"Speedup: {slowest['elapsed_seconds'] / fastest['elapsed_seconds']:.1f}x faster")
        
        print("\nMode | Steps | Time (s) | Est. Quality")
        print("-----|-------|----------|-------------")
        for m in successful:
            quality_map = {8: "Preview", 12: "Good", 20: "Very Good", 30: "Excellent"}
            quality = quality_map.get(m["steps"], "Unknown")
            print(f"{m['mode'].upper():4} | {m['steps']:5} | {m['elapsed_seconds']:8.1f} | {quality}")
    
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
