#!/usr/bin/env python3
"""
HSC Demo Runner

This script provides a unified interface for running all HSC system demos
and tests in an organized manner.
"""

import sys
import argparse
from pathlib import Path

def main():
    """Main demo runner."""
    parser = argparse.ArgumentParser(description="HSC System Demo Runner")
    parser.add_argument("demo", choices=[
        "basic", "focused", "large-codebase", "error-recovery", 
        "hierarchical", "real-programs", "all"
    ], help="Demo to run")
    
    parser.add_argument("--output-dir", default="demo_outputs", 
                       help="Output directory for results")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Verbose output")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("🚀 HSC System Demo Runner")
    print("="*60)
    
    if args.demo == "all":
        run_all_demos(output_dir, args.verbose)
    else:
        run_single_demo(args.demo, output_dir, args.verbose)

def run_single_demo(demo_name: str, output_dir: Path, verbose: bool):
    """Run a single demo."""
    demo_map = {
        "basic": "demos.basic_demo",
        "focused": "demos.focused_demo", 
        "large-codebase": "demos.large_codebase_demo",
        "error-recovery": "tests.test_error_recovery",
        "hierarchical": "tests.test_hierarchical",
        "real-programs": "tests.test_real_programs"
    }
    
    if demo_name not in demo_map:
        print(f"❌ Unknown demo: {demo_name}")
        return
    
    module_name = demo_map[demo_name]
    print(f"🎬 Running {demo_name} demo...")
    
    try:
        # Import and run the demo
        module = __import__(module_name, fromlist=['main'])
        if hasattr(module, 'main'):
            module.main()
        else:
            print(f"⚠️ Demo {demo_name} has no main() function")
            
    except ImportError as e:
        print(f"❌ Could not import {module_name}: {e}")
    except Exception as e:
        print(f"❌ Demo {demo_name} failed: {e}")
        if verbose:
            import traceback
            traceback.print_exc()

def run_all_demos(output_dir: Path, verbose: bool):
    """Run all demos in sequence."""
    demos = [
        ("basic", "Basic HSC System Demo"),
        ("focused", "Memory-Focused Demo"),
        ("error-recovery", "Error Recovery System"),
        ("real-programs", "Real Program Testing"),
        ("hierarchical", "Hierarchical Embedding"),
        ("large-codebase", "Large Codebase Demo")
    ]
    
    results = []
    
    for demo_name, description in demos:
        print(f"\n🎬 Running {description}...")
        print("-" * 40)
        
        try:
            run_single_demo(demo_name, output_dir, verbose)
            results.append((demo_name, "✅ SUCCESS"))
        except Exception as e:
            print(f"❌ {demo_name} failed: {e}")
            results.append((demo_name, "❌ FAILED"))
    
    # Print summary
    print("\n" + "="*60)
    print("📊 Demo Summary")
    print("="*60)
    
    for demo_name, status in results:
        print(f"   {demo_name:20} {status}")
    
    success_count = sum(1 for _, status in results if "SUCCESS" in status)
    total_count = len(results)
    
    print(f"\n🎉 Completed {success_count}/{total_count} demos successfully!")

if __name__ == "__main__":
    main() 