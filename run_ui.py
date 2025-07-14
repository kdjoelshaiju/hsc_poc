#!/usr/bin/env python3
"""
HSC Semantic Search UI Launcher

Run this script to start the Streamlit web interface.
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Launch the Streamlit UI."""
    ui_path = Path("ui/app.py")
    
    if not ui_path.exists():
        print("❌ UI not found. Please ensure ui/app.py exists.")
        sys.exit(1)
    
    print("🚀 Starting HSC Semantic Search UI...")
    print("📱 Opening web browser...")
    print("🔗 URL: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop")
    
    try:
        # Run streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(ui_path), "--server.port", "8501", "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n👋 UI stopped by user")
    except Exception as e:
        print(f"❌ Error starting UI: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 