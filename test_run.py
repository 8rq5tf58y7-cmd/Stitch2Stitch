#!/usr/bin/env python3
"""
Test script to run the application and capture any errors
"""
import sys
import traceback
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    print("Starting application...")
    print("=" * 50)
    
    from src.main import main
    
    print("Main function imported successfully")
    print("Calling main()...")
    print("=" * 50)
    
    exit_code = main()
    
    print("=" * 50)
    print(f"Application exited with code: {exit_code}")
    
except Exception as e:
    print("=" * 50)
    print("ERROR: Application failed to start")
    print("=" * 50)
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {e}")
    print("\nFull traceback:")
    traceback.print_exc()
    sys.exit(1)

