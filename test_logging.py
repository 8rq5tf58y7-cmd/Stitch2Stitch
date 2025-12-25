#!/usr/bin/env python3
"""
Test script to verify logging is working
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

print("Testing logging setup...")
print("=" * 60)

try:
    from src.utils.logger import setup_logger, get_log_file_path
    
    logger = setup_logger("test_logging")
    
    print("\n1. Logger created successfully")
    
    # Test different log levels
    logger.debug("This is a DEBUG message")
    logger.info("This is an INFO message")
    logger.warning("This is a WARNING message")
    logger.error("This is an ERROR message")
    
    print("\n2. Log messages sent")
    
    # Check log file
    log_path = get_log_file_path()
    if log_path:
        print(f"\n3. Log file location: {log_path.absolute()}")
        if log_path.exists():
            print(f"   ✓ Log file exists")
            print(f"   Size: {log_path.stat().st_size} bytes")
            print(f"\n   First few lines of log file:")
            print("   " + "-" * 56)
            try:
                with open(log_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    for i, line in enumerate(lines[-10:], 1):  # Last 10 lines
                        print(f"   {line.rstrip()}")
            except Exception as e:
                print(f"   Error reading log file: {e}")
        else:
            print(f"   ✗ Log file does not exist")
    else:
        print("\n3. ✗ No log file path available")
    
    print("\n" + "=" * 60)
    print("Logging test complete!")
    
except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

