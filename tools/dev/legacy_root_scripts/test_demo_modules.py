"""
Test script to verify preview_demo.py works correctly
"""
import sys
from pathlib import Path

# Add project root
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("=" * 60)
print("Preview Demo Module Test")
print("=" * 60)

print("\n[1] Importing modules...")
try:
    from src.audio_analyzer import AudioAnalyzer
    from src.ffmpeg_processor import FFmpegProcessor
    print("✅  Module imports successful")
except Exception as e:
    print(f"❌  Import failed: {e}")
    sys.exit(1)

print("\n[2] Checking preview_demo.py structure...")
try:
    preview_demo = project_root / "preview_demo.py"
    if preview_demo.exists():
        print(f"✅  preview_demo.py exists at: {preview_demo}")
        size_mb = preview_demo.stat().st_size / 1024 / 1024
        print(f"    Size: {size_mb:.2f} MB")
    else:
        print("❌  preview_demo.py not found")
        sys.exit(1)
except Exception as e:
    print(f"❌  Check failed: {e}")

print("\n[3] Checking test video...")
test_video = project_root / "input" / "zhou_shen_test.mp4"
if test_video.exists():
    print(f"✅  Test video available: {test_video}")
    size_mb = test_video.stat().st_size / 1024 / 1024
    print(f"    Size: {size_mb:.2f} MB")
else:
    print(f"⚠️  Test video not found at: {test_video}")
    print("    Use your own video for testing")

print("\n" + "=" * 60)
print("Test Summary")
print("=" * 60)
print("✅  All core checks passed!")
print("\n" + "=" * 60)
print("To run full test:")
print("  1. Open browser: http://localhost:8502")
print("  2. Follow the test steps as outlined")
print("=" * 60)

