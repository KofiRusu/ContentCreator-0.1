#!/usr/bin/env python3
"""
Quick test script for video generation functionality
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pipeline.video_gen import VideoGenerator, generate_scene_video
from pipeline.scene_parser import Scene

def test_video_generation():
    """Test video generation in dry run mode"""
    
    print("🎬 Testing Video Generation with fal.ai Veo 3")
    print("=" * 50)
    print("Mode: DRY RUN (no actual costs)")
    print()
    
    # Test scene
    scene = Scene(
        id=1,
        title="Epic Forest Battle",
        characters=["Hero", "Villain"],
        setting="Mystical forest with glowing trees and ancient ruins",
        summary="The hero discovers a hidden portal while being pursued by dark forces",
        tone="epic, mysterious, cinematic"
    )
    
    # Test 1: Single video generation
    print("1. Testing single video generation...")
    video_path = generate_scene_video(scene, dry_run=True)
    
    if video_path:
        print(f"✅ Video generated successfully!")
        print(f"📁 Saved to: {video_path}")
        
        # Verify file exists
        if Path(video_path).exists():
            print("✅ File exists on disk")
        else:
            print("❌ File not found")
    else:
        print("❌ Video generation failed")
        return False
    
    print()
    
    # Test 2: VideoGenerator with session tracking
    print("2. Testing VideoGenerator class...")
    generator = VideoGenerator(dry_run=True, budget_limit=20.0)
    
    # Generate videos for multiple scenes
    success_count = 0
    for i in range(1, 4):
        test_scene = Scene(
            id=i,
            title=f"Test Scene {i}",
            characters=[f"Character{i}"],
            setting=f"Setting for scene {i}",
            summary=f"Action happening in scene {i}",
            tone="dramatic"
        )
        
        path = generator.generate_scene_video(test_scene)
        if path:
            print(f"✅ Scene {i} video generated")
            success_count += 1
        else:
            print(f"❌ Scene {i} failed")
    
    # Get session summary
    summary = generator.get_session_summary()
    
    print()
    print("📊 Final Session Summary:")
    print(f"  • Videos generated: {summary['videos_generated']}")
    print(f"  • Mock total cost: ${summary['total_cost']:.2f}")
    print(f"  • Budget remaining: ${summary['budget_remaining']:.2f}")
    print(f"  • Dry run mode: {summary['dry_run']}")
    print()
    
    if success_count >= 3:
        print("🎉 All tests passed! Video generation module working correctly!")
        return True
    else:
        print("❌ Some tests failed")
        return False

if __name__ == "__main__":
    success = test_video_generation()
    sys.exit(0 if success else 1) 