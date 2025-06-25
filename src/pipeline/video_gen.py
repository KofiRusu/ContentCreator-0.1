"""
Video Generation Module

Generates video clips per scene using the fal.ai Veo 3 API.
Input is scene metadata + optional still image reference.
Output is a downloadable video clip saved as scene_{id}.mp4.
"""

import hashlib
import logging
import os
import time

from pathlib import Path
from typing import Dict, Optional

import requests
from dotenv import load_dotenv

try:
    import fal_client
except ImportError:
    fal_client = None

from src.pipeline.scene_parser import Scene

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration constants
DEFAULT_DURATION = 10  # seconds
DEFAULT_RESOLUTION = "1080p"
MAX_RETRIES = 2
RETRY_DELAY = 5  # seconds
REQUEST_TIMEOUT = 300  # 5 minutes max wait per request
CACHE_TTL = 86400  # 24 hours cache validity

# Cost tracking (approximate USD)
ESTIMATED_COST_PER_SECOND = 0.50
MAX_BUDGET_PER_SESSION = 50.00


class VideoGenerationError(Exception):
    """Custom exception for video generation errors."""
    pass


class BudgetExceededException(VideoGenerationError):
    """Raised when budget limits are exceeded."""
    pass


class VideoGenerator:
    """Main video generation class with cost tracking and caching."""

    def __init__(self,
                 dry_run: bool = False,
                 simulate: bool = False,
                 budget_limit: float = MAX_BUDGET_PER_SESSION):
        """
        Initialize video generator.

        Args:
            dry_run: If True, skip actual API calls and return mock results
            simulate: If True, simulate API calls with delays but no costs
            budget_limit: Maximum budget for this session (USD)
        """
        self.dry_run = dry_run
        self.simulate = simulate
        self.budget_limit = budget_limit
        self.session_cost = 0.0
        self.generated_videos = []
        self.cache = {}

        # Ensure assets directory exists
        self.assets_dir = Path(__file__).parent.parent / "assets"
        self.assets_dir.mkdir(exist_ok=True)

        # Initialize fal client
        self._init_fal_client()

    def _init_fal_client(self) -> None:
        """Initialize fal client with API key."""
        if self.dry_run or self.simulate:
            logger.info(
                "Running in test mode - skipping fal client initialization")
            return

        if not fal_client:
            raise VideoGenerationError(
                "fal_client not installed. Run: pip install fal"
            )

        api_key = os.getenv("FAL_KEY")
        if not api_key:
            raise VideoGenerationError(
                "FAL_KEY not found in environment variables"
            )

        logger.info("Fal client initialized successfully")

    def _generate_prompt_hash(self, prompt: str) -> str:
        """Generate hash for prompt caching."""
        return hashlib.md5(prompt.encode()).hexdigest()[:12]

    def _build_video_prompt(self, scene: Scene,
                            image_path: Optional[str] = None) -> str:
        """
        Build comprehensive video generation prompt from scene data.

        Args:
            scene: Scene object with metadata
            image_path: Optional path to reference image

        Returns:
            str: Formatted prompt for video generation
        """
        # Base prompt construction
        prompt_parts = []

        # Title and setting
        prompt_parts.append(f"Scene: {scene.title}")
        prompt_parts.append(f"Setting: {scene.setting}")

        # Characters (if any)
        if scene.characters:
            characters_str = ", ".join(scene.characters)
            prompt_parts.append(f"Characters: {characters_str}")

        # Summary and tone
        prompt_parts.append(f"Action: {scene.summary}")
        prompt_parts.append(f"Tone: {scene.tone}")

        # Technical specifications
        prompt_parts.extend([
            "Style: Cinematic, high quality, professional lighting",
            "Camera: Dynamic cinematography with smooth movements",
            "Duration: 8-12 seconds",
            "Quality: 1080p resolution, 24fps"
        ])

        prompt = ". ".join(prompt_parts)

        logger.info(
            f"Generated prompt for scene {scene.id}: {prompt[:100]}...")
        return prompt

    def _check_budget(self, estimated_cost: float) -> None:
        """
        Check if operation would exceed budget.

        Args:
            estimated_cost: Estimated cost of operation

        Raises:
            BudgetExceededException: If budget would be exceeded
        """
        total_cost = self.session_cost + estimated_cost
        if total_cost > self.budget_limit:
            raise BudgetExceededException(
                f"Operation would exceed budget: ${
                    total_cost:.2f} > ${
                    self.budget_limit:.2f}"
            )

    def _download_video(self, video_url: str, output_path: Path) -> bool:
        """
        Download video from URL to local path.

        Args:
            video_url: URL of the video to download
            output_path: Local path to save video

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info(f"Downloading video from {video_url}")

            response = requests.get(video_url, stream=True, timeout=60)
            response.raise_for_status()

            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            file_size = output_path.stat().st_size
            logger.info(f"Video downloaded successfully: {file_size:,} bytes")
            return True

        except Exception as e:
            logger.error(f"Failed to download video: {e}")
            return False

    def _call_fal_api(self, prompt: str,
                      image_path: Optional[str] = None) -> Optional[Dict]:
        """
        Call fal.ai Veo 3 API with retry logic.

        Args:
            prompt: Video generation prompt
            image_path: Optional reference image path

        Returns:
            Dict or None: API response data or None if failed
        """
        if self.dry_run:
            logger.info("DRY RUN: Skipping actual API call")
            return {
                "video": {"url": "https://example.com/mock_video.mp4"},
                "duration": DEFAULT_DURATION
            }

        if self.simulate:
            logger.info("SIMULATE: Mocking API call with delay")
            time.sleep(10)  # Simulate API delay
            return {
                "video": {"url": "https://example.com/simulated_video.mp4"},
                "duration": DEFAULT_DURATION
            }

        # Prepare request payload
        payload = {
            "prompt": prompt,
            "duration": DEFAULT_DURATION,
            "resolution": DEFAULT_RESOLUTION,
            "aspect_ratio": "16:9",
            "fps": 24
        }

        # Add image reference if provided
        if image_path and Path(image_path).exists():
            logger.info(f"Including reference image: {image_path}")
            # Note: Actual implementation would upload image to fal
            # payload["image_url"] = uploaded_image_url

        # Retry loop
        for attempt in range(MAX_RETRIES + 1):
            try:
                logger.info(
                    f"Calling fal.ai API (attempt {attempt + 1}/{MAX_RETRIES + 1})")

                # Call fal.ai Veo 3 API
                result = fal_client.subscribe(
                    "fal-ai/veo-3",
                    arguments=payload,
                    with_logs=True,
                    timeout=REQUEST_TIMEOUT
                )

                if result and "video" in result:
                    logger.info("Successfully received video from fal.ai")
                    return result
                else:
                    logger.error("Invalid response format from fal.ai")

            except Exception as e:
                logger.error(
                    f"fal.ai API call failed (attempt {
                        attempt + 1}): {e}")

                if attempt < MAX_RETRIES:
                    logger.info(f"Retrying in {RETRY_DELAY} seconds...")
                    time.sleep(RETRY_DELAY)
                    continue
                else:
                    logger.error("All retry attempts failed")
                    break

        return None

    def generate_scene_video(self,
                             scene: Scene,
                             image_path: Optional[str] = None) -> Optional[str]:
        """
        Generate video clip for a single scene.

        Args:
            scene: Scene object with metadata
            image_path: Optional path to reference image

        Returns:
            str or None: Path to generated video file, or None if failed
        """
        try:
            # Build prompt
            prompt = self._build_video_prompt(scene, image_path)
            prompt_hash = self._generate_prompt_hash(prompt)

            # Check cache first
            if prompt_hash in self.cache:
                cached_path = self.cache[prompt_hash]
                if Path(cached_path).exists():
                    logger.info(f"Using cached video for scene {scene.id}")
                    return cached_path

            # Check budget
            estimated_cost = DEFAULT_DURATION * ESTIMATED_COST_PER_SECOND
            self._check_budget(estimated_cost)

            # Generate output path
            output_filename = f"scene_{scene.id}.mp4"
            output_path = self.assets_dir / output_filename

            # Call API
            start_time = time.time()
            result = self._call_fal_api(prompt, image_path)

            if not result:
                logger.error(f"Failed to generate video for scene {scene.id}")
                return None

            # Download video
            video_url = result.get("video", {}).get("url")
            if not video_url:
                logger.error("No video URL in API response")
                return None

            if not self.dry_run and not self.simulate:
                if not self._download_video(video_url, output_path):
                    return None
            else:
                # Create mock file for testing
                output_path.write_text(f"Mock video for scene {scene.id}")

            # Update tracking
            generation_time = time.time() - start_time
            actual_cost = estimated_cost  # Could be adjusted based on actual duration

            self.session_cost += actual_cost
            self.generated_videos.append({
                "scene_id": scene.id,
                "path": str(output_path),
                "cost": actual_cost,
                "generation_time": generation_time,
                "prompt_hash": prompt_hash
            })

            # Cache result
            self.cache[prompt_hash] = str(output_path)

            logger.info(
                f"Video generated for scene {scene.id}: {output_path} "
                f"(${actual_cost:.2f}, {generation_time:.1f}s)"
            )

            return str(output_path)

        except BudgetExceededException as e:
            logger.error(f"Budget exceeded for scene {scene.id}: {e}")
            return None

        except Exception as e:
            logger.error(
                f"Unexpected error generating video for scene {
                    scene.id}: {e}")
            return None

    def get_session_summary(self) -> Dict:
        """Get summary of current generation session."""
        return {
            "videos_generated": len(self.generated_videos),
            "total_cost": self.session_cost,
            "budget_remaining": self.budget_limit - self.session_cost,
            "videos": self.generated_videos,
            "dry_run": self.dry_run,
            "simulate": self.simulate
        }


# Convenience function for single video generation
def generate_scene_video(scene: Scene,
                         image_path: Optional[str] = None,
                         dry_run: bool = False,
                         simulate: bool = False) -> Optional[str]:
    """
    Generate a single scene video.

    Args:
        scene: Scene object with metadata
        image_path: Optional path to reference image
        dry_run: If True, skip actual API calls
        simulate: If True, simulate API calls with delays

    Returns:
        str or None: Path to generated video file, or None if failed
    """
    generator = VideoGenerator(dry_run=dry_run, simulate=simulate)
    return generator.generate_scene_video(scene, image_path)


# Example usage and testing
if __name__ == "__main__":
    from scene_parser import Scene

    # Test scene
    test_scene = Scene(
        id=1,
        title="Forest Encounter",
        characters=["Lena", "Mysterious Stranger"],
        setting="Dark forest with ancient oak trees",
        summary="Lena walks through the forest and meets a cloaked stranger who speaks in riddles",
        tone="mysterious, suspenseful"
    )

    # Test in dry run mode
    print("Testing video generation in dry run mode...")
    video_path = generate_scene_video(test_scene, dry_run=True)
    print(f"Generated video path: {video_path}")

    # Test with full generator
    generator = VideoGenerator(dry_run=True)
    video_path = generator.generate_scene_video(test_scene)

    summary = generator.get_session_summary()
    print(f"Session summary: {summary}")
