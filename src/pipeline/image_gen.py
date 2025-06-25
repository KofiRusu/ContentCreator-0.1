"""
Image Generation Module

Generates scene images using OpenAI DALL·E 3 based on structured scene metadata.
Each image is saved to src/assets/ and the file path is returned.
"""

import logging
import os
import requests
from pathlib import Path
from typing import Optional

import openai
from dotenv import load_dotenv

from .scene_parser import Scene

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_openai_client():
    """Get OpenAI client instance for image generation."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY not found in environment variables")
        return None
    return openai.OpenAI(api_key=api_key)


def construct_image_prompt(scene: Scene) -> str:
    """
    Construct a detailed image prompt from scene metadata.
    
    Args:
        scene (Scene): Scene object with metadata
        
    Returns:
        str: Detailed prompt for DALL·E 3
    """
    
    # Base prompt with setting and title context
    prompt_parts = [
        f"A cinematic scene titled '{scene.title}'",
        f"Setting: {scene.setting}",
    ]
    
    # Add characters if present
    if scene.characters:
        characters_str = ", ".join(scene.characters)
        prompt_parts.append(f"Characters: {characters_str}")
    
    # Add tone-specific styling
    tone_styles = {
        "suspenseful": "dramatic lighting with deep shadows and tension",
        "mysterious": "atmospheric with fog and hidden details",
        "romantic": "warm golden lighting and soft focus",
        "comedic": "bright, colorful, and whimsical style",
        "dramatic": "intense lighting and strong contrasts",
        "eerie": "dark, unsettling atmosphere with cold colors",
        "peaceful": "soft, calming colors and gentle lighting",
        "chaotic": "dynamic composition with bold colors",
        "melancholic": "muted colors and soft, diffused lighting",
        "triumphant": "bright, heroic lighting with warm tones",
        "tense": "sharp contrasts and angular compositions",
        "whimsical": "playful colors and fantastical elements",
        "dark": "low-key lighting with stark shadows",
        "hopeful": "warm, uplifting lighting and bright colors",
        "nostalgic": "vintage tones and soft, dreamy quality"
    }
    
    style = tone_styles.get(scene.tone.lower(), "cinematic and atmospheric")
    prompt_parts.append(f"Style: {style}")
    
    # Add technical specifications for quality
    prompt_parts.extend([
        "High quality, detailed, professional photography",
        "16:9 aspect ratio, cinematic composition"
    ])
    
    # Join all parts with proper formatting
    full_prompt = ". ".join(prompt_parts) + "."
    
    logger.info(f"Generated prompt for scene {scene.id}: {full_prompt[:100]}...")
    return full_prompt


def download_image(image_url: str, file_path: Path) -> bool:
    """
    Download image from URL and save to file path.
    
    Args:
        image_url (str): URL of the generated image
        file_path (Path): Local path to save the image
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()
        
        # Ensure directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save image
        with open(file_path, 'wb') as f:
            f.write(response.content)
        
        logger.info(f"Image saved to {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download image: {e}")
        return False


def generate_scene_image(scene: Scene, max_retries: int = 1) -> Optional[str]:
    """
    Generate scene image using DALL·E 3 based on scene metadata.
    
    Args:
        scene (Scene): Scene object containing metadata
        max_retries (int): Maximum number of retry attempts
        
    Returns:
        Optional[str]: File path of generated image, or None if failed
    """
    
    client = get_openai_client()
    if not client:
        logger.error("OpenAI API key not configured")
        return None
    
    # Construct image prompt
    prompt = construct_image_prompt(scene)
    
    # Define output file path
    assets_dir = Path(__file__).parent.parent / "assets"
    image_path = assets_dir / f"scene_{scene.id}.png"
    
    for attempt in range(max_retries + 1):
        try:
            logger.info(f"Generating image for scene {scene.id} (attempt {attempt + 1}/{max_retries + 1})")
            
            # Call DALL·E 3 API
            response = client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size="1792x1024",  # 16:9 aspect ratio
                quality="standard",  # or "hd" for higher quality but more cost
                n=1  # DALL·E 3 only supports n=1
            )
            
            # Extract image URL
            image_url = response.data[0].url
            logger.info(f"Image generated successfully: {image_url}")
            
            # Download and save image
            if download_image(image_url, image_path):
                return str(image_path)
            else:
                logger.error(f"Failed to download generated image for scene {scene.id}")
                if attempt < max_retries:
                    continue
                return None
                
        except openai.BadRequestError as e:
            logger.error(f"DALL·E 3 request error for scene {scene.id}: {e}")
            # Content policy violations are usually not retryable
            if "content_policy_violation" in str(e).lower():
                logger.warning(f"Content policy violation for scene {scene.id}, skipping retries")
                return None
            if attempt < max_retries:
                continue
            return None
            
        except openai.RateLimitError as e:
            logger.warning(f"Rate limit exceeded for scene {scene.id}: {e}")
            if attempt < max_retries:
                logger.info("Waiting before retry...")
                import time
                time.sleep(5)  # Wait 5 seconds before retry
                continue
            return None
            
        except Exception as e:
            logger.error(f"Unexpected error generating image for scene {scene.id}: {e}")
            if attempt < max_retries:
                continue
            return None
    
    return None


def generate_batch_images(scenes: list[Scene], max_concurrent: int = 3) -> dict[int, Optional[str]]:
    """
    Generate images for multiple scenes with basic rate limiting.
    
    Args:
        scenes (list[Scene]): List of scenes to generate images for
        max_concurrent (int): Maximum concurrent requests (basic rate limiting)
        
    Returns:
        dict[int, Optional[str]]: Mapping of scene ID to image path (or None if failed)
    """
    
    results = {}
    
    logger.info(f"Starting batch image generation for {len(scenes)} scenes")
    
    for i, scene in enumerate(scenes):
        logger.info(f"Processing scene {i + 1}/{len(scenes)}: {scene.title}")
        
        image_path = generate_scene_image(scene)
        results[scene.id] = image_path
        
        if image_path:
            logger.info(f"✅ Scene {scene.id} image generated: {image_path}")
        else:
            logger.error(f"❌ Failed to generate image for scene {scene.id}")
        
        # Basic rate limiting - wait between requests
        if i < len(scenes) - 1:  # Don't wait after the last scene
            import time
            time.sleep(2)  # Wait 2 seconds between requests
    
    success_count = sum(1 for path in results.values() if path is not None)
    logger.info(f"Batch generation complete: {success_count}/{len(scenes)} successful")
    
    return results


def cleanup_generated_images(scene_ids: list[int]) -> None:
    """
    Clean up generated image files for specified scene IDs.
    
    Args:
        scene_ids (list[int]): List of scene IDs to clean up
    """
    
    assets_dir = Path(__file__).parent.parent / "assets"
    
    for scene_id in scene_ids:
        image_path = assets_dir / f"scene_{scene_id}.png"
        
        if image_path.exists():
            try:
                image_path.unlink()
                logger.info(f"Cleaned up image for scene {scene_id}")
            except Exception as e:
                logger.error(f"Failed to clean up image for scene {scene_id}: {e}")


# Example usage and testing
if __name__ == "__main__":
    # Test with a sample scene
    from .scene_parser import Scene
    
    sample_scene = Scene(
        id=1,
        title="The Mysterious Library",
        characters=["Elena", "The Librarian"],
        setting="Ancient library with towering bookshelves and flickering candles",
        summary="Elena discovers a hidden section containing forbidden knowledge.",
        tone="mysterious"
    )
    
    print("🎨 Testing DALL·E 3 Image Generation")
    print("=" * 50)
    
    image_path = generate_scene_image(sample_scene)
    
    if image_path:
        print(f"✅ Image generated successfully: {image_path}")
    else:
        print("❌ Image generation failed - check OpenAI API key configuration") 