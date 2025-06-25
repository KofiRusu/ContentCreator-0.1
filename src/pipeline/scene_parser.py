"""
Scene Parser Module

Extracts structured scene metadata from story text using GPT-4o.
Each scene includes title, characters, setting, summary, and tone.
"""

import json
import logging
import os
from pathlib import Path
from typing import List, Optional

import openai
from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader
from pydantic import BaseModel, Field, ValidationError

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# OpenAI client will be initialized when needed
def get_openai_client():
    """Get OpenAI client instance."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY not found in environment variables")
        return None
    return openai.OpenAI(api_key=api_key)


class Scene(BaseModel):
    """Pydantic model for a single scene."""

    id: int = Field(..., description="Scene number/identifier")
    title: str = Field(..., max_length=100,
                       description="Short descriptive title")
    characters: List[str] = Field(
        default_factory=list, description="List of character names"
    )
    setting: str = Field(..., max_length=200,
                         description="Location and context")
    summary: str = Field(..., max_length=500,
                         description="1-3 line plot summary")
    tone: str = Field(..., max_length=50,
                      description="Emotional tone or style")


class SceneParseResult(BaseModel):
    """Pydantic model for the complete scene parsing result."""

    scenes: List[Scene] = Field(default_factory=list)
    success: bool = Field(default=True)
    error_message: Optional[str] = Field(default=None)


def load_prompt_template(story_text: str) -> str:
    """Load and render the scene parsing prompt template."""
    try:
        # Get template directory
        template_dir = Path(__file__).parent.parent / "templates"

        # Initialize Jinja2 environment
        env = Environment(loader=FileSystemLoader(template_dir))
        template = env.get_template("scene_parse_prompt.jinja2")

        # Render template with story text
        return template.render(story_text=story_text)

    except Exception as e:
        logger.error(f"Failed to load prompt template: {e}")
        # Fallback to inline prompt
        return f"""
You are an expert story analyst. Parse the following story into structured scenes.

For each scene, extract:
- id: scene number (starting from 1)
- title: short descriptive name
- characters: list of character names mentioned
- setting: brief location/context description
- summary: 1-3 line plot summary
- tone: emotional tone (e.g., suspenseful, comedic, dramatic)

Return ONLY valid JSON in this exact format:
{{
  "scenes": [
    {{
      "id": 1,
      "title": "Scene Title",
      "characters": ["Character1", "Character2"],
      "setting": "Location description",
      "summary": "Brief plot summary of what happens.",
      "tone": "emotional_tone"
    }}
  ]
}}

Story to parse:
{story_text}
"""


def call_openai_api(prompt: str, max_retries: int = 1) -> Optional[dict]:
    """Call OpenAI API with retry logic."""

    client = get_openai_client()
    if not client:
        logger.error("OpenAI API key not configured")
        return None

    for attempt in range(max_retries + 1):
        try:
            logger.info(
                f"Calling OpenAI API (attempt {attempt + 1}/{max_retries + 1})")

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert story analyst. Return only valid JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=2000,
            )

            # Extract content from response
            content = response.choices[0].message.content.strip()
            logger.info("Successfully received response from OpenAI")

            # Parse JSON response
            try:
                return json.loads(content)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                logger.debug(f"Raw response: {content}")
                if attempt < max_retries:
                    continue
                return None

        except Exception as e:
            logger.error(
                f"OpenAI API call failed (attempt {
                    attempt + 1}): {e}")
            if attempt < max_retries:
                continue
            return None

    return None


def parse_scenes(story_text: str) -> List[dict]:
    """
    Parse a story into structured scenes using GPT-4o.

    Args:
        story_text (str): The story text to parse

    Returns:
        List[dict]: List of scene dictionaries with structured metadata
    """

    if not story_text or not story_text.strip():
        logger.error("Empty or invalid story text provided")
        return []

    try:
        # Load and render prompt template
        prompt = load_prompt_template(story_text.strip())

        # Call OpenAI API
        response_data = call_openai_api(prompt)

        if not response_data:
            logger.error("Failed to get valid response from OpenAI API")
            return []

        # Validate response structure
        try:
            result = SceneParseResult(**response_data)

            # Convert Pydantic models to dictionaries
            scenes_list = [scene.model_dump() for scene in result.scenes]

            logger.info(f"Successfully parsed {len(scenes_list)} scenes")
            return scenes_list

        except ValidationError as e:
            logger.error(f"Response validation failed: {e}")
            return []

    except Exception as e:
        logger.error(f"Unexpected error in parse_scenes: {e}")
        return []


def parse_scenes_with_metadata(story_text: str) -> SceneParseResult:
    """
    Parse scenes and return full result with metadata.

    Args:
        story_text (str): The story text to parse

    Returns:
        SceneParseResult: Complete result with success status and error info
    """

    try:
        scenes = parse_scenes(story_text)

        if scenes:
            return SceneParseResult(
                scenes=[Scene(**scene) for scene in scenes], success=True
            )
        else:
            return SceneParseResult(
                scenes=[],
                success=False,
                error_message="Failed to parse any scenes from the story",
            )

    except Exception as e:
        logger.error(f"Error in parse_scenes_with_metadata: {e}")
        return SceneParseResult(scenes=[], success=False, error_message=str(e))


# Example usage and testing
if __name__ == "__main__":
    # Test with a sample story
    sample_story = """
    The forest was eerily quiet as Lena stepped through the undergrowth.
    Shadows danced between the trees, and she could feel eyes watching her.

    Suddenly, a figure emerged from behind an ancient oak. The stranger wore
    a dark cloak and spoke in riddles about the path ahead. "Not all who
    wander are lost," he said cryptically, "but some are exactly where they
    need to be."

    Lena felt a chill run down her spine. She knew this encounter would
    change everything.
    """

    scenes = parse_scenes(sample_story)
    print(json.dumps(scenes, indent=2))
