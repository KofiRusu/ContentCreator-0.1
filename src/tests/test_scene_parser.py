"""
Unit tests for scene_parser.py module.
"""

import json
import os
from unittest.mock import Mock, patch

import pytest
from pydantic import ValidationError

from src.pipeline.scene_parser import (
    Scene,
    SceneParseResult,
    call_openai_api,
    load_prompt_template,
    parse_scenes,
    parse_scenes_with_metadata,
)


class TestSceneModel:
    """Tests for the Scene Pydantic model."""

    def test_valid_scene_creation(self):
        """Test creating a valid scene object."""
        scene_data = {
            "id": 1,
            "title": "The Forest Encounter",
            "characters": ["Lena", "The Stranger"],
            "setting": "Dense forest at dusk",
            "summary": "Lena stumbles upon a mysterious figure.",
            "tone": "Suspenseful",
        }

        scene = Scene(**scene_data)

        assert scene.id == 1
        assert scene.title == "The Forest Encounter"
        assert scene.characters == ["Lena", "The Stranger"]
        assert scene.setting == "Dense forest at dusk"
        assert scene.tone == "Suspenseful"

    def test_scene_with_empty_characters(self):
        """Test scene with no characters."""
        scene_data = {
            "id": 1,
            "title": "Empty Scene",
            "setting": "Abandoned house",
            "summary": "The house stands empty.",
            "tone": "Eerie",
        }

        scene = Scene(**scene_data)
        assert scene.characters == []

    def test_invalid_scene_missing_required_fields(self):
        """Test that missing required fields raise ValidationError."""
        scene_data = {
            "id": 1,
            "title": "Incomplete Scene",
            # Missing required fields: setting, summary, tone
        }

        with pytest.raises(ValidationError):
            Scene(**scene_data)


class TestSceneParseResult:
    """Tests for the SceneParseResult model."""

    def test_successful_result(self):
        """Test creating a successful parse result."""
        scenes = [
            Scene(
                id=1,
                title="Test Scene",
                characters=["Alice"],
                setting="Test Location",
                summary="Test summary",
                tone="Test tone",
            )
        ]

        result = SceneParseResult(scenes=scenes)

        assert result.success is True
        assert result.error_message is None
        assert len(result.scenes) == 1

    def test_error_result(self):
        """Test creating an error result."""
        result = SceneParseResult(
            scenes=[], success=False, error_message="API call failed"
        )

        assert result.success is False
        assert result.error_message == "API call failed"
        assert len(result.scenes) == 0


class TestPromptTemplate:
    """Tests for prompt template loading."""

    def test_load_prompt_template_fallback(self):
        """Test that fallback prompt is used when template file doesn't exist."""
        story_text = "Once upon a time..."

        # This should use the fallback prompt since template might not exist
        prompt = load_prompt_template(story_text)

        assert story_text in prompt
        assert "You are an expert story analyst" in prompt
        assert "JSON" in prompt


class TestOpenAIAPI:
    """Tests for OpenAI API integration."""

    @patch("src.pipeline.scene_parser.get_openai_client")
    def test_successful_api_call(self, mock_get_client):
        """Test successful OpenAI API call."""
        # Mock client and response
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps(
            {
                "scenes": [
                    {
                        "id": 1,
                        "title": "Test Scene",
                        "characters": ["Alice"],
                        "setting": "Wonderland",
                        "summary": "Alice falls down the rabbit hole.",
                        "tone": "Whimsical",
                    }
                ]
            }
        )
        mock_client.chat.completions.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        result = call_openai_api("test prompt")

        assert result is not None
        assert "scenes" in result
        assert len(result["scenes"]) == 1

    @patch("src.pipeline.scene_parser.get_openai_client")
    def test_api_call_with_invalid_json(self, mock_get_client):
        """Test API call that returns invalid JSON."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Invalid JSON response"
        mock_client.chat.completions.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        result = call_openai_api("test prompt")

        assert result is None

    @patch("src.pipeline.scene_parser.get_openai_client")
    def test_api_call_exception(self, mock_get_client):
        """Test API call that raises an exception."""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        mock_get_client.return_value = mock_client

        result = call_openai_api("test prompt")

        assert result is None


class TestSceneParsing:
    """Integration tests for scene parsing."""

    def test_parse_scenes_empty_input(self):
        """Test parsing empty or None input."""
        assert parse_scenes("") == []
        assert parse_scenes("   ") == []
        assert parse_scenes(None) == []

    @patch("src.pipeline.scene_parser.call_openai_api")
    def test_parse_scenes_success(self, mock_api):
        """Test successful scene parsing."""
        # Mock API response
        mock_api.return_value = {
            "scenes": [
                {
                    "id": 1,
                    "title": "Opening Scene",
                    "characters": ["Hero"],
                    "setting": "Village square",
                    "summary": "The hero begins their journey.",
                    "tone": "Hopeful",
                },
                {
                    "id": 2,
                    "title": "The Challenge",
                    "characters": ["Hero", "Villain"],
                    "setting": "Dark forest",
                    "summary": "The hero faces their first obstacle.",
                    "tone": "Tense",
                },
            ]
        }

        story = "A hero begins a journey and faces challenges."
        scenes = parse_scenes(story)

        assert len(scenes) == 2
        assert scenes[0]["title"] == "Opening Scene"
        assert scenes[1]["characters"] == ["Hero", "Villain"]

    @patch("src.pipeline.scene_parser.call_openai_api")
    def test_parse_scenes_api_failure(self, mock_api):
        """Test scene parsing when API fails."""
        mock_api.return_value = None

        story = "A simple story."
        scenes = parse_scenes(story)

        assert scenes == []

    @patch("src.pipeline.scene_parser.call_openai_api")
    def test_parse_scenes_invalid_response(self, mock_api):
        """Test scene parsing with invalid API response structure."""
        mock_api.return_value = {"invalid": "structure"}

        story = "A simple story."
        scenes = parse_scenes(story)

        assert scenes == []


class TestParseWithMetadata:
    """Tests for parse_scenes_with_metadata function."""

    @patch("src.pipeline.scene_parser.parse_scenes")
    def test_parse_with_metadata_success(self, mock_parse):
        """Test successful parsing with metadata."""
        mock_parse.return_value = [
            {
                "id": 1,
                "title": "Test Scene",
                "characters": ["Alice"],
                "setting": "Wonderland",
                "summary": "Alice explores.",
                "tone": "Curious",
            }
        ]

        result = parse_scenes_with_metadata("Test story")

        assert result.success is True
        assert result.error_message is None
        assert len(result.scenes) == 1
        assert result.scenes[0].title == "Test Scene"

    @patch("src.pipeline.scene_parser.parse_scenes")
    def test_parse_with_metadata_failure(self, mock_parse):
        """Test parsing failure with metadata."""
        mock_parse.return_value = []

        result = parse_scenes_with_metadata("Test story")

        assert result.success is False
        assert result.error_message is not None
        assert len(result.scenes) == 0


# Sample story fixtures for testing
@pytest.fixture
def sample_story_short():
    """Short story for testing."""
    return """
    The old lighthouse keeper walked to the edge of the cliff. 
    The storm was approaching, and he needed to light the beacon.
    As the first ships appeared on the horizon, he smiled with satisfaction.
    """


@pytest.fixture
def sample_story_multi_scene():
    """Multi-scene story for testing."""
    return """
    Chapter 1: The Beginning
    
    Sarah packed her bags hurriedly. The letter had arrived that morning,
    and everything had changed. She looked around her small apartment one 
    last time before heading to the door.
    
    Chapter 2: The Journey
    
    The train station was crowded with commuters. Sarah clutched her ticket
    and searched for the right platform. An announcement echoed through
    the hall about delays, but she barely heard it.
    
    Chapter 3: Arrival
    
    The countryside rolled past the window as Sarah dozed. When she woke,
    the train was pulling into a small station she'd never seen before.
    This was it - her new beginning.
    """


class TestIntegrationWithFixtures:
    """Integration tests using story fixtures."""

    @patch("src.pipeline.scene_parser.call_openai_api")
    def test_short_story_parsing(self, mock_api, sample_story_short):
        """Test parsing a short story."""
        mock_api.return_value = {
            "scenes": [
                {
                    "id": 1,
                    "title": "The Lighthouse Keeper's Duty",
                    "characters": ["lighthouse keeper"],
                    "setting": "Cliff edge near lighthouse during storm",
                    "summary": "The keeper prepares to light the beacon as ships approach.",
                    "tone": "Determined",
                }
            ]
        }

        scenes = parse_scenes(sample_story_short)

        assert len(scenes) == 1
        assert "lighthouse keeper" in scenes[0]["characters"]
        assert "storm" in scenes[0]["setting"]

    def test_empty_story_handling(self):
        """Test handling of empty story input."""
        result = parse_scenes_with_metadata("")

        assert result.success is False
        assert len(result.scenes) == 0
        assert result.error_message is not None
