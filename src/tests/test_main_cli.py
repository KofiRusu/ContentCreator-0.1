"""
Tests for main.py CLI orchestration.

Test cases include:
- CLI command validation and argument parsing
- Scene parsing integration with mocked responses
- Image and video generation with mocked outputs
- Output formatting and logging validation
- Asset existence checking and skipping logic
- Error handling for various failure scenarios
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from typer.testing import CliRunner

from src.main import app
from src.pipeline.scene_parser import Scene


@pytest.fixture
def runner():
    """Create a CliRunner instance for testing."""
    return CliRunner()


@pytest.fixture
def mock_scenes():
    """Create mock scenes for testing."""
    return [
        Scene(
            id=1,
            title="Forest Encounter",
            characters=["Sarah", "Mysterious Figure"],
            setting="Dark forest with ancient oak trees",
            summary="Detective Sarah explores the eerie forest and encounters a mysterious cloaked figure",
            tone="mysterious, suspenseful"
        ),
        Scene(
            id=2,
            title="The Revelation",
            characters=["Sarah", "Mysterious Figure"],
            setting="Moonlit clearing in the forest",
            summary="The mysterious figure reveals crucial information about Sarah's investigation",
            tone="dramatic, revelatory"
        ),
        Scene(
            id=3,
            title="Chase Through Shadows",
            characters=["Sarah"],
            setting="Dense forest undergrowth",
            summary="Sarah pursues clues through the treacherous forest terrain",
            tone="intense, action-packed"
        )
    ]


@pytest.fixture
def temp_story_file():
    """Create a temporary story file for testing."""
    story_content = """
    The ancient forest was eerily quiet as Detective Sarah Chen stepped through
    the undergrowth. Her flashlight cut through the darkness, revealing twisted
    branches that seemed to reach out like gnarled fingers.

    Suddenly, a figure emerged from behind an ancient oak tree. The stranger
    wore a dark cloak and spoke in riddles about the path ahead. "Not all who
    wander are lost," he said cryptically, "but some are exactly where they
    need to be."

    Sarah felt a chill run down her spine as the figure vanished into the mist.
    She knew this encounter would change everything about her investigation.
    """

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(story_content)
        temp_path = f.name

    yield temp_path

    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


class TestGenerateMediaCommand:
    """Test cases for the generate-media CLI command."""

    @patch('src.main.generate_scene_image')
    @patch('src.main.parse_scenes')
    def test_generate_media_success(self, mock_parse_scenes, mock_generate_image,
                                    runner, mock_scenes, temp_story_file):
        """Test successful media generation with mocked functions."""
        # Setup mocks
        mock_parse_scenes.return_value = [
            scene.model_dump() for scene in mock_scenes]
        mock_generate_image.side_effect = [
            "/path/to/scene_1.png",
            "/path/to/scene_2.png",
            "/path/to/scene_3.png"
        ]

        # Run command
        result = runner.invoke(app, ["generate-media", temp_story_file])

        # Assertions
        assert result.exit_code == 0
        assert "Loading story from:" in result.stdout
        assert "Parsing scenes with GPT-4o" in result.stdout
        assert "Parsed 3 scenes successfully" in result.stdout
        assert "Generating images with DALL·E 3" in result.stdout
        assert "Processing Complete!" in result.stdout

        # Verify mocks were called
        mock_parse_scenes.assert_called_once()
        assert mock_generate_image.call_count == 3

    @patch('src.main.generate_scene_image')
    @patch('src.main.parse_scenes')
    def test_generate_media_with_verbose(self,
        mock_parse_scenes,
        mock_generate_image,
                                         runner, mock_scenes, temp_story_file):
        """Test generate-media with verbose output."""
        mock_parse_scenes.return_value = [
            scene.model_dump() for scene in mock_scenes]
        mock_generate_image.return_value = "/path/to/scene_1.png"

        result = runner.invoke(
            app, ["generate-media", temp_story_file, "--verbose"])

        assert result.exit_code == 0
        assert "Scene 1:" in result.stdout
        assert "Scene 2:" in result.stdout
        assert "Scene 3:" in result.stdout

    @patch('src.main.generate_scene_image')
    @patch('src.main.parse_scenes')
    def test_generate_media_max_scenes(self, mock_parse_scenes, mock_generate_image,
                                       runner, mock_scenes, temp_story_file):
        """Test generate-media with max-scenes limit."""
        mock_parse_scenes.return_value = [
            scene.model_dump() for scene in mock_scenes]
        mock_generate_image.return_value = "/path/to/scene_1.png"

        result = runner.invoke(
            app, ["generate-media", temp_story_file, "--max-scenes", "2"])

        assert result.exit_code == 0
        assert "Limited to first 2 scenes" in result.stdout
        assert mock_generate_image.call_count == 2

    @patch('src.main.check_existing_assets')
    @patch('src.main.generate_scene_image')
    @patch('src.main.parse_scenes')
    def test_generate_media_skip_existing(self,
        mock_parse_scenes,
        mock_generate_image,
                                          mock_check_assets,
                                              runner,
                                              mock_scenes,
                                              temp_story_file):
        """Test generate-media with existing asset skipping."""
        mock_parse_scenes.return_value = [
            scene.model_dump() for scene in mock_scenes]
        mock_check_assets.return_value = {
            1: True, 2: False, 3: False}  # Scene 1 exists
        mock_generate_image.return_value = "/path/to/scene.png"

        result = runner.invoke(
            app, ["generate-media", temp_story_file, "--skip-existing"])

        assert result.exit_code == 0
        assert "Found 1 existing images" in result.stdout
        assert "skipped (already exists)" in result.stdout
        assert mock_generate_image.call_count == 2  # Only scenes 2 and 3

    @patch('src.main.parse_scenes')
    def test_generate_media_no_scenes_parsed(
            self, mock_parse_scenes, runner, temp_story_file):
        """Test generate-media when no scenes are parsed."""
        mock_parse_scenes.return_value = []

        result = runner.invoke(app, ["generate-media", temp_story_file])

        assert result.exit_code == 1
        assert "No scenes could be parsed from the story" in result.stdout

    def test_generate_media_file_not_found(self, runner):
        """Test generate-media with non-existent file."""
        result = runner.invoke(app, ["generate-media", "nonexistent.txt"])

        assert result.exit_code == 1
        assert "File not found" in result.stdout

    @patch('src.main.generate_scene_image')
    @patch('src.main.parse_scenes')
    def test_generate_media_image_generation_failure(self,
        mock_parse_scenes,
        mock_generate_image,
                                                     runner,
                                                         mock_scenes,
                                                         temp_story_file):
        """Test generate-media when some image generations fail."""
        mock_parse_scenes.return_value = [
            scene.model_dump() for scene in mock_scenes[:2]]
        mock_generate_image.side_effect = [
            "/path/to/scene_1.png", None]  # Second fails

        result = runner.invoke(app, ["generate-media", temp_story_file])

        assert result.exit_code == 1
        assert "generation failed" in result.stdout


class TestGenerateVideosCommand:
    """Test cases for the generate-videos CLI command."""

    @patch('src.main.VideoGenerator')
    @patch('src.main.parse_scenes')
    def test_generate_videos_dry_run(self, mock_parse_scenes, mock_video_generator,
                                     runner, mock_scenes, temp_story_file):
        """Test generate-videos in dry run mode."""
        mock_parse_scenes.return_value = [
            scene.model_dump() for scene in mock_scenes]
        mock_generator_instance = Mock()
        mock_generator_instance.generate_scene_video.return_value = "/path/to/scene_1.mp4"
        mock_generator_instance.get_session_summary.return_value = {
            "videos_generated": 3,
            "total_cost": 15.0,
            "budget_remaining": 35.0,
            "videos": []
        }
        mock_video_generator.return_value = mock_generator_instance

        result = runner.invoke(
            app, ["generate-videos", temp_story_file, "--dry-run"])

        assert result.exit_code == 0
        assert "Generating videos with fal.ai Veo 3 (DRY RUN mode)" in result.stdout
        assert "Video Generation Complete!" in result.stdout

    @patch('src.main.VideoGenerator')
    @patch('src.main.parse_scenes')
    def test_generate_videos_with_budget_warning(self,
        mock_parse_scenes,
        mock_video_generator,
                                                 runner,
                                                     mock_scenes,
                                                     temp_story_file):
        """Test generate-videos shows budget warning in production mode."""
        mock_parse_scenes.return_value = [
            scene.model_dump() for scene in mock_scenes]

        # Simulate user declining the budget warning
        result = runner.invoke(
            app, ["generate-videos", temp_story_file], input="n\n")

        assert result.exit_code == 0
        assert "WARNING: Video generation incurs real costs" in result.stdout
        assert "Operation cancelled by user" in result.stdout

    @patch('src.main.check_existing_assets')
    @patch('src.main.VideoGenerator')
    @patch('src.main.parse_scenes')
    def test_generate_videos_with_images(self,
        mock_parse_scenes,
        mock_video_generator,
                                         mock_check_assets,
                                             runner,
                                             mock_scenes,
                                             temp_story_file):
        """Test generate-videos using existing images as references."""
        mock_parse_scenes.return_value = [
            scene.model_dump() for scene in mock_scenes]
        mock_check_assets.return_value = {
            1: True, 2: True, 3: False}  # 2 images exist
        mock_generator_instance = Mock()
        mock_generator_instance.generate_scene_video.return_value = "/path/to/scene.mp4"
        mock_generator_instance.get_session_summary.return_value = {
            "videos_generated": 1,
            "total_cost": 5.0,
            "budget_remaining": 45.0,
            "videos": []
        }
        mock_video_generator.return_value = mock_generator_instance

        result = runner.invoke(
            app, ["generate-videos", temp_story_file, "--dry-run", "--use-images"])

        assert result.exit_code == 0
        assert "Found 2 existing images to use as references" in result.stdout


class TestGenerateAllCommand:
    """Test cases for the generate-all CLI command."""

    @patch('src.main.VideoGenerator')
    @patch('src.main.generate_scene_image')
    @patch('src.main.parse_scenes')
    def test_generate_all_success(self, mock_parse_scenes, mock_generate_image,
                                  mock_video_generator,
                                      runner,
                                      mock_scenes,
                                      temp_story_file):
        """Test complete generate-all pipeline."""
        mock_parse_scenes.return_value = [
            scene.model_dump() for scene in mock_scenes]
        mock_generate_image.return_value = "/path/to/scene.png"

        mock_generator_instance = Mock()
        mock_generator_instance.generate_scene_video.return_value = "/path/to/scene.mp4"
        mock_generator_instance.get_session_summary.return_value = {
            "videos_generated": 3,
            "total_cost": 15.0,
            "budget_remaining": 35.0,
            "videos": []
        }
        mock_video_generator.return_value = mock_generator_instance

        result = runner.invoke(
            app, ["generate-all", temp_story_file, "--dry-run-videos"], input="y\n")

        assert result.exit_code == 0
        assert "Starting Complete AI Scene-to-Video Pipeline" in result.stdout
        assert "Step 1: Generating scene images" in result.stdout
        assert "Step 2: Generating scene videos" in result.stdout
        assert "Complete Pipeline Finished!" in result.stdout

    @patch('src.main.generate_scene_image')
    @patch('src.main.parse_scenes')
    def test_generate_all_image_failure_continue(self,
        mock_parse_scenes,
        mock_generate_image,
                                                 runner,
                                                     mock_scenes,
                                                     temp_story_file):
        """Test generate-all continues after image generation failures."""
        mock_parse_scenes.return_value = [
            scene.model_dump() for scene in mock_scenes]
        mock_generate_image.side_effect = Exception("Image generation failed")

        result = runner.invoke(app,
                               ["generate-all",
                                temp_story_file,
                                "--dry-run-videos"],
                               input="y\ny\n")

        assert result.exit_code == 0
        assert "Image generation failed" in result.stdout
        assert "Continue with video generation despite image failures?" in result.stdout


class TestOtherCommands:
    """Test cases for other CLI commands."""

    @patch('src.main.VideoGenerator')
    @patch('src.main.parse_scenes')
    def test_test_command(self, mock_parse_scenes,
                          mock_video_generator, runner, mock_scenes):
        """Test the test command functionality."""
        mock_parse_scenes.return_value = [
            scene.model_dump() for scene in mock_scenes]

        mock_generator_instance = Mock()
        mock_generator_instance.generate_scene_video.return_value = "/path/to/scene.mp4"
        mock_generator_instance.get_session_summary.return_value = {
            "videos_generated": 2,
            "total_cost": 10.0,
            "budget_remaining": 0.0,
            "videos": []
        }
        mock_video_generator.return_value = mock_generator_instance

        result = runner.invoke(app, ["test"])

        assert result.exit_code == 0
        assert "Testing AI Scene-to-Video Pipeline" in result.stdout
        assert "Testing video generation (dry run mode)" in result.stdout
        assert "Pipeline test complete!" in result.stdout

    @patch('src.main.parse_scenes')
    def test_test_command_no_scenes(self, mock_parse_scenes, runner):
        """Test the test command when scene parsing fails."""
        mock_parse_scenes.return_value = []

        result = runner.invoke(app, ["test"])

        assert result.exit_code == 0
        assert "Scene parsing returned empty results" in result.stdout
        assert "Testing video generation with mock scene" in result.stdout

    def test_version_command(self, runner):
        """Test the version command."""
        result = runner.invoke(app, ["version"])

        assert result.exit_code == 0
        assert "AI Scene-to-Video Pipeline" in result.stdout
        assert "Version: 0.1.0" in result.stdout
        assert "fal.ai Veo 3" in result.stdout
        assert "Available Commands:" in result.stdout


class TestErrorHandling:
    """Test cases for error handling scenarios."""

    def test_invalid_file_extension_warning(self, runner):
        """Test warning for non-.txt files."""
        with tempfile.NamedTemporaryFile(suffix='.doc', delete=False) as f:
            f.write(b"test content")
            temp_path = f.name

        try:
            result = runner.invoke(app, ["generate-media", temp_path])
            assert "doesn't have .txt extension" in result.stdout
        finally:
            Path(temp_path).unlink(missing_ok=True)

    @patch('src.main.load_story_from_file')
    def test_unicode_decode_error(
            self, mock_load_story, runner, temp_story_file):
        """Test handling of unicode decode errors."""
        mock_load_story.side_effect = UnicodeDecodeError(
            "utf-8", b"", 0, 1, "invalid")

        result = runner.invoke(app, ["generate-media", temp_story_file])

        assert result.exit_code == 1
        assert "Encoding Error" in result.stdout

    @patch('src.main.parse_scenes')
    def test_unexpected_error_handling(
            self, mock_parse_scenes, runner, temp_story_file):
        """Test handling of unexpected errors."""
        mock_parse_scenes.side_effect = Exception("Unexpected error")

        result = runner.invoke(app, ["generate-media", temp_story_file])

        assert result.exit_code == 1
        assert "Unexpected Error" in result.stdout


class TestOutputFormatting:
    """Test cases for output formatting and display."""

    @patch('src.main.generate_scene_image')
    @patch('src.main.parse_scenes')
    def test_summary_table_creation(self, mock_parse_scenes, mock_generate_image,
                                    runner, mock_scenes, temp_story_file):
        """Test that summary table is properly displayed."""
        mock_parse_scenes.return_value = [
            scene.model_dump() for scene in mock_scenes]
        mock_generate_image.side_effect = [
            "/path/to/scene_1.png", None, "/path/to/scene_3.png"]

        result = runner.invoke(app, ["generate-media", temp_story_file])

        # Check that summary statistics are displayed
        assert "Statistics:" in result.stdout
        assert "Total scenes:" in result.stdout
        assert "Generated:" in result.stdout
        assert "Failed:" in result.stdout

    @patch('src.main.generate_scene_image')
    @patch('src.main.parse_scenes')
    def test_progress_indicators(self, mock_parse_scenes, mock_generate_image,
                                 runner, mock_scenes, temp_story_file):
        """Test that progress indicators are shown."""
        mock_parse_scenes.return_value = [
            scene.model_dump() for scene in mock_scenes]
        mock_generate_image.return_value = "/path/to/scene.png"

        result = runner.invoke(app, ["generate-media", temp_story_file])

        # Progress indicators should be present
        assert "Parsing story into scenes" in result.stdout
        assert "Generating scene images" in result.stdout


class TestAssetManagement:
    """Test cases for asset existence checking and management."""

    @patch('src.main.check_existing_assets')
    @patch('src.main.generate_scene_image')
    @patch('src.main.parse_scenes')
    def test_asset_directory_display(self, mock_parse_scenes, mock_generate_image,
                                     mock_check_assets,
                                         runner,
                                         mock_scenes,
                                         temp_story_file):
        """Test that assets directory is properly displayed."""
        mock_parse_scenes.return_value = [
            scene.model_dump() for scene in mock_scenes]
        mock_generate_image.return_value = "/path/to/scene.png"
        mock_check_assets.return_value = {1: False, 2: False, 3: False}

        result = runner.invoke(app, ["generate-media", temp_story_file])

        assert "Assets saved to:" in result.stdout

    @patch('src.main.check_existing_assets')
    @patch('src.main.parse_scenes')
    def test_overwrite_mode(
            self,
                mock_parse_scenes,
                mock_check_assets,
                runner,
                mock_scenes,
                temp_story_file):
        """Test overwrite mode ignores existing assets."""
        mock_parse_scenes.return_value = [
            scene.model_dump() for scene in mock_scenes]
        mock_check_assets.return_value = {
            1: True, 2: True, 3: True}  # All exist

        with patch('src.main.generate_scene_image') as mock_generate_image:
            mock_generate_image.return_value = "/path/to/scene.png"

            result = runner.invoke(
                app, ["generate-media", temp_story_file, "--overwrite"])

            assert result.exit_code == 0
            # Should generate all 3 scenes despite existing assets
            assert mock_generate_image.call_count == 3


# Integration tests
class TestCLIIntegration:
    """Integration test cases for complete CLI workflows."""

    @patch('src.main.VideoGenerator')
    @patch('src.main.generate_scene_image')
    @patch('src.main.parse_scenes')
    def test_complete_workflow_dry_run(self, mock_parse_scenes, mock_generate_image,
                                       mock_video_generator,
                                           runner,
                                           mock_scenes,
                                           temp_story_file):
        """Test complete workflow from story to videos in dry run mode."""
        # Setup all mocks
        mock_parse_scenes.return_value = [
            scene.model_dump() for scene in mock_scenes]
        mock_generate_image.return_value = "/path/to/scene.png"

        mock_generator_instance = Mock()
        mock_generator_instance.generate_scene_video.return_value = "/path/to/scene.mp4"
        mock_generator_instance.get_session_summary.return_value = {
            "videos_generated": 3,
            "total_cost": 0.0,  # Dry run
            "budget_remaining": 50.0,
            "videos": []
        }
        mock_video_generator.return_value = mock_generator_instance

        # Test generate-media first
        result1 = runner.invoke(app, ["generate-media", temp_story_file])
        assert result1.exit_code == 0

        # Test generate-videos in dry run
        result2 = runner.invoke(
            app, ["generate-videos", temp_story_file, "--dry-run"])
        assert result2.exit_code == 0

        # Test complete pipeline
        result3 = runner.invoke(
            app, ["generate-all", temp_story_file, "--dry-run-videos"])
        assert result3.exit_code == 0

        # Verify all components worked
        assert "Processing Complete!" in result1.stdout
        assert "Video Generation Complete!" in result2.stdout
        assert "Complete Pipeline Finished!" in result3.stdout
