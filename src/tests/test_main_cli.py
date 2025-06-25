"""
Unit tests for main CLI functionality.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import pytest
from typer.testing import CliRunner

from src.main import app, load_story_from_file, check_existing_assets, create_summary_table
from src.pipeline.scene_parser import Scene


class TestStoryLoading:
    """Tests for story file loading."""

    def test_load_valid_story(self):
        """Test loading a valid story file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write("Once upon a time in a land far away...")
            temp_path = Path(f.name)
        
        try:
            content = load_story_from_file(temp_path)
            assert content == "Once upon a time in a land far away..."
        finally:
            temp_path.unlink()

    def test_load_nonexistent_file(self):
        """Test loading a file that doesn't exist."""
        with pytest.raises(FileNotFoundError):
            load_story_from_file(Path("nonexistent_file.txt"))

    def test_load_empty_file(self):
        """Test loading an empty file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write("")
            temp_path = Path(f.name)
        
        try:
            with pytest.raises(ValueError, match="Story file is empty"):
                load_story_from_file(temp_path)
        finally:
            temp_path.unlink()

    def test_load_whitespace_only_file(self):
        """Test loading a file with only whitespace."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write("   \n\t  \n   ")
            temp_path = Path(f.name)
        
        try:
            with pytest.raises(ValueError, match="Story file is empty"):
                load_story_from_file(temp_path)
        finally:
            temp_path.unlink()


class TestAssetChecking:
    """Tests for existing asset checking."""

    def test_check_existing_assets_none_exist(self):
        """Test checking when no assets exist."""
        scenes = [
            Scene(id=1, title="Scene 1", characters=[], setting="Test", summary="Test.", tone="test"),
            Scene(id=2, title="Scene 2", characters=[], setting="Test", summary="Test.", tone="test")
        ]
        
        with patch("src.main.Path") as mock_path:
            mock_path.return_value.parent = Path("test_assets")
            mock_path.return_value.exists.return_value = False
            
            # Mock the Path constructor behavior
            def mock_path_constructor(path_str):
                mock_obj = Mock()
                mock_obj.exists.return_value = False
                return mock_obj
            
            mock_path.side_effect = mock_path_constructor
            
            result = check_existing_assets(scenes)
            
            assert result == {1: False, 2: False}

    def test_check_existing_assets_some_exist(self):
        """Test checking when some assets exist."""
        scenes = [
            Scene(id=1, title="Scene 1", characters=[], setting="Test", summary="Test.", tone="test"),
            Scene(id=2, title="Scene 2", characters=[], setting="Test", summary="Test.", tone="test")
        ]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            assets_dir = Path(temp_dir)
            
            # Create one existing asset
            (assets_dir / "scene_1.png").touch()
            
            with patch("src.main.Path") as mock_path:
                mock_path.return_value.parent = assets_dir
                
                # Mock Path behavior to return correct paths
                def mock_path_div(self, other):
                    return assets_dir / other
                
                mock_path.return_value.__truediv__ = mock_path_div
                
                result = check_existing_assets(scenes)
                
                assert result[1] == True
                assert result[2] == False


class TestSummaryTable:
    """Tests for summary table creation."""

    def test_create_basic_summary_table(self):
        """Test creating a basic summary table."""
        scenes = [
            Scene(id=1, title="Scene 1", characters=["Alice"], setting="Forest", summary="Test.", tone="mysterious"),
            Scene(id=2, title="Scene 2", characters=["Bob"], setting="City", summary="Test.", tone="dramatic")
        ]
        
        results = {1: "path/to/scene_1.png", 2: None}
        existing_assets = {1: False, 2: False}
        
        table = create_summary_table(scenes, results, existing_assets, verbose=False)
        
        assert table.title == "🎬 Scene Processing Summary"
        assert len(table.columns) == 3  # Scene, Title, Status

    def test_create_verbose_summary_table(self):
        """Test creating a verbose summary table."""
        scenes = [
            Scene(id=1, title="Scene 1", characters=["Alice"], setting="Forest", summary="Test.", tone="mysterious")
        ]
        
        results = {1: "path/to/scene_1.png"}
        existing_assets = {1: False}
        
        table = create_summary_table(scenes, results, existing_assets, verbose=True)
        
        assert len(table.columns) == 5  # Scene, Title, Status, Characters, Tone


class TestCLICommands:
    """Tests for CLI commands."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def create_test_story_file(self, content: str) -> Path:
        """Create a temporary story file for testing."""
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8')
        temp_file.write(content)
        temp_file.close()
        return Path(temp_file.name)

    def test_generate_media_nonexistent_file(self):
        """Test generate-media command with nonexistent file."""
        result = self.runner.invoke(app, ["generate-media", "nonexistent_file.txt"])
        
        assert result.exit_code == 1
        assert "File not found" in result.stdout

    @patch('src.main.parse_scenes')
    @patch('src.main.generate_scene_image')
    def test_generate_media_successful(self, mock_generate_image, mock_parse_scenes):
        """Test successful generate-media command."""
        # Create test story file
        story_content = "Once upon a time, there was a magical forest."
        story_file = self.create_test_story_file(story_content)
        
        try:
            # Mock scene parsing
            mock_scenes = [
                Scene(id=1, title="Magical Forest", characters=["Hero"], setting="Forest", summary="Test.", tone="whimsical")
            ]
            mock_parse_scenes.return_value = mock_scenes
            
            # Mock image generation
            mock_generate_image.return_value = "src/assets/scene_1.png"
            
            # Run command
            result = self.runner.invoke(app, ["generate-media", str(story_file)])
            
            assert result.exit_code == 0
            assert "Parsed 1 scenes successfully" in result.stdout
            assert "Processing Complete!" in result.stdout
            
            # Verify mocks were called
            mock_parse_scenes.assert_called_once()
            mock_generate_image.assert_called_once()
            
        finally:
            story_file.unlink()

    @patch('src.main.parse_scenes')
    def test_generate_media_no_scenes_parsed(self, mock_parse_scenes):
        """Test generate-media when no scenes are parsed."""
        story_content = "This is not a proper story."
        story_file = self.create_test_story_file(story_content)
        
        try:
            # Mock empty scene parsing result
            mock_parse_scenes.return_value = []
            
            result = self.runner.invoke(app, ["generate-media", str(story_file)])
            
            assert result.exit_code == 1
            assert "No scenes could be parsed" in result.stdout
            
        finally:
            story_file.unlink()

    @patch('src.main.parse_scenes')
    @patch('src.main.generate_scene_image')
    def test_generate_media_with_verbose(self, mock_generate_image, mock_parse_scenes):
        """Test generate-media command with verbose flag."""
        story_content = "A story about adventure."
        story_file = self.create_test_story_file(story_content)
        
        try:
            # Mock scene parsing
            mock_scenes = [
                Scene(id=1, title="Adventure Begins", characters=["Hero"], setting="Mountain", summary="Test.", tone="adventurous")
            ]
            mock_parse_scenes.return_value = mock_scenes
            mock_generate_image.return_value = "src/assets/scene_1.png"
            
            result = self.runner.invoke(app, ["generate-media", str(story_file), "--verbose"])
            
            assert result.exit_code == 0
            assert "Scene 1: Adventure Begins" in result.stdout
            
        finally:
            story_file.unlink()

    @patch('src.main.parse_scenes')
    @patch('src.main.generate_scene_image')
    @patch('src.main.check_existing_assets')
    def test_generate_media_skip_existing(self, mock_check_assets, mock_generate_image, mock_parse_scenes):
        """Test generate-media command skipping existing assets."""
        story_content = "A story with existing assets."
        story_file = self.create_test_story_file(story_content)
        
        try:
            # Mock scene parsing
            mock_scenes = [
                Scene(id=1, title="Existing Scene", characters=["Hero"], setting="Castle", summary="Test.", tone="medieval")
            ]
            mock_parse_scenes.return_value = mock_scenes
            
            # Mock existing assets
            mock_check_assets.return_value = {1: True}  # Scene 1 already exists
            
            result = self.runner.invoke(app, ["generate-media", str(story_file)])
            
            assert result.exit_code == 0
            assert "skipped (already exists)" in result.stdout
            
            # Image generation should not be called for existing assets
            mock_generate_image.assert_not_called()
            
        finally:
            story_file.unlink()

    @patch('src.main.parse_scenes')
    @patch('src.main.generate_scene_image')
    def test_generate_media_max_scenes_limit(self, mock_generate_image, mock_parse_scenes):
        """Test generate-media command with max scenes limit."""
        story_content = "A long story with many scenes."
        story_file = self.create_test_story_file(story_content)
        
        try:
            # Mock scene parsing with multiple scenes
            mock_scenes = [
                Scene(id=i, title=f"Scene {i}", characters=[f"Character{i}"], 
                     setting=f"Location{i}", summary="Test.", tone="test")
                for i in range(1, 6)  # 5 scenes
            ]
            mock_parse_scenes.return_value = mock_scenes
            mock_generate_image.return_value = "src/assets/scene_1.png"
            
            result = self.runner.invoke(app, ["generate-media", str(story_file), "--max-scenes", "3"])
            
            assert result.exit_code == 0
            assert "Limited to first 3 scenes" in result.stdout
            
            # Should only call image generation 3 times
            assert mock_generate_image.call_count == 3
            
        finally:
            story_file.unlink()

    @patch('src.main.parse_scenes')
    def test_test_command(self, mock_parse_scenes):
        """Test the test pipeline command."""
        mock_scenes = [
            Scene(id=1, title="Test Scene", characters=["TestChar"], setting="Test Location", summary="Test.", tone="test")
        ]
        mock_parse_scenes.return_value = mock_scenes
        
        result = self.runner.invoke(app, ["test"])
        
        assert result.exit_code == 0
        assert "Testing AI Scene-to-Video Pipeline" in result.stdout
        assert "Successfully parsed 1 scenes" in result.stdout

    def test_version_command(self):
        """Test the version command."""
        result = self.runner.invoke(app, ["version"])
        
        assert result.exit_code == 0
        assert "AI Scene-to-Video Pipeline" in result.stdout
        assert "Version: 0.1.0" in result.stdout

    def test_generate_media_invalid_file_extension(self):
        """Test generate-media with non-.txt file (should show warning but continue)."""
        # Create a file with different extension
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8')
        temp_file.write("# Story\nOnce upon a time...")
        temp_file.close()
        story_file = Path(temp_file.name)
        
        try:
            with patch('src.main.parse_scenes') as mock_parse_scenes:
                mock_parse_scenes.return_value = []  # Empty to avoid further processing
                
                result = self.runner.invoke(app, ["generate-media", str(story_file)])
                
                assert "doesn't have .txt extension" in result.stdout
                
        finally:
            story_file.unlink()


class TestErrorHandling:
    """Tests for error handling scenarios."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    @patch('src.main.parse_scenes')
    def test_generate_media_parsing_exception(self, mock_parse_scenes):
        """Test generate-media when scene parsing raises an exception."""
        story_content = "Test story content."
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write(story_content)
            story_file = Path(f.name)
        
        try:
            # Mock scene parsing to raise an exception
            mock_parse_scenes.side_effect = Exception("API Error")
            
            result = self.runner.invoke(app, ["generate-media", str(story_file)])
            
            assert result.exit_code == 1
            assert "Unexpected Error" in result.stdout
            
        finally:
            story_file.unlink()

    @patch('src.main.parse_scenes')
    @patch('src.main.generate_scene_image')
    def test_generate_media_image_generation_failure(self, mock_generate_image, mock_parse_scenes):
        """Test generate-media when image generation fails."""
        story_content = "Test story."
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write(story_content)
            story_file = Path(f.name)
        
        try:
            # Mock successful parsing but failed image generation
            mock_scenes = [
                Scene(id=1, title="Test Scene", characters=[], setting="Test", summary="Test.", tone="test")
            ]
            mock_parse_scenes.return_value = mock_scenes
            mock_generate_image.return_value = None  # Simulate failure
            
            result = self.runner.invoke(app, ["generate-media", str(story_file)])
            
            assert result.exit_code == 1  # Should exit with error due to failed generation
            assert "generation failed" in result.stdout
            
        finally:
            story_file.unlink()


# Integration test fixtures
@pytest.fixture
def sample_story_file():
    """Create a sample story file for testing."""
    content = """
    Chapter 1: The Beginning
    
    Sarah packed her bags hurriedly. The letter had arrived that morning,
    and everything had changed. She looked around her small apartment one 
    last time before heading to the door.
    
    Chapter 2: The Journey
    
    The train station was crowded with commuters. Sarah clutched her ticket
    and searched for the right platform. An announcement echoed through
    the hall about delays, but she barely heard it.
    """
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        f.write(content)
        yield Path(f.name)
        Path(f.name).unlink()


class TestIntegration:
    """Integration tests for the complete CLI workflow."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    @patch('src.main.parse_scenes')
    @patch('src.main.generate_scene_image')
    def test_complete_workflow(self, mock_generate_image, mock_parse_scenes, sample_story_file):
        """Test the complete workflow from story file to generated images."""
        # Mock successful scene parsing
        mock_scenes = [
            Scene(
                id=1, 
                title="The Beginning", 
                characters=["Sarah"], 
                setting="Small apartment", 
                summary="Sarah prepares to leave.", 
                tone="melancholic"
            ),
            Scene(
                id=2, 
                title="The Journey", 
                characters=["Sarah"], 
                setting="Train station", 
                summary="Sarah begins her journey.", 
                tone="hopeful"
            )
        ]
        mock_parse_scenes.return_value = mock_scenes
        
        # Mock successful image generation
        mock_generate_image.side_effect = [
            "src/assets/scene_1.png",
            "src/assets/scene_2.png"
        ]
        
        result = self.runner.invoke(app, ["generate-media", str(sample_story_file), "--verbose"])
        
        assert result.exit_code == 0
        assert "Parsed 2 scenes successfully" in result.stdout
        assert "Scene 1: The Beginning" in result.stdout
        assert "Scene 2: The Journey" in result.stdout
        assert "Processing Complete!" in result.stdout
        
        # Verify both scenes were processed
        assert mock_generate_image.call_count == 2 