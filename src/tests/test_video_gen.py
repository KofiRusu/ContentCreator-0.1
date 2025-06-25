"""
Tests for video generation module.

Test cases include:
- Video generation with mock API responses
- Budget tracking and limits
- Error handling and retries
- Caching functionality
- Dry run and simulation modes
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.pipeline.scene_parser import Scene
from src.pipeline.video_gen import (
    BudgetExceededException,
    VideoGenerationError,
    VideoGenerator,
    generate_scene_video,
)


@pytest.fixture
def sample_scene():
    """Create a sample scene for testing."""
    return Scene(
        id=1,
        title="Forest Encounter",
        characters=["Lena", "Mysterious Stranger"],
        setting="Dark forest with ancient oak trees, moonlight filtering through leaves",
        summary="Lena walks cautiously through the eerie forest when a cloaked stranger emerges from the shadows and speaks in riddles about her destiny",
        tone="mysterious, suspenseful, atmospheric"
    )


@pytest.fixture
def video_generator():
    """Create a VideoGenerator instance for testing."""
    return VideoGenerator(dry_run=True, budget_limit=10.0)


class TestVideoGenerator:
    """Test cases for VideoGenerator class."""
    
    def test_init_dry_run(self):
        """Test VideoGenerator initialization in dry run mode."""
        generator = VideoGenerator(dry_run=True)
        assert generator.dry_run is True
        assert generator.simulate is False
        assert generator.session_cost == 0.0
        assert generator.generated_videos == []
        assert generator.assets_dir.exists()
    
    def test_init_simulate(self):
        """Test VideoGenerator initialization in simulate mode."""
        generator = VideoGenerator(simulate=True)
        assert generator.dry_run is False
        assert generator.simulate is True
        assert generator.session_cost == 0.0
    
    def test_budget_tracking(self, video_generator):
        """Test budget tracking functionality."""
        # Test budget check passes
        video_generator._check_budget(5.0)
        
        # Test budget exceeded
        with pytest.raises(BudgetExceededException):
            video_generator._check_budget(15.0)
    
    def test_prompt_generation(self, video_generator, sample_scene):
        """Test video prompt generation from scene data."""
        prompt = video_generator._build_video_prompt(sample_scene)
        
        assert "Forest Encounter" in prompt
        assert "Dark forest" in prompt
        assert "Lena" in prompt
        assert "mysterious, suspenseful" in prompt
        assert "Cinematic" in prompt
        assert "1080p" in prompt
    
    def test_prompt_hash_generation(self, video_generator):
        """Test prompt hash generation for caching."""
        prompt1 = "Test prompt"
        prompt2 = "Different prompt"
        
        hash1 = video_generator._generate_prompt_hash(prompt1)
        hash2 = video_generator._generate_prompt_hash(prompt2)
        
        assert isinstance(hash1, str)
        assert len(hash1) == 12
        assert hash1 != hash2
        
        # Same prompt should generate same hash
        hash1_repeat = video_generator._generate_prompt_hash(prompt1)
        assert hash1 == hash1_repeat
    
    def test_video_generation_dry_run(self, video_generator, sample_scene):
        """Test video generation in dry run mode."""
        video_path = video_generator.generate_scene_video(sample_scene)
        
        assert video_path is not None
        assert Path(video_path).exists()
        assert "scene_1.mp4" in video_path
        
        # Check session tracking
        summary = video_generator.get_session_summary()
        assert summary["videos_generated"] == 1
        assert summary["total_cost"] > 0
        assert summary["dry_run"] is True
    
    def test_video_generation_simulate(self, sample_scene):
        """Test video generation in simulate mode."""
        generator = VideoGenerator(simulate=True, budget_limit=10.0)
        video_path = generator.generate_scene_video(sample_scene)
        
        assert video_path is not None
        assert Path(video_path).exists()
        
        # Check session tracking
        summary = generator.get_session_summary()
        assert summary["videos_generated"] == 1
        assert summary["simulate"] is True
    
    def test_budget_exceeded(self, sample_scene):
        """Test budget exceeded scenario."""
        generator = VideoGenerator(dry_run=True, budget_limit=1.0)
        
        # First video should succeed
        video_path1 = generator.generate_scene_video(sample_scene)
        assert video_path1 is not None
        
        # Second video should fail due to budget
        scene2 = Scene(**sample_scene.model_dump())
        scene2.id = 2
        video_path2 = generator.generate_scene_video(scene2)
        assert video_path2 is None
    
    def test_caching_functionality(self, video_generator, sample_scene):
        """Test video caching functionality."""
        # Generate video first time
        video_path1 = video_generator.generate_scene_video(sample_scene)
        assert video_path1 is not None
        
        # Generate same scene again - should use cache
        with patch.object(video_generator, '_call_fal_api') as mock_api:
            video_path2 = video_generator.generate_scene_video(sample_scene)
            assert video_path2 == video_path1
            mock_api.assert_not_called()  # Should not call API due to cache
    
    def test_session_summary(self, video_generator, sample_scene):
        """Test session summary functionality."""
        # Initial summary
        summary = video_generator.get_session_summary()
        assert summary["videos_generated"] == 0
        assert summary["total_cost"] == 0.0
        
        # Generate a video
        video_generator.generate_scene_video(sample_scene)
        
        # Updated summary
        summary = video_generator.get_session_summary()
        assert summary["videos_generated"] == 1
        assert summary["total_cost"] > 0
        assert len(summary["videos"]) == 1
        assert summary["videos"][0]["scene_id"] == 1


class TestVideoGenerationAPI:
    """Test cases for fal.ai API interaction."""
    
    @patch('src.pipeline.video_gen.fal_client')
    def test_fal_api_success(self, mock_fal_client, sample_scene):
        """Test successful fal.ai API call."""
        # Mock successful API response
        mock_fal_client.subscribe.return_value = {
            "video": {"url": "https://example.com/test_video.mp4"},
            "duration": 10
        }
        
        with patch('src.pipeline.video_gen.requests.get') as mock_get:
            # Mock successful download
            mock_response = Mock()
            mock_response.iter_content.return_value = [b"fake video data"]
            mock_get.return_value = mock_response
            
            generator = VideoGenerator(dry_run=False, simulate=False)
            
            with patch.dict('os.environ', {'FAL_KEY': 'test_key'}):
                video_path = generator.generate_scene_video(sample_scene)
                
                assert video_path is not None
                mock_fal_client.subscribe.assert_called_once()
    
    @patch('src.pipeline.video_gen.fal_client')
    def test_fal_api_failure(self, mock_fal_client, sample_scene):
        """Test fal.ai API failure handling."""
        # Mock API failure
        mock_fal_client.subscribe.side_effect = Exception("API Error")
        
        generator = VideoGenerator(dry_run=False, simulate=False)
        
        with patch.dict('os.environ', {'FAL_KEY': 'test_key'}):
            video_path = generator.generate_scene_video(sample_scene)
            assert video_path is None
    
    def test_missing_fal_key(self, sample_scene):
        """Test handling of missing FAL_KEY environment variable."""
        generator = VideoGenerator(dry_run=False, simulate=False)
        
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(VideoGenerationError, match="FAL_KEY not found"):
                generator._init_fal_client()


class TestConvenienceFunctions:
    """Test cases for convenience functions."""
    
    def test_generate_scene_video_function(self, sample_scene):
        """Test the convenience generate_scene_video function."""
        video_path = generate_scene_video(sample_scene, dry_run=True)
        
        assert video_path is not None
        assert Path(video_path).exists()
        assert "scene_1.mp4" in video_path
    
    def test_generate_scene_video_with_image(self, sample_scene):
        """Test video generation with reference image."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_image:
            temp_image.write(b"fake image data")
            temp_image_path = temp_image.name
        
        try:
            video_path = generate_scene_video(
                sample_scene, 
                image_path=temp_image_path, 
                dry_run=True
            )
            
            assert video_path is not None
            assert Path(video_path).exists()
        finally:
            Path(temp_image_path).unlink(missing_ok=True)


class TestErrorHandling:
    """Test cases for error handling scenarios."""
    
    def test_invalid_scene_data(self):
        """Test handling of invalid scene data."""
        # This should be caught by Pydantic validation in the Scene model
        with pytest.raises(Exception):
            Scene(id="invalid", title="", characters=None)
    
    def test_network_errors(self, video_generator, sample_scene):
        """Test handling of network errors during download."""
        with patch.object(video_generator, '_call_fal_api') as mock_api:
            mock_api.return_value = {
                "video": {"url": "https://invalid-url.com/video.mp4"},
                "duration": 10
            }
            
            with patch.object(video_generator, '_download_video') as mock_download:
                mock_download.return_value = False
                
                video_path = video_generator.generate_scene_video(sample_scene)
                assert video_path is None
    
    def test_empty_api_response(self, video_generator, sample_scene):
        """Test handling of empty API response."""
        with patch.object(video_generator, '_call_fal_api') as mock_api:
            mock_api.return_value = None
            
            video_path = video_generator.generate_scene_video(sample_scene)
            assert video_path is None


class TestFileOperations:
    """Test cases for file operations."""
    
    def test_assets_directory_creation(self):
        """Test that assets directory is created automatically."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_assets = Path(temp_dir) / "test_assets"
            
            # Mock the assets directory path
            with patch('src.pipeline.video_gen.Path') as mock_path:
                mock_path.return_value.parent.parent.resolve.return_value = Path(temp_dir)
                
                generator = VideoGenerator(dry_run=True)
                assert generator.assets_dir.exists()
    
    def test_video_file_naming(self, video_generator, sample_scene):
        """Test video file naming convention."""
        video_path = video_generator.generate_scene_video(sample_scene)
        
        assert video_path is not None
        filename = Path(video_path).name
        assert filename == "scene_1.mp4"
        
        # Test with different scene ID
        scene2 = Scene(**sample_scene.model_dump())
        scene2.id = 42
        video_path2 = video_generator.generate_scene_video(scene2)
        
        filename2 = Path(video_path2).name
        assert filename2 == "scene_42.mp4"


# Integration tests
class TestIntegration:
    """Integration test cases."""
    
    def test_multiple_scene_generation(self, video_generator):
        """Test generating multiple scenes in sequence."""
        scenes = [
            Scene(
                id=i,
                title=f"Scene {i}",
                characters=[f"Character {i}"],
                setting=f"Setting {i}",
                summary=f"Summary {i}",
                tone="dramatic"
            )
            for i in range(1, 4)
        ]
        
        video_paths = []
        for scene in scenes:
            path = video_generator.generate_scene_video(scene)
            video_paths.append(path)
        
        # All should succeed
        assert all(path is not None for path in video_paths)
        assert len(set(video_paths)) == len(video_paths)  # All unique paths
        
        # Check session summary
        summary = video_generator.get_session_summary()
        assert summary["videos_generated"] == 3
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        # Create test scene
        scene = Scene(
            id=1,
            title="Epic Battle",
            characters=["Hero", "Villain"],
            setting="Ancient castle courtyard at sunset",
            summary="The hero confronts the villain in an epic sword fight while storm clouds gather overhead",
            tone="dramatic, intense, cinematic"
        )
        
        # Generate video in dry run mode
        video_path = generate_scene_video(scene, dry_run=True)
        
        # Verify result
        assert video_path is not None
        assert Path(video_path).exists()
        
        # Read the mock file content
        content = Path(video_path).read_text()
        assert "Mock video for scene 1" in content 