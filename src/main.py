"""
AI Scene-to-Video CLI

Main command-line interface for the AI Scene-to-Video pipeline.
Converts story scripts into scene images using OpenAI's GPT-4o and DALL·E 3.
"""

import logging

from pathlib import Path
from typing import List, Optional

import typer
from rich import print as rprint
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from src.pipeline.image_gen import generate_scene_image
from src.pipeline.scene_parser import Scene, parse_scenes
from src.pipeline.video_gen import VideoGenerator

# Initialize CLI app and console
app = typer.Typer(
    name="ai-scene-to-video",
    help="AI-powered scene-to-video generation pipeline",
    rich_markup_mode="rich"
)
console = Console()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_story_from_file(file_path: Path) -> str:
    """
    Load story content from a text file.
    
    Args:
        file_path (Path): Path to the story file
        
    Returns:
        str: Story content
        
    Raises:
        FileNotFoundError: If file doesn't exist
        UnicodeDecodeError: If file can't be decoded as UTF-8
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        if not content:
            raise ValueError("Story file is empty")
            
        logger.info(f"Loaded story from {file_path} ({len(content)} characters)")
        return content
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Story file not found: {file_path}")
    except UnicodeDecodeError as e:
        raise UnicodeDecodeError(
            f"Could not decode file as UTF-8: {file_path}. Error: {e}"
        )


def check_existing_assets(scenes: List[Scene], asset_type: str = "image") -> dict[int, bool]:
    """
    Check which scene assets already exist.
    
    Args:
        scenes (List[Scene]): List of scenes to check
        asset_type (str): Type of asset to check ("image" or "video")
        
    Returns:
        dict[int, bool]: Mapping of scene ID to whether asset exists
    """
    assets_dir = Path(__file__).parent / "assets"
    existing_assets = {}
    
    for scene in scenes:
        if asset_type == "image":
            asset_path = assets_dir / f"scene_{scene.id}.png"
        elif asset_type == "video":
            asset_path = assets_dir / f"scene_{scene.id}.mp4"
        else:
            raise ValueError(f"Invalid asset_type: {asset_type}")
            
        existing_assets[scene.id] = asset_path.exists()
        
        if existing_assets[scene.id]:
            logger.info(f"Found existing {asset_type} asset for scene {scene.id}")
    
    return existing_assets


def create_summary_table(
    scenes: List[Scene], 
    results: dict[int, Optional[str]], 
    existing_assets: dict[int, bool],
    asset_type: str = "image",
    verbose: bool = False
) -> Table:
    """
    Create a rich table summarizing the processing results.
    
    Args:
        scenes (List[Scene]): Processed scenes
        results (dict[int, Optional[str]]): Generation results
        existing_assets (dict[int, bool]): Existing asset status
        asset_type (str): Type of asset ("image" or "video")
        verbose (bool): Whether to include verbose details
        
    Returns:
        Table: Rich table with results
    """
    emoji = "🎬" if asset_type == "video" else "🖼️"
    table = Table(title=f"{emoji} Scene {asset_type.title()} Processing Summary")
    
    table.add_column("Scene", justify="center", style="cyan", no_wrap=True)
    table.add_column("Title", style="magenta", max_width=30)
    table.add_column("Status", justify="center", style="green")
    
    if verbose:
        table.add_column("Characters", style="blue", max_width=20)
        table.add_column("Tone", style="yellow", max_width=12)
    
    for scene in scenes:
        scene_id = scene.id
        
        # Determine status
        if existing_assets.get(scene_id, False):
            status = "⚠️ Skipped (exists)"
            status_style = "yellow"
        elif results.get(scene_id):
            status = "✅ Generated"
            status_style = "green"
        else:
            status = "❌ Failed"
            status_style = "red"
        
        # Prepare row data
        row_data = [
            str(scene_id),
            scene.title[:27] + "..." if len(scene.title) > 30 else scene.title,
            f"[{status_style}]{status}[/{status_style}]"
        ]
        
        if verbose:
            characters = ", ".join(scene.characters[:2])  # Show first 2 characters
            if len(scene.characters) > 2:
                characters += "..."
            row_data.extend([
                characters or "None",
                scene.tone
            ])
        
        table.add_row(*row_data)
    
    return table


@app.command("generate-media")
def generate_media(
    script_path: str = typer.Argument(..., help="Path to the story script (.txt file)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
    skip_existing: bool = typer.Option(True, "--skip-existing/--overwrite", help="Skip existing images"),
    max_scenes: Optional[int] = typer.Option(None, "--max-scenes", help="Maximum number of scenes to process")
):
    """
    Generate images for all scenes in a story script.
    
    Reads a story from a text file, parses it into scenes using GPT-4o,
    and generates images for each scene using DALL·E 3.
    """
    
    # Configure logging level based on verbose flag
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose mode enabled")
    
    try:
        # Validate and load story file
        file_path = Path(script_path)
        if not file_path.exists():
            rprint(f"[red]❌ Error: File not found: {script_path}[/red]")
            raise typer.Exit(1)
        
        if not file_path.suffix.lower() == '.txt':
            rprint(f"[yellow]⚠️ Warning: File doesn't have .txt extension: {script_path}[/yellow]")
        
        rprint(f"[cyan]📖 Loading story from: {script_path}[/cyan]")
        story_content = load_story_from_file(file_path)
        
        # Parse scenes
        rprint("[cyan]🧠 Parsing scenes with GPT-4o...[/cyan]")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Parsing story into scenes...", total=None)
            scenes = parse_scenes(story_content)
            progress.remove_task(task)
        
        if not scenes:
            rprint("[red]❌ No scenes could be parsed from the story.[/red]")
            rprint("[yellow]💡 Please check your OpenAI API key configuration.[/yellow]")
            raise typer.Exit(1)
        
        # Apply max_scenes limit if specified
        if max_scenes and max_scenes > 0:
            original_count = len(scenes)
            scenes = scenes[:max_scenes]
            if len(scenes) < original_count:
                rprint(f"[yellow]📋 Limited to first {max_scenes} scenes (total: {original_count})[/yellow]")
        
        rprint(f"[green]✅ Parsed {len(scenes)} scenes successfully[/green]")
        
        # Convert scene dictionaries to Scene objects
        scene_objects = [Scene(**scene) for scene in scenes]
        
        if verbose:
            for scene in scene_objects:
                rprint(f"  [cyan]Scene {scene.id}:[/cyan] {scene.title}")
        
        # Check existing assets
        existing_assets = check_existing_assets(scene_objects, "image") if skip_existing else {}
        existing_count = sum(existing_assets.values()) if skip_existing else 0
        
        if existing_count > 0:
            rprint(f"[yellow]📁 Found {existing_count} existing images (will skip)[/yellow]")
        
        # Generate images
        rprint(f"[cyan]🎨 Generating images with DALL·E 3...[/cyan]")
        results = {}
        
        with Progress(console=console) as progress:
            task = progress.add_task(
                "Generating scene images...", 
                total=len(scene_objects)
            )
            
            for i, scene in enumerate(scene_objects, 1):
                progress.update(
                    task, 
                    description=f"Processing Scene {scene.id}: {scene.title[:30]}..."
                )
                
                # Skip if image already exists
                if skip_existing and existing_assets.get(scene.id, False):
                    rprint(f"[yellow]🖼️  Scene {scene.id}: \"{scene.title}\" → ⚠️ skipped (already exists)[/yellow]")
                    results[scene.id] = "skipped"
                    progress.advance(task)
                    continue
                
                # Generate image
                try:
                    image_path = generate_scene_image(scene)
                    if image_path:
                        results[scene.id] = image_path
                        rprint(f"[green]🖼️  Scene {scene.id}: \"{scene.title}\" → ✅ image saved[/green]")
                        if verbose:
                            rprint(f"     [dim]Saved to: {image_path}[/dim]")
                    else:
                        results[scene.id] = None
                        rprint(f"[red]🖼️  Scene {scene.id}: \"{scene.title}\" → ❌ generation failed[/red]")
                        
                except Exception as e:
                    results[scene.id] = None
                    rprint(f"[red]🖼️  Scene {scene.id}: \"{scene.title}\" → ❌ error: {str(e)[:50]}[/red]")
                    if verbose:
                        logger.error(f"Error generating image for scene {scene.id}: {e}")
                
                progress.advance(task)
        
        # Generate summary
        successful_generations = sum(1 for result in results.values() if result and result != "skipped")
        skipped_count = sum(1 for result in results.values() if result == "skipped")
        failed_count = sum(1 for result in results.values() if not result)
        
        rprint("\n" + "="*60)
        rprint("[bold green]✅ Processing Complete![/bold green]")
        
        # Display summary table
        summary_table = create_summary_table(scene_objects, results, existing_assets, "image", verbose)
        console.print(summary_table)
        
        # Display statistics
        rprint(f"\n[cyan]📊 Statistics:[/cyan]")
        rprint(f"  • Total scenes: [bold]{len(scenes)}[/bold]")
        rprint(f"  • Generated: [green]{successful_generations}[/green]")
        if skipped_count > 0:
            rprint(f"  • Skipped: [yellow]{skipped_count}[/yellow]")
        if failed_count > 0:
            rprint(f"  • Failed: [red]{failed_count}[/red]")
        
        # Show assets directory
        assets_dir = Path(__file__).parent / "assets"
        rprint(f"\n[cyan]📁 Assets saved to:[/cyan] {assets_dir.absolute()}")
        
        if failed_count > 0:
            rprint(f"\n[yellow]💡 Tip: Check your OpenAI API key and rate limits if generations failed.[/yellow]")
            raise typer.Exit(1)
        
    except FileNotFoundError as e:
        rprint(f"[red]❌ File Error: {e}[/red]")
        raise typer.Exit(1)
    except UnicodeDecodeError as e:
        rprint(f"[red]❌ Encoding Error: {e}[/red]")
        rprint("[yellow]💡 Tip: Ensure your file is saved with UTF-8 encoding.[/yellow]")
        raise typer.Exit(1)
    except Exception as e:
        rprint(f"[red]❌ Unexpected Error: {e}[/red]")
        if verbose:
            logger.exception("Unexpected error in generate-media command")
        raise typer.Exit(1)


@app.command("generate-videos")
def generate_videos(
    script_path: str = typer.Argument(..., help="Path to the story script (.txt file)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
    skip_existing: bool = typer.Option(True, "--skip-existing/--overwrite", help="Skip existing videos"),
    max_scenes: Optional[int] = typer.Option(None, "--max-scenes", help="Maximum number of scenes to process"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Test mode - no actual API calls or costs"),
    simulate: bool = typer.Option(False, "--simulate", help="Simulate API calls with delays but no costs"),
    budget_limit: float = typer.Option(50.0, "--budget", help="Maximum budget for video generation (USD)"),
    use_images: bool = typer.Option(False, "--use-images", help="Use existing scene images as references")
):
    """
    Generate videos for all scenes in a story script using fal.ai Veo 3.
    
    This is a high-cost operation. Use --dry-run for testing without costs.
    """
    
    # Configure logging level based on verbose flag
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose mode enabled")
    
    # Display cost warning unless in test mode
    if not dry_run and not simulate:
        rprint("[yellow]⚠️  WARNING: Video generation incurs real costs (~$5 per 10s clip)[/yellow]")
        rprint(f"[yellow]💰 Budget limit: ${budget_limit:.2f}[/yellow]")
        confirm = typer.confirm("Continue with video generation?")
        if not confirm:
            rprint("[yellow]Operation cancelled by user[/yellow]")
            raise typer.Exit(0)
    
    try:
        # Validate and load story file
        file_path = Path(script_path)
        if not file_path.exists():
            rprint(f"[red]❌ Error: File not found: {script_path}[/red]")
            raise typer.Exit(1)
        
        rprint(f"[cyan]📖 Loading story from: {script_path}[/cyan]")
        story_content = load_story_from_file(file_path)
        
        # Parse scenes
        rprint("[cyan]🧠 Parsing scenes with GPT-4o...[/cyan]")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Parsing story into scenes...", total=None)
            scenes = parse_scenes(story_content)
            progress.remove_task(task)
        
        if not scenes:
            rprint("[red]❌ No scenes could be parsed from the story.[/red]")
            raise typer.Exit(1)
        
        # Convert scene dictionaries to Scene objects
        scene_objects = [Scene(**scene) for scene in scenes]
        
        # Apply max_scenes limit if specified
        if max_scenes and max_scenes > 0:
            original_count = len(scene_objects)
            scene_objects = scene_objects[:max_scenes]
            if len(scene_objects) < original_count:
                rprint(f"[yellow]📋 Limited to first {max_scenes} scenes (total: {original_count})[/yellow]")
        
        rprint(f"[green]✅ Parsed {len(scene_objects)} scenes successfully[/green]")
        
        # Check existing video assets
        existing_videos = check_existing_assets(scene_objects, "video") if skip_existing else {}
        existing_count = sum(existing_videos.values()) if skip_existing else 0
        
        if existing_count > 0:
            rprint(f"[yellow]🎬 Found {existing_count} existing videos (will skip)[/yellow]")
        
        # Check for existing images if using as references
        existing_images = {}
        if use_images:
            existing_images = check_existing_assets(scene_objects, "image")
            image_count = sum(existing_images.values())
            rprint(f"[cyan]🖼️  Found {image_count} existing images to use as references[/cyan]")
        
        # Initialize video generator
        video_generator = VideoGenerator(
            dry_run=dry_run,
            simulate=simulate,
            budget_limit=budget_limit
        )
        
        # Generate videos
        mode_text = "DRY RUN" if dry_run else "SIMULATE" if simulate else "PRODUCTION"
        rprint(f"[cyan]🎬 Generating videos with fal.ai Veo 3 ({mode_text} mode)...[/cyan]")
        
        results = {}
        
        with Progress(console=console) as progress:
            task = progress.add_task(
                "Generating scene videos...", 
                total=len(scene_objects)
            )
            
            for i, scene in enumerate(scene_objects, 1):
                progress.update(
                    task, 
                    description=f"Processing Scene {scene.id}: {scene.title[:30]}..."
                )
                
                # Skip if video already exists
                if skip_existing and existing_videos.get(scene.id, False):
                    rprint(f"[yellow]🎬 Scene {scene.id}: \"{scene.title}\" → ⚠️ skipped (already exists)[/yellow]")
                    results[scene.id] = "skipped"
                    progress.advance(task)
                    continue
                
                # Check budget before generation
                session_summary = video_generator.get_session_summary()
                if session_summary["budget_remaining"] <= 0:
                    rprint(f"[red]💰 Budget exhausted after {session_summary['videos_generated']} videos[/red]")
                    break
                
                # Get reference image path if using images
                image_path = None
                if use_images and existing_images.get(scene.id, False):
                    assets_dir = Path(__file__).parent / "assets"
                    image_path = str(assets_dir / f"scene_{scene.id}.png")
                
                # Generate video
                try:
                    video_path = video_generator.generate_scene_video(scene, image_path)
                    if video_path:
                        results[scene.id] = video_path
                        cost_info = f"${session_summary['total_cost']:.2f}" if not dry_run and not simulate else ""
                        rprint(f"[green]🎬 Scene {scene.id}: \"{scene.title}\" → ✅ video saved {cost_info}[/green]")
                        if verbose:
                            rprint(f"     [dim]Saved to: {video_path}[/dim]")
                    else:
                        results[scene.id] = None
                        rprint(f"[red]🎬 Scene {scene.id}: \"{scene.title}\" → ❌ generation failed[/red]")
                        
                except Exception as e:
                    results[scene.id] = None
                    rprint(f"[red]🎬 Scene {scene.id}: \"{scene.title}\" → ❌ error: {str(e)[:50]}[/red]")
                    if verbose:
                        logger.error(f"Error generating video for scene {scene.id}: {e}")
                
                progress.advance(task)
        
        # Get final session summary
        final_summary = video_generator.get_session_summary()
        
        # Generate summary
        successful_generations = sum(1 for result in results.values() if result and result != "skipped")
        skipped_count = sum(1 for result in results.values() if result == "skipped")
        failed_count = sum(1 for result in results.values() if not result)
        
        rprint("\n" + "="*60)
        rprint("[bold green]✅ Video Generation Complete![/bold green]")
        
        # Display summary table
        summary_table = create_summary_table(scene_objects, results, existing_videos, "video", verbose)
        console.print(summary_table)
        
        # Display statistics
        rprint(f"\n[cyan]📊 Statistics:[/cyan]")
        rprint(f"  • Total scenes: [bold]{len(scenes)}[/bold]")
        rprint(f"  • Generated: [green]{successful_generations}[/green]")
        if skipped_count > 0:
            rprint(f"  • Skipped: [yellow]{skipped_count}[/yellow]")
        if failed_count > 0:
            rprint(f"  • Failed: [red]{failed_count}[/red]")
        
        # Display cost information
        if not dry_run:
            rprint(f"\n[cyan]💰 Cost Summary:[/cyan]")
            rprint(f"  • Total cost: [bold]${final_summary['total_cost']:.2f}[/bold]")
            rprint(f"  • Budget remaining: [green]${final_summary['budget_remaining']:.2f}[/green]")
            if final_summary['videos']:
                avg_cost = final_summary['total_cost'] / len(final_summary['videos'])
                rprint(f"  • Average per video: [cyan]${avg_cost:.2f}[/cyan]")
        
        # Show assets directory
        assets_dir = Path(__file__).parent / "assets"
        rprint(f"\n[cyan]📁 Videos saved to:[/cyan] {assets_dir.absolute()}")
        
        if failed_count > 0:
            rprint(f"\n[yellow]💡 Tip: Check your FAL_KEY and rate limits if generations failed.[/yellow]")
            if not dry_run and not simulate:
                raise typer.Exit(1)
        
    except Exception as e:
        rprint(f"[red]❌ Unexpected Error: {e}[/red]")
        if verbose:
            logger.exception("Unexpected error in generate-videos command")
        raise typer.Exit(1)


@app.command("generate-all")
def generate_all(
    script_path: str = typer.Argument(..., help="Path to the story script (.txt file)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
    skip_existing: bool = typer.Option(True, "--skip-existing/--overwrite", help="Skip existing assets"),
    max_scenes: Optional[int] = typer.Option(None, "--max-scenes", help="Maximum number of scenes to process"),
    dry_run_videos: bool = typer.Option(False, "--dry-run-videos", help="Test video generation without costs"),
    budget_limit: float = typer.Option(50.0, "--budget", help="Maximum budget for video generation (USD)")
):
    """
    Generate both images and videos for all scenes in a story script.
    
    This runs the complete pipeline: scene parsing → image generation → video generation.
    """
    
    rprint("[bold cyan]🚀 Starting Complete AI Scene-to-Video Pipeline[/bold cyan]")
    
    # Configure logging level based on verbose flag
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Validate file
        file_path = Path(script_path)
        if not file_path.exists():
            rprint(f"[red]❌ Error: File not found: {script_path}[/red]")
            raise typer.Exit(1)
        
        # Step 1: Generate images
        rprint("[cyan]🎨 Step 1: Generating scene images...[/cyan]")
        try:
            from typer.testing import CliRunner
            # Test runner not used in final implementation
            
            # Build arguments for generate-media command
            args = [script_path]
            if verbose:
                args.append("--verbose")
            if skip_existing:
                args.append("--skip-existing")
            if max_scenes:
                args.extend(["--max-scenes", str(max_scenes)])
            
            # Call generate-media command internally
            # For now, we'll use a simpler approach by calling the logic directly
            story_content = load_story_from_file(file_path)
            scenes = parse_scenes(story_content)
            
            if not scenes:
                rprint("[red]❌ No scenes could be parsed from the story.[/red]")
                raise typer.Exit(1)
            
            # Convert scene dictionaries to Scene objects
            scene_objects = [Scene(**scene) for scene in scenes]
            
            if max_scenes and max_scenes > 0:
                scene_objects = scene_objects[:max_scenes]
            
            # Generate images
            existing_images = check_existing_assets(scene_objects, "image") if skip_existing else {}
            image_results = {}
            
            rprint(f"[cyan]🖼️  Generating {len(scene_objects)} scene images...[/cyan]")
            for scene in scene_objects:
                if skip_existing and existing_images.get(scene.id, False):
                    image_results[scene.id] = "skipped"
                    continue
                
                try:
                    image_path = generate_scene_image(scene)
                    image_results[scene.id] = image_path
                    if image_path:
                        rprint(f"[green]✅ Image generated for scene {scene.id}[/green]")
                except Exception as e:
                    image_results[scene.id] = None
                    rprint(f"[red]❌ Image failed for scene {scene.id}: {e}[/red]")
            
            successful_images = sum(1 for r in image_results.values() if r and r != "skipped")
            rprint(f"[green]✅ Images complete: {successful_images}/{len(scene_objects)} generated[/green]")
            
        except Exception as e:
            rprint(f"[red]❌ Image generation failed: {e}[/red]")
            if not typer.confirm("Continue with video generation despite image failures?"):
                raise typer.Exit(1)
        
        # Step 2: Generate videos
        rprint("\n[cyan]🎬 Step 2: Generating scene videos...[/cyan]")
        
        if not dry_run_videos:
            rprint("[yellow]⚠️  WARNING: Video generation incurs real costs (~$5 per 10s clip)[/yellow]")
            rprint(f"[yellow]💰 Budget limit: ${budget_limit:.2f}[/yellow]")
            confirm = typer.confirm("Continue with video generation?")
            if not confirm:
                rprint("[yellow]Skipping video generation[/yellow]")
                raise typer.Exit(0)
        
        # Initialize video generator
        video_generator = VideoGenerator(
            dry_run=dry_run_videos,
            simulate=False,
            budget_limit=budget_limit
        )
        
        # Check existing videos
        existing_videos = check_existing_assets(scene_objects, "video") if skip_existing else {}
        video_results = {}
        
        rprint(f"[cyan]🎬 Generating {len(scene_objects)} scene videos...[/cyan]")
        for scene in scene_objects:
            if skip_existing and existing_videos.get(scene.id, False):
                video_results[scene.id] = "skipped"
                continue
            
            # Use generated image as reference if available
            image_path = None
            if image_results.get(scene.id) and image_results[scene.id] != "skipped":
                image_path = image_results[scene.id]
            
            try:
                video_path = video_generator.generate_scene_video(scene, image_path)
                video_results[scene.id] = video_path
                if video_path:
                    rprint(f"[green]✅ Video generated for scene {scene.id}[/green]")
            except Exception as e:
                video_results[scene.id] = None
                rprint(f"[red]❌ Video failed for scene {scene.id}: {e}[/red]")
        
        # Final summary
        successful_videos = sum(1 for r in video_results.values() if r and r != "skipped")
        final_summary = video_generator.get_session_summary()
        
        rprint("\n" + "="*60)
        rprint("[bold green]🎉 Complete Pipeline Finished![/bold green]")
        rprint(f"[cyan]📊 Final Results:[/cyan]")
        rprint(f"  • Total scenes: [bold]{len(scenes)}[/bold]")
        rprint(f"  • Images generated: [green]{successful_images}[/green]")
        rprint(f"  • Videos generated: [green]{successful_videos}[/green]")
        
        if not dry_run_videos:
            rprint(f"  • Total video cost: [bold]${final_summary['total_cost']:.2f}[/bold]")
        
        assets_dir = Path(__file__).parent / "assets"
        rprint(f"\n[cyan]📁 All assets saved to:[/cyan] {assets_dir.absolute()}")
        
    except Exception as e:
        rprint(f"[red]❌ Pipeline error: {e}[/red]")
        if verbose:
            logger.exception("Unexpected error in generate-all command")
        raise typer.Exit(1)


@app.command("test")
def test_pipeline():
    """Test the complete pipeline with a sample story."""
    
    rprint("[cyan]🧪 Testing AI Scene-to-Video Pipeline[/cyan]")
    
    # Sample story for testing
    sample_story = """
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
    
    rprint("[cyan]📖 Using sample story for testing...[/cyan]")
    
    try:
        # Test scene parsing
        rprint("[cyan]🧠 Testing scene parsing...[/cyan]")
        scenes = parse_scenes(sample_story)
        
        if scenes:
            rprint(f"[green]✅ Successfully parsed {len(scenes)} scenes[/green]")
            for scene in scenes:
                rprint(f"  [cyan]Scene {scene.id}:[/cyan] {scene.title}")
                
            # Test video generation (dry run)
            rprint("\n[cyan]🎬 Testing video generation (dry run mode)...[/cyan]")
            
            video_generator = VideoGenerator(dry_run=True, budget_limit=10.0)
            
            for scene in scenes[:2]:  # Test first 2 scenes only
                try:
                    video_path = video_generator.generate_scene_video(scene)
                    if video_path:
                        rprint(f"[green]✅ Mock video generated for scene {scene.id}[/green]")
                        rprint(f"     [dim]Path: {video_path}[/dim]")
                    else:
                        rprint(f"[yellow]⚠️ Video generation returned None for scene {scene.id}[/yellow]")
                except Exception as e:
                    rprint(f"[red]❌ Video generation error for scene {scene.id}: {e}[/red]")
            
            # Show session summary
            summary = video_generator.get_session_summary()
            rprint(f"\n[cyan]📊 Video Generation Test Summary:[/cyan]")
            rprint(f"  • Videos generated: {summary['videos_generated']}")
            rprint(f"  • Mock total cost: ${summary['total_cost']:.2f}")
            rprint(f"  • Budget remaining: ${summary['budget_remaining']:.2f}")
            
        else:
            rprint("[yellow]⚠️ Scene parsing returned empty results (API key needed)[/yellow]")
            rprint("[cyan]🎬 Testing video generation with mock scene...[/cyan]")
            
            # Create mock scene for video testing
            mock_scene = Scene(
                id=1,
                title="Test Scene",
                characters=["Test Character"],
                setting="Test setting for demonstration",
                summary="This is a test scene for pipeline validation",
                tone="test, demonstration"
            )
            
            video_generator = VideoGenerator(dry_run=True)
            video_path = video_generator.generate_scene_video(mock_scene)
            
            if video_path:
                rprint(f"[green]✅ Mock video generation successful[/green]")
            else:
                rprint(f"[yellow]⚠️ Mock video generation failed[/yellow]")
        
        rprint(f"\n[green]✅ Pipeline test complete![/green]")
        rprint("[cyan]💡 Ready for production use with proper API keys![/cyan]")
        
    except Exception as e:
        rprint(f"[red]❌ Pipeline test failed: {e}[/red]")
        if "OPENAI_API_KEY" in str(e):
            rprint("[yellow]💡 Tip: Set your OPENAI_API_KEY environment variable[/yellow]")
        if "FAL_KEY" in str(e):
            rprint("[yellow]💡 Tip: Set your FAL_KEY environment variable for video generation[/yellow]")
        raise typer.Exit(1)


@app.command("version")
def show_version():
    """Show version information."""
    rprint("[cyan]🎬 AI Scene-to-Video Pipeline[/cyan]")
    rprint("Version: 0.1.0")
    rprint("Built with:")
    rprint("  • OpenAI GPT-4o (scene parsing)")
    rprint("  • DALL·E 3 (image generation)")  
    rprint("  • fal.ai Veo 3 (video generation)")
    rprint("")
    rprint("[cyan]Available Commands:[/cyan]")
    rprint("  • generate-media    - Generate scene images")
    rprint("  • generate-videos   - Generate scene videos")
    rprint("  • generate-all      - Complete pipeline (images + videos)")
    rprint("  • test              - Test pipeline functionality")


if __name__ == "__main__":
    app()
