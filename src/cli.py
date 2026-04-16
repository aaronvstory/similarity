import os
import sys
import tkinter as tk
from tkinter import filedialog
from typing import Optional, List, Tuple
import glob
import re
import shutil

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.prompt import Prompt, Confirm
from rich.table import Table

from src.engine import FaceEngine

console = Console()

class ProCLI:
    """
    Command Line Interface for the Face Similarity Application.
    Uses 'rich' for elegant and professional terminal outputs.
    """

    def __init__(self):
        self.engine = FaceEngine()
        self.img1_keyword = "extracted"
        self.img2_keyword = "selfie"
        self.models_initialized = False

    def _ensure_models_initialized(self):
        if not self.models_initialized:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
                console=console
            ) as progress:
                progress.add_task("[cyan]Initializing ML Models (ArcFace & RetinaFace)...", total=None)
                try:
                    self.engine.initialize_models()
                    self.models_initialized = True
                except Exception as e:
                    progress.stop()
                    console.print(f"[bold red]Initialization Error:[/bold red] {e}")
                    sys.exit(1)

    def prompt_for_file(self, title: str) -> Optional[str]:
        """
        Opens a native OS file dialog to select an image file.
        """
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        file_path = filedialog.askopenfilename(
            title=title,
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.webp")]
        )
        root.destroy()
        return file_path if file_path else None

    def prompt_for_directory(self, title: str) -> Optional[str]:
        """
        Opens a native OS file dialog to select a directory.
        """
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        dir_path = filedialog.askdirectory(title=title)
        root.destroy()
        return dir_path if dir_path else None

    def run(self, img1_path: Optional[str] = None, img2_path: Optional[str] = None) -> None:
        """
        Executes the CLI workflow. If paths are provided, runs single comparison.
        Otherwise, opens the interactive menu.
        """
        console.print(Panel.fit("[bold cyan]Face Similarity Pro CLI[/bold cyan]", border_style="cyan"))

        if img1_path or img2_path:
            self._ensure_models_initialized()
            self._run_single_comparison(img1_path, img2_path)
            return

        while True:
            console.print("\n[bold]Main Menu[/bold]")
            console.print("1. Single Comparison")
            console.print("2. Batch Folder Processing")
            console.print("3. Settings")
            console.print("4. Exit")
            
            choice = Prompt.ask("Choose an option", choices=["1", "2", "3", "4"], default="1")
            
            if choice == "1":
                self._run_single_comparison()
            elif choice == "2":
                self._run_batch_processing()
            elif choice == "3":
                self._run_settings()
            elif choice == "4":
                console.print("[green]Exiting...[/green]")
                break

    def _run_settings(self):
        console.print("\n[bold magenta]--- Settings ---[/bold magenta]")
        console.print(f"Current Image 1 Regex/Keyword: [green]{self.img1_keyword}[/green]")
        console.print(f"Current Image 2 Regex/Keyword: [green]{self.img2_keyword}[/green]")
        
        self.img1_keyword = Prompt.ask("Enter new keyword/regex for Image 1", default=self.img1_keyword)
        self.img2_keyword = Prompt.ask("Enter new keyword/regex for Image 2", default=self.img2_keyword)
        console.print("[bold green]Settings updated successfully![/bold green]")

    def _run_single_comparison(self, img1_path: Optional[str] = None, img2_path: Optional[str] = None) -> None:
        if not img1_path:
            console.print("[yellow]Please select the first image...[/yellow]")
            img1_path = self.prompt_for_file("Select First Image")
            if not img1_path:
                console.print("[bold red]Action cancelled. No first image selected.[/bold red]")
                return
            console.print(f"Selected: [green]{img1_path}[/green]")

        if not img2_path:
            console.print("[yellow]Please select the second image...[/yellow]")
            img2_path = self.prompt_for_file("Select Second Image")
            if not img2_path:
                console.print("[bold red]Action cancelled. No second image selected.[/bold red]")
                return
            console.print(f"Selected: [green]{img2_path}[/green]\n")

        self._ensure_models_initialized()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
            console=console
        ) as progress:
            progress.add_task("[magenta]Running face detection and similarity comparison...", total=None)
            result = self.engine.compare_images(img1_path, img2_path)

        self._display_result(result)

    def _find_image_with_keyword(self, folder_path: str, keyword: str) -> Optional[str]:
        """Finds the first image in the folder whose filename matches the keyword regex."""
        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
        try:
            pattern = re.compile(keyword, re.IGNORECASE)
        except re.error:
            # Fallback to simple substring match if invalid regex
            pattern = None

        for file in os.listdir(folder_path):
            if file.lower().endswith(valid_extensions):
                if pattern and pattern.search(file):
                    return os.path.join(folder_path, file)
                elif not pattern and keyword.lower() in file.lower():
                    return os.path.join(folder_path, file)
        return None

    def _get_new_folder_name(self, old_folder_path: str, score: float) -> str:
        """
        Generates a new folder name by inserting the rounded score.
        e.g., "FAILED PERSONA - Morgan" -> "FAILED PERSONA 81 - Morgan"
        """
        parent_dir = os.path.dirname(old_folder_path)
        folder_name = os.path.basename(old_folder_path)
        rounded_score = int(score)
        
        # Check if it already has a score like " 81 - "
        if re.search(r'\s\d{1,3}\s-\s', folder_name):
            # Already scored, skip renaming
            return old_folder_path

        if " - " in folder_name:
            parts = folder_name.split(" - ", 1)
            new_name = f"{parts[0]} {rounded_score} - {parts[1]}"
        else:
            new_name = f"{folder_name} {rounded_score}"
            
        return os.path.join(parent_dir, new_name)

    def _run_batch_processing(self):
        console.print("[yellow]Please select the root directory for batch processing...[/yellow]")
        root_dir = self.prompt_for_directory("Select Root Directory")
        if not root_dir:
            console.print("[bold red]Action cancelled. No directory selected.[/bold red]")
            return

        console.print(f"Scanning directory recursively: [green]{root_dir}[/green]")
        
        # Find all subdirectories that might contain the images
        folders_to_process = []
        for dirpath, dirnames, filenames in os.walk(root_dir):
            img1 = self._find_image_with_keyword(dirpath, self.img1_keyword)
            img2 = self._find_image_with_keyword(dirpath, self.img2_keyword)
            if img1 and img2:
                folders_to_process.append((dirpath, img1, img2))

        if not folders_to_process:
            console.print(f"[yellow]No folders found containing both image keywords ('{self.img1_keyword}', '{self.img2_keyword}').[/yellow]")
            return

        console.print(f"Found [bold]{len(folders_to_process)}[/bold] folders to process.")
        if not Confirm.ask("Do you want to proceed with batch processing?"):
            return

        self._ensure_models_initialized()

        results_table = Table(title="Batch Processing Results")
        results_table.add_column("Folder", style="cyan", no_wrap=False)
        results_table.add_column("Score", justify="right", style="magenta")
        results_table.add_column("Match", justify="center")
        results_table.add_column("Status/Error", style="red")

        successful_renames = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            task = progress.add_task("[magenta]Processing folders...", total=len(folders_to_process))
            
            for folder, img1, img2 in folders_to_process:
                folder_name = os.path.basename(folder)
                progress.update(task, description=f"[magenta]Processing: {folder_name}")
                
                result = self.engine.compare_images(img1, img2)
                
                if result.get("error"):
                    results_table.add_row(folder_name, "-", "N/A", result["error"])
                else:
                    score = result["score"]
                    is_match = result["match"]
                    match_text = "[green]YES[/green]" if is_match else "[red]NO[/red]"
                    
                    # Rename the folder
                    try:
                        new_folder_path = self._get_new_folder_name(folder, score)
                        if new_folder_path != folder:
                            os.rename(folder, new_folder_path)
                            new_folder_name = os.path.basename(new_folder_path)
                            results_table.add_row(f"{folder_name}\n-> {new_folder_name}", f"{score}%", match_text, "[green]Renamed[/green]")
                            successful_renames += 1
                        else:
                            results_table.add_row(folder_name, f"{score}%", match_text, "[yellow]Already named[/yellow]")
                    except Exception as e:
                        results_table.add_row(folder_name, f"{score}%", match_text, f"[red]Rename failed: {e}[/red]")
                
                progress.advance(task)

        console.print(results_table)
        console.print(f"\n[bold green]Batch processing complete. Successfully renamed {successful_renames} folders.[/bold green]")


    def _display_result(self, result: dict) -> None:
        """
        Formats and prints the final outcome to the terminal.
        """
        if result.get("error"):
            console.print(Panel(f"[bold red]Error:[/bold red] {result['error']}", border_style="red"))
            return

        score = result["score"]
        is_match = result["match"]

        if is_match:
            status_text = "are"
            color = "green"
        else:
            status_text = "are not"
            color = "red"

        # The strict output structure required:
        # "Face similarity ratio: [X]%, The two photos [are / are not] the same person."
        output_msg = (
            f"Face similarity ratio: [{color} bold]{score}%[/{color} bold], "
            f"The two photos [{color} bold]{status_text}[/{color} bold] the same person."
        )

        console.print(Panel(output_msg, border_style=color, expand=False))

if __name__ == "__main__":
    cli = ProCLI()
    cli.run()
