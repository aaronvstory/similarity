import os
import sys
import tkinter as tk
from tkinter import filedialog
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.engine import FaceEngine

console = Console()

class ProCLI:
    """
    Command Line Interface for the Face Similarity Application.
    Uses 'rich' for elegant and professional terminal outputs.
    """

    def __init__(self):
        self.engine = FaceEngine()

    def prompt_for_file(self, title: str) -> Optional[str]:
        """
        Opens a native OS file dialog to select an image file.
        """
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        file_path = filedialog.askopenfilename(
            title=title,
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.webp")]
        )
        root.destroy()
        return file_path if file_path else None

    def run(self, img1_path: Optional[str] = None, img2_path: Optional[str] = None) -> None:
        """
        Executes the CLI workflow.
        """
        console.print(Panel.fit("[bold cyan]Face Similarity Pro CLI[/bold cyan]", border_style="cyan"))

        # 1. Gather file paths
        if not img1_path:
            console.print("[yellow]Please select the first image...[/yellow]")
            img1_path = self.prompt_for_file("Select First Image")
            if not img1_path:
                console.print("[bold red]Action cancelled. No first image selected.[/bold red]")
                sys.exit(1)
            console.print(f"Selected: [green]{img1_path}[/green]")

        if not img2_path:
            console.print("[yellow]Please select the second image...[/yellow]")
            img2_path = self.prompt_for_file("Select Second Image")
            if not img2_path:
                console.print("[bold red]Action cancelled. No second image selected.[/bold red]")
                sys.exit(1)
            console.print(f"Selected: [green]{img2_path}[/green]\n")

        # 2. Initialize Models with Progress
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
            console=console
        ) as progress:
            task = progress.add_task("[cyan]Initializing ML Models (ArcFace & RetinaFace)...", total=None)
            try:
                self.engine.initialize_models()
            except Exception as e:
                progress.stop()
                console.print(f"[bold red]Initialization Error:[/bold red] {e}")
                sys.exit(1)

        # 3. Process Comparison with Progress
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
            console=console
        ) as progress:
            task = progress.add_task("[magenta]Running face detection and similarity comparison...", total=None)
            result = self.engine.compare_images(img1_path, img2_path)

        # 4. Display Results
        self._display_result(result)

    def _display_result(self, result: dict) -> None:
        """
        Formats and prints the final outcome to the terminal.
        """
        if result.get("error"):
            console.print(Panel(f"[bold red]Error:[/bold red] {result['error']}", border_style="red"))
            sys.exit(1)

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
