import os
import sys
import tkinter as tk
from tkinter import filedialog
from typing import Optional, List, Dict, Any
from difflib import SequenceMatcher
import re
import json
from datetime import datetime

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
        self.config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json")
        
        # Default Settings
        self.config = {
            "img1_keyword": "extracted",
            "img2_keyword": "selfie",
            "extraction_keyword": "front",
            "padding_ratio": 0.175,
            "existing_file_mode": "index"  # index, skip, overwrite
        }
        
        self.load_config()
        self.models_initialized = False

    def load_config(self):
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r") as f:
                    saved_config = json.load(f)
                    self.config.update(saved_config)
            except Exception as e:
                console.print(f"[yellow]Warning: Could not load config.json ({e}). Using defaults.[/yellow]")

    def save_config(self):
        try:
            with open(self.config_path, "w") as f:
                json.dump(self.config, f, indent=4)
        except Exception as e:
            console.print(f"[red]Error: Could not save config.json ({e}).[/red]")

    def _display_current_settings(self):
        table = Table(title="Current Active Settings", box=None, show_header=False, padding=(0, 2))
        table.add_column("Key", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Similarity Img 1 Keyword", self.config["img1_keyword"])
        table.add_row("Similarity Img 2 Keyword", self.config["img2_keyword"])
        table.add_row("Extraction Keyword", self.config["extraction_keyword"])
        table.add_row("Extraction Padding", f"{self.config['padding_ratio']:.3f}")
        table.add_row("Existing File Mode", self.config["existing_file_mode"])
        
        console.print(Panel(table, border_style="magenta", expand=False))

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
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        dir_path = filedialog.askdirectory(title=title)
        root.destroy()
        return dir_path if dir_path else None

    def run(self, img1_path: Optional[str] = None, img2_path: Optional[str] = None) -> None:
        console.print(Panel.fit("[bold cyan]Face Similarity Pro CLI[/bold cyan]", border_style="cyan"))
        self._display_current_settings()

        if img1_path or img2_path:
            self._ensure_models_initialized()
            self._run_single_comparison(img1_path, img2_path)
            return

        while True:
            console.print("\n[bold]Main Menu[/bold]")
            console.print("1. Single Similarity Comparison")
            console.print("2. Batch Folder Similarity Check")
            console.print("3. Single Face Extraction")
            console.print("4. Batch Face Extraction")
            console.print("5. Settings")
            console.print("6. Exit")
            
            choice = Prompt.ask("Choose an option", choices=["1", "2", "3", "4", "5", "6"], default="1")
            
            if choice == "1":
                self._run_single_comparison()
            elif choice == "2":
                self._run_batch_processing()
            elif choice == "3":
                self._run_single_extraction()
            elif choice == "4":
                self._run_batch_extraction()
            elif choice == "5":
                self._run_settings()
            elif choice == "6":
                console.print("[green]Exiting...[/green]")
                break

    def _run_settings(self):
        console.print("\n[bold magenta]--- Settings ---[/bold magenta]")
        
        self.config["img1_keyword"] = Prompt.ask("Enter keyword/regex for Similarity Img 1", default=self.config["img1_keyword"])
        self.config["img2_keyword"] = Prompt.ask("Enter keyword/regex for Similarity Img 2", default=self.config["img2_keyword"])
        self.config["extraction_keyword"] = Prompt.ask("Enter keyword/regex for Extraction", default=self.config["extraction_keyword"])
        
        while True:
            try:
                val = Prompt.ask("Enter Extraction Padding Ratio (e.g., 0.175)", default=str(self.config["padding_ratio"]))
                self.config["padding_ratio"] = float(val)
                break
            except ValueError:
                console.print("[red]Invalid number. Please enter a float.[/red]")

        self.config["existing_file_mode"] = Prompt.ask(
            "Mode for existing extracted files", 
            choices=["index", "skip", "overwrite"], 
            default=self.config["existing_file_mode"]
        )
        
        self.save_config()
        console.print("[bold green]Settings updated and saved successfully![/bold green]")
        self._display_current_settings()

    def _run_single_comparison(self, img1_path: Optional[str] = None, img2_path: Optional[str] = None) -> None:
        if not img1_path:
            console.print("[yellow]Please select the first image...[/yellow]")
            img1_path = self.prompt_for_file("Select First Image")
            if not img1_path: return
        if not img2_path:
            console.print("[yellow]Please select the second image...[/yellow]")
            img2_path = self.prompt_for_file("Select Second Image")
            if not img2_path: return

        self._ensure_models_initialized()
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True, console=console) as progress:
            progress.add_task("[magenta]Running similarity comparison...", total=None)
            result = self.engine.compare_images(img1_path, img2_path)
        
        self._display_result(result)
        
        # Log to manifest in the folder of the first image
        folder = os.path.dirname(img1_path)
        self._log_to_manifest(folder, "single_similarity", [{
            "img1": os.path.basename(img1_path),
            "img2": os.path.basename(img2_path),
            "score": result.get("score"),
            "match": result.get("match"),
            "error": result.get("error")
        }])

    def _run_single_extraction(self):
        console.print("[yellow]Please select the source image for face extraction...[/yellow]")
        img_path = self.prompt_for_file("Select Image for Extraction")
        if not img_path: return

        dirpath = os.path.dirname(img_path)
        filename = os.path.basename(img_path)
        out_path = self._get_available_path(dirpath, filename)
        
        if not out_path:
            console.print("[yellow]File exists and mode is set to 'skip'. Action cancelled.[/yellow]")
            return

        self._ensure_models_initialized()
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True, console=console) as progress:
            progress.add_task("[magenta]Extracting face...", total=None)
            try:
                conf = self.engine.extract_face(img_path, out_path, padding=self.config["padding_ratio"])
                console.print(f"[bold green]Success![/bold green] Extracted face to [cyan]{os.path.basename(out_path)}[/cyan] (Confidence: {conf:.1%})")
                self._log_to_manifest(dirpath, "single_extraction", [{
                    "source": filename,
                    "output": os.path.basename(out_path),
                    "status": "success",
                    "confidence": conf
                }])
            except Exception as e:
                console.print(f"[bold red]Extraction failed:[/bold red] {e}")
                self._log_to_manifest(dirpath, "single_extraction", [{
                    "source": filename,
                    "status": "failed",
                    "error": str(e)
                }])

    def _get_available_path(self, dirpath: str, filename: str) -> Optional[str]:
        """Handles skip/overwrite/index logic for extraction."""
        ext = os.path.splitext(filename)[1]
        base_name = "extracted"
        target_path = os.path.join(dirpath, f"{base_name}{ext}")
        
        mode = self.config["existing_file_mode"]
        
        if not os.path.exists(target_path):
            return target_path
            
        if mode == "skip":
            return None
        elif mode == "overwrite":
            return target_path
        elif mode == "index":
            idx = 2
            while True:
                new_path = os.path.join(dirpath, f"{base_name}{idx}{ext}")
                if not os.path.exists(new_path):
                    return new_path
                idx += 1
        return target_path

    def _log_to_manifest(self, root_dir: str, op_type: str, results: List[Dict[str, Any]]):
        manifest_path = os.path.join(root_dir, "manifest.json")
        data = {"operations": []}
        
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, "r") as f:
                    data = json.load(f)
            except: pass
            
        new_op = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "type": op_type,
            "config": self.config.copy(),
            "results": results
        }
        data["operations"].append(new_op)
        
        try:
            with open(manifest_path, "w") as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            console.print(f"[red]Failed to write manifest: {e}[/red]")

    def _run_batch_extraction(self):
        console.print("[yellow]Please select the root directory for batch extraction...[/yellow]")
        root_dir = self.prompt_for_directory("Select Root Directory")
        if not root_dir: return

        console.print(f"Scanning directory recursively: [green]{root_dir}[/green]")
        
        folders_to_process = []
        for dirpath, dirnames, filenames in os.walk(root_dir):
            match = self._find_image_with_keyword(dirpath, self.config["extraction_keyword"])
            if match:
                out_path = self._get_available_path(dirpath, os.path.basename(match))
                if out_path:
                    folders_to_process.append((match, out_path))

        if not folders_to_process:
            console.print("[yellow]No new images found to extract.[/yellow]")
            return

        console.print(f"Found [bold]{len(folders_to_process)}[/bold] images to process.")
        if not Confirm.ask("Proceed?"): return

        self._ensure_models_initialized()
        results_table = Table(title="Batch Extraction Results")
        results_table.add_column("Folder", style="cyan"); results_table.add_column("Source", style="green")
        results_table.add_column("Confidence", justify="right"); results_table.add_column("Status")

        op_results = []
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), TaskProgressColumn(), console=console) as progress:
            task = progress.add_task("[magenta]Extracting...", total=len(folders_to_process))
            for src, out in folders_to_process:
                folder = os.path.basename(os.path.dirname(src))
                filename = os.path.basename(src)
                try:
                    conf = self.engine.extract_face(src, out, padding=self.config["padding_ratio"])
                    results_table.add_row(folder, filename, f"{conf:.1%}", "[green]SUCCESS[/green]")
                    op_results.append({"folder": folder, "source": filename, "output": os.path.basename(out), "status": "success", "confidence": conf})
                except Exception as e:
                    results_table.add_row(folder, filename, "0%", f"[red]FAIL: {e}[/red]")
                    op_results.append({"folder": folder, "source": filename, "status": "failed", "error": str(e)})
                progress.advance(task)

        console.print(results_table)
        self._log_to_manifest(root_dir, "batch_extraction", op_results)
        console.print(f"\n[bold green]Complete. Manifest saved to {root_dir}/manifest.json[/bold green]")

    def _find_image_with_keyword(self, folder_path: str, keyword: str) -> Optional[str]:
        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
        try:
            pattern = re.compile(keyword, re.IGNORECASE)
        except re.error:
            pattern = None

        keyword_lower = keyword.lower()
        candidates = [
            file for file in os.listdir(folder_path)
            if file.lower().endswith(valid_extensions)
        ]

        # First pass: regex/contains exact-ish matching.
        for file in candidates:
            file_lower = file.lower()
            if (pattern and pattern.search(file)) or (not pattern and keyword_lower in file_lower):
                return os.path.join(folder_path, file)

        # Second pass: fuzzy fallback for approximate name matching.
        best_path = None
        best_score = 0.0
        for file in candidates:
            stem = os.path.splitext(file)[0]
            score = self._fuzzy_match_score(keyword_lower, stem)
            if score > best_score:
                best_score = score
                best_path = os.path.join(folder_path, file)

        if best_path and best_score >= 0.60:
            return best_path
        return None

    def _fuzzy_match_score(self, keyword: str, filename_stem: str) -> float:
        keyword_clean = keyword.strip().lower()
        filename_clean = filename_stem.strip().lower()
        if not keyword_clean or not filename_clean:
            return 0.0

        sequence_score = SequenceMatcher(None, keyword_clean, filename_clean).ratio()
        filename_parts = [p for p in re.split(r"[_\-\s]+", filename_clean) if p]
        part_score = max(
            (SequenceMatcher(None, keyword_clean, part).ratio() for part in filename_parts),
            default=0.0,
        )

        keyword_tokens = set(re.findall(r"[a-z0-9]+", keyword_clean))
        filename_tokens = set(re.findall(r"[a-z0-9]+", filename_clean))
        token_overlap = (
            len(keyword_tokens & filename_tokens) / len(keyword_tokens)
            if keyword_tokens else 0.0
        )

        return max(sequence_score, token_overlap, part_score)

    def _get_new_folder_name(self, old_folder_path: str, score: float) -> str:
        parent_dir = os.path.dirname(old_folder_path)
        folder_name = os.path.basename(old_folder_path)
        rounded_score = int(score)

        # Remove one trailing score token so we overwrite rather than append.
        def strip_score_token(text: str) -> str:
            return re.sub(r"\s+\d{1,3}%?$", "", text).rstrip()

        if " - " in folder_name:
            parts = folder_name.split(" - ", 1)
            left = strip_score_token(parts[0]) or parts[0].strip()
            new_name = f"{left} {rounded_score} - {parts[1]}"
        else:
            base = strip_score_token(folder_name) or folder_name.strip()
            new_name = f"{base} {rounded_score}"

        return os.path.join(parent_dir, new_name)

    def _run_batch_processing(self):
        console.print("[yellow]Please select root for similarity check...[/yellow]")
        root_dir = self.prompt_for_directory("Select Root Directory")
        if not root_dir: return

        folders_to_process = []
        for dirpath, dirnames, filenames in os.walk(root_dir):
            img1 = self._find_image_with_keyword(dirpath, self.config["img1_keyword"])
            img2 = self._find_image_with_keyword(dirpath, self.config["img2_keyword"])
            if img1 and img2: folders_to_process.append((dirpath, img1, img2))

        if not folders_to_process:
            console.print("[yellow]No folders matched keywords.[/yellow]")
            return

        console.print(f"Found [bold]{len(folders_to_process)}[/bold] folders. Proceed?")
        if not Confirm.ask("Confirm"): return

        self._ensure_models_initialized()
        results_table = Table(title="Batch Similarity Results")
        results_table.add_column("Folder"); results_table.add_column("Score"); results_table.add_column("Match"); results_table.add_column("Status")

        op_results = []
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), TaskProgressColumn(), console=console) as progress:
            task = progress.add_task("[magenta]Processing...", total=len(folders_to_process))
            for folder, i1, i2 in folders_to_process:
                folder_name = os.path.basename(folder)
                result = self.engine.compare_images(i1, i2)
                if result.get("error"):
                    results_table.add_row(folder_name, "-", "N/A", result["error"])
                    op_results.append({"folder": folder_name, "status": "error", "error": result["error"]})
                else:
                    score, is_match = result["score"], result["match"]
                    match_text = "[green]YES[/green]" if is_match else "[red]NO[/red]"
                    try:
                        new_path = self._get_new_folder_name(folder, score)
                        status = "Skip (Same)"
                        if new_path != folder:
                            os.rename(folder, new_path)
                            folder_name = f"{folder_name} -> {os.path.basename(new_path)}"
                            status = "Renamed"
                        results_table.add_row(folder_name, f"{score}%", match_text, f"[green]{status}[/green]")
                        op_results.append({"folder": folder_name, "score": score, "match": is_match, "status": status})
                    except Exception as e:
                        results_table.add_row(folder_name, f"{score}%", match_text, f"[red]Rename fail: {e}[/red]")
                        op_results.append({"folder": folder_name, "score": score, "match": is_match, "status": "rename_failed", "error": str(e)})
                progress.advance(task)

        console.print(results_table)
        self._log_to_manifest(root_dir, "batch_similarity", op_results)
        console.print(f"\n[bold green]Complete. Manifest saved to {root_dir}/manifest.json[/bold green]")

    def _display_result(self, result: dict) -> None:
        if result.get("error"):
            console.print(Panel(f"[bold red]Error:[/bold red] {result['error']}", border_style="red"))
            return
        score, is_match = result["score"], result["match"]
        color = "green" if is_match else "red"
        status = "are" if is_match else "are not"
        console.print(Panel(f"Face similarity ratio: [{color} bold]{score}%[/{color} bold], The two photos [{color} bold]{status}[/{color} bold] the same person.", border_style=color, expand=False))

if __name__ == "__main__":
    cli = ProCLI()
    cli.run()
