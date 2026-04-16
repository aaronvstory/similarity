import argparse
import sys

def main():
    """
    Entry point for the Face Similarity Application.
    Routes to either the GUI (default) or the CLI based on arguments.
    """
    parser = argparse.ArgumentParser(
        description="Enterprise-Grade Local Face Similarity Application",
        epilog="By default, launches the modern dark-mode GUI."
    )
    parser.add_argument(
        "--cli", 
        action="store_true", 
        help="Run in Command Line Interface mode."
    )
    parser.add_argument(
        "--img1", 
        type=str, 
        help="Path to the first image (CLI mode only)."
    )
    parser.add_argument(
        "--img2", 
        type=str, 
        help="Path to the second image (CLI mode only)."
    )

    args = parser.parse_args()

    # If --cli is passed or if any image arguments are provided, launch CLI.
    if args.cli or args.img1 or args.img2:
        try:
            from src.cli import ProCLI
            cli = ProCLI()
            cli.run(img1_path=args.img1, img2_path=args.img2)
        except ImportError as e:
            print(f"Failed to load CLI components. Ensure all dependencies are installed: {e}")
            sys.exit(1)
    else:
        try:
            from src.gui import run_gui
            run_gui()
        except ImportError as e:
            print(f"Failed to load GUI components. Ensure all dependencies are installed: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()
