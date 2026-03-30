# Repository Guidelines

## Project Structure & Module Organization
- `API/`: Python APIs and scripts for detection/segmentation plus examples and tests (e.g., `grounding_dino_api.py`, `grounding_sam_api.py`, `test_api.py`).
- `Grounded-SAM-2/`: Upstream Grounded-SAM-2 project with model code, checkpoints, and install docs.
- `dataset/`: Data used by local experiments.
- `table/`: Result tables and sample outputs.
- `README.md`: Project overview and quick-start commands.

## Build, Test, and Development Commands
- Install dependencies (see `Grounded-SAM-2/INSTALL.md`):
  - `pip install torch torchvision opencv-python supervision pycocotools hydra-core iopath timm transformers`
- Install editable packages:
  - `pip install -e Grounded-SAM-2`
  - `pip install -e Grounded-SAM-2/grounding_dino --no-build-isolation`
- Run APIs from `API/`:
  - `python grounding_dino_api.py --image ../Grounded-SAM-2/bag.jpg --prompt "bag." --output result.jpg`
  - `python grounding_sam_api.py --image ../Grounded-SAM-2/bag.jpg --prompt "bag." --output output/`
- Example usage:
  - `python API/example_usage.py`

## Coding Style & Naming Conventions
- Python code uses 4-space indentation and `snake_case` module/script names (e.g., `grounding_sam_api.py`).
- Prompt strings are lowercase English words ending with a period and separated by `. ` (e.g., `"cat. dog."`).
- Formatting tools appear in the upstream project (`black` and `yapf` are listed under `Grounded-SAM-2`); prefer them if editing that subtree.

## Testing Guidelines
- No formal test runner is configured. Use `python API/test_api.py` for smoke tests and confirm output images/JSON under the configured output paths.
- Keep any new tests lightweight and runnable from the repo root.

## Commit & Pull Request Guidelines
- This directory is not a Git repository, so no commit history is available. Use concise, imperative commit subjects (e.g., `Add SAM API example`) if you initialize Git.
- For PRs: include a short summary, steps to reproduce, and sample outputs (image/JSON) when changing detection or segmentation behavior.

## Configuration & Assets
- Model weights are large and stored outside Git. Ensure checkpoint paths under `Grounded-SAM-2/checkpoints/` and `Grounded-SAM-2/gdino_checkpoints/` exist before running.
- Update hard-coded local paths in `API/README.md` when sharing scripts across machines.
