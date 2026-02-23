# pdf-ocr-md

Python CLI that scans **only the current directory root** for PDFs, runs OCR/transcription with LM Studio, and writes **one aggregated markdown file per PDF**.

Each output markdown includes:

- A **Clean Aggregate** section for the full file
- Per-slide sections with:
  - Retranscribed text
  - Math (LaTeX-friendly markdown)
  - Image descriptions

## Requirements

- Python 3.10+
- LM Studio running locally with a vision-capable model loaded
- Start LM Studio's local server (default: `http://localhost:1234`)

## Install

From this folder:

```bash
pip install -e .
```

## Usage

Run from the folder that contains your PDFs:

```bash
pdf-ocr-md
```

Interactive TUI launcher:

```bash
pdf-ocr-md-tui
```

The TUI lets you select:

- which PDFs to process,
- model and LM Studio URL,
- quality preset (fast/balanced/high),
- output folder and dry-run.

Options:

```bash
pdf-ocr-md --llm-url http://localhost:1234 --dpi 180
```

Useful flags:

- `--output-dir <path>`: write markdown files to another directory
- `--min-native-chars <n>`: threshold for trusting native PDF text as context
- `--dry-run`: list/discover processing targets without calling the LLM
- `--pdf <name.pdf>`: process only selected PDF(s); repeat for multiple files
- `--workers <n>`: parallel page OCR requests per PDF (default: `2`)
- `--native-fast-path`: only call the LLM when weak native text, math, or likely diagrams are detected
- `--skip-aggregate-cleanup`: skip the final whole-document rewrite call for speed
- `--no-tui`: disable the OCR queue progress slider in terminal

## Output

For each `name.pdf`, the tool writes `markdown/name.md` by default.

After a PDF is successfully handled, the source file is moved to `processed/name.pdf`.
If a name collision exists, a suffix like `-1` is appended.

## Notes

- Root-only scan is non-recursive by design.
- OCR quality depends on model quality and page render DPI.
- Math is requested from the model in LaTeX-ready markdown format.

## Speed tips

- Start with: `pdf-ocr-md --native-fast-path --workers 3 --skip-aggregate-cleanup`
- `--native-fast-path` applies a smart trigger per page:
  - calls the LLM when math signals are detected,
  - calls the LLM when diagram/visual signals are detected,
  - skips the LLM for plain text-heavy pages with usable native extraction.
- By default, the terminal TUI shows:
  - documents completed vs remaining,
  - per-PDF all slides analyzed (including native-fast-path and OCR pages),
  - slides in the OCR queue,
  - slides completed vs left,
  - global slides completed vs total slides across all selected PDFs,
  - elapsed time and ETA.
