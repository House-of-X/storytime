# Storytime

A text-to-speech audiobook generator that converts EPUB and Markdown files into high-quality MP3 audiobooks with natural-sounding narration.

## Features

- Converts EPUB and Markdown files to MP3 audiobooks
- Generates separate MP3 files for each chapter/section
- Hybrid voice blending for natural-sounding narration
- Professional audio post-processing
- Interactive chapter selection (EPUB)
- Index-based chapter range selection (EPUB)
- Extract-only mode for text extraction without audio generation

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Command

```bash
python main.py <file>
```

Supported formats: `.epub`, `.md`

### Command-Line Options

| Option | Type | Description |
|--------|------|-------------|
| `filename` | Required | Path to the EPUB or Markdown file to process |
| `--output-dir` | Optional | Output directory for generated files (default: `output_audio`) |
| `--voice-type` | Optional | Voice gender: `male` or `female` (default: `female`) |
| `--start-chapter` | Optional | Starting chapter index, 0-based (EPUB only) |
| `--end-chapter` | Optional | Ending chapter index, 0-based, inclusive (EPUB only) |
| `--keep-artifacts` | Flag | Retain intermediate files (raw markdown, processed text, raw audio) |
| `--extract-only` | Flag | Extract text artifacts only, no audio generation |
| `--print-toc` | Flag | Print table of contents with indexes and exit (EPUB only) |

### Examples

#### Print Table of Contents

Display all chapters with their indexes:

```bash
python main.py book.epub --print-toc
```

#### Generate All Chapters

Process the entire book with interactive chapter selection:

```bash
python main.py book.epub
```

#### Generate from Markdown

Process a Markdown file (splits on `#` and `##` headings):

```bash
python main.py document.md
```

#### Extract Text Only

Extract text artifacts without generating audio:

```bash
python main.py book.epub --extract-only
python main.py document.md --extract-only
```

#### Generate Specific Chapter Range

Generate chapters 0 through 5 using index-based selection:

```bash
python main.py book.epub --start-chapter 0 --end-chapter 5
```

#### Generate Single Chapter

Generate only chapter 3:

```bash
python main.py book.epub --start-chapter 3 --end-chapter 3
```

#### Custom Output Directory

Specify a custom output directory:

```bash
python main.py book.epub --output-dir my_audiobook
```

#### Male Voice

Use male voice narration:

```bash
python main.py book.epub --voice-type male
```

#### Keep Intermediate Files

Retain processing artifacts for debugging:

```bash
python main.py book.epub --keep-artifacts
```

## Output

The tool generates:

- **MP3 files**: One MP3 file per chapter/section, named after the chapter title
- **Temp directory**: Contains intermediate processing files (if `--keep-artifacts` or `--extract-only` is used)

### Output Structure

```
output_audio/
├── Chapter_1_Title.mp3
├── Chapter_2_Title.mp3
├── Chapter_3_Title.mp3
└── temp/
    ├── raw.md
    ├── processed.txt
    └── chunks.txt
```

## Supported Formats

### EPUB

- Full table of contents support
- Interactive and index-based chapter selection
- HTML-to-plaintext conversion with deduplication

### Markdown

- Splits into sections at `#` and `##` headings
- Heading text is used as the section title
- All sections are processed (no chapter range selection)

## Chapter Selection

Chapters are indexed starting from 0. Use `--print-toc` to view available chapters and their indexes before processing.

### Index-Based Selection (EPUB only)

- `--start-chapter 0 --end-chapter 2`: Generates chapters 0, 1, and 2
- `--start-chapter 5`: Generates from chapter 5 to the end
- `--end-chapter 10`: Generates from chapter 0 to chapter 10

### Error Handling

The tool validates chapter indexes and reports errors for:

- Invalid start chapter index (out of range)
- Invalid end chapter index (out of range or less than start)
