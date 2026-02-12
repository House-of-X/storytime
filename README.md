# Storytime

A text-to-speech audiobook generator that converts EPUB files into high-quality MP3 audiobooks with natural-sounding narration.

## Features

- Converts EPUB files to MP3 audiobooks
- Generates separate MP3 files for each chapter
- Hybrid voice blending for natural-sounding narration
- Professional audio post-processing
- Interactive chapter selection
- Index-based chapter range selection

## Usage

### Basic Command

```bash
python main.py <epub_file>
```

### Command-Line Options

| Option | Type | Description |
|--------|------|-------------|
| `filename` | Required | Path to the EPUB file to process |
| `--output-dir` | Optional | Output directory for generated files (default: `output_audio`) |
| `--voice-type` | Optional | Voice gender: `male` or `female` (default: `female`) |
| `--start-chapter` | Optional | Starting chapter index (0-based) |
| `--end-chapter` | Optional | Ending chapter index (0-based, inclusive) |
| `--keep-artifacts` | Flag | Retain intermediate files (raw markdown, processed text, raw audio) |
| `--print-toc` | Flag | Print table of contents with indexes and exit |

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

- **MP3 files**: One MP3 file per chapter, named after the chapter title
- **Temp directory**: Contains intermediate processing files (if `--keep-artifacts` is used)

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

## Chapter Selection

Chapters are indexed starting from 0. Use `--print-toc` to view available chapters and their indexes before processing.

### Index-Based Selection

- `--start-chapter 0 --end-chapter 2`: Generates chapters 0, 1, and 2
- `--start-chapter 5`: Generates from chapter 5 to the end
- `--end-chapter 10`: Generates from chapter 0 to chapter 10

### Error Handling

The tool validates chapter indexes and reports errors for:

- Invalid start chapter index (out of range)
- Invalid end chapter index (out of range or less than start)
