import os
import sys
import argparse
import re
import json
import random
import numpy as np
import torch
import soundfile as sf
import boto3
import pymupdf4llm
from pathlib import Path
from kokoro import KPipeline
from pydub import AudioSegment
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from pedalboard import (
    Pedalboard, 
    Compressor, 
    Reverb, 
    Limiter, 
    Gain, 
    Distortion, 
    HighShelfFilter, 
    LowShelfFilter,
    PeakFilter
)
from pedalboard.io import AudioFile

# --- CONFIGURATION ---
VOICE_1 = 'af_bella'
VOICE_2 = 'af_sarah'
BLEND_RATIO = 0.5  # 0.5 = 50% Bella, 50% Sarah
OUTPUT_DIR = Path('output_audio')
TEMP_DIR = Path('temp_processing')

# Ensure directories exist
OUTPUT_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)

# --- 1. ARGUMENT PARSING ---
parser = argparse.ArgumentParser(description='Process PDF files for text-to-speech with Audio Engineering')
parser.add_argument('filename', help='PDF file to process')
parser.add_argument('--start-page', type=int, default=1, help='Starting page number')
parser.add_argument('--end-page', type=int, help='Ending page number')
parser.add_argument('--no-llm', action='store_true', help='Skip Claude cleaning (faster, less expensive)')
parser.add_argument('--keep-artifacts', action='store_true', help='Retain intermediate files (PDF markdown, LLM output, raw audio)')
args = parser.parse_args()

if not args.filename.lower().endswith('.pdf'):
    print(f"Error: Unsupported format. Please provide a .pdf file.", file=sys.stderr)
    sys.exit(1)

# --- 2. SETUP MODELS ---
print("--- Initializing Models ---")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Initialize Bedrock (Claude)
bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

# Initialize Kokoro Pipeline
pipeline = KPipeline(lang_code='a', device=device)

# --- 3. VOICE BLENDING LOGIC (THE FIX) ---
def get_blended_voice(v1_name, v2_name, ratio=0.5):
    """
    Loads two voice tensors, blends them mathematically, and saves a temporary voice file.
    This prevents 'ghosting' artifacts and crashes during generation.
    """
    print(f"--- Blending Voices: {v1_name} ({ratio*100}%) + {v2_name} ({(1-ratio)*100}%) ---")
    
    # KPipeline usually stores voices in a dictionary or downloads them to a cache.
    # We need to access the raw tensors. 
    # NOTE: This assumes standard KPipeline behavior. If voices aren't downloaded, 
    # run the pipeline once with a dummy text to trigger download.
    
    try:
        # Load voices from the internal cache or huggingface path
        # If this fails, ensure you have run the voices at least once or download .pt files manually
        v1 = pipeline.load_voice(v1_name)
        v2 = pipeline.load_voice(v2_name)
        
        # Blend the tensors
        # Ensure they are on the same device
        if isinstance(v1, torch.Tensor):
            v1 = v1.to(device)
            v2 = v2.to(device)
            blended_tensor = (v1 * ratio) + (v2 * (1 - ratio))
            return blended_tensor
        else:
            # Fallback for numpy arrays
            return (v1 * ratio) + (v2 * (1 - ratio))
            
    except Exception as e:
        print(f"Warning: Could not blend voices automatically ({e}). Using {v1_name} only.")
        return v1_name

# Create the hybrid voice tensor
hybrid_voice = get_blended_voice(VOICE_1, VOICE_2, BLEND_RATIO)

# --- 4. TEXT PROCESSING FUNCTIONS ---
def normalize_text(text):
    text = re.sub(r'-\n', '', text)
    text = re.sub(r'(?<![.!?:])\n(?!\n)', ' ', text)
    text = re.sub(r'¡\s*', '', text)
    text = re.sub(r'~~[^~]+~~', '', text)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'Figure \d+\.\d+[^\n]*', '', text)
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def clean_text_with_llm(text):
    if len(text.strip()) < 10: return text
    
    prompt = f"""
Act as a professional script editor for audiobooks. Your task is to prepare the following book text for a Text-to-Speech engine.

Now, your primary goal is to make the text "ear-friendly" for a listener. Please follow these specific guidelines:

 * **Maintain the Structure:** Do not remove chapter titles, chapter numbers, or chapter summaries. These must remain intact.

 * **Clean the Layout:** Remove all "print-only" artifacts such page numbers, page headers, and URLs.

 * **Do modify the prose**

 * **Expand Abbreviations:** Spell out abbreviations that sound awkward when spoken aloud---for example, change "e.g." to "for example" and "etc." to "and so on."

 * **Guide the Listener:** Add transition words like "Now," "Interestingly," or "On the other hand" to help the listener follow the logic of the narrative.

 * **Add Pacing:** Use an em-dash (---) to indicate a dramatic pause or an ellipsis (...) to signify a trailing thought.

The voice should feel authentic and engaging, as if a professional narrator is guiding the audience through the story.

RESULT FORMAT:
 * Return plain text and not Markdown
 * Do not include preamble, explanation, or additional commentary

TEXT:
{text}
"""
    
    try:
        response = bedrock.invoke_model(
            modelId="anthropic.claude-3-haiku-20240307-v1:0",
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": len(text) + 200,
                "temperature": 0.3,
                "messages": [{"role": "user", "content": prompt}]
            })
        )
        result = json.loads(response["body"].read())
        cleaned = result["content"][0]["text"].strip()
        return cleaned if len(cleaned) > len(text) * 0.2 else text
    except Exception as e:
        print(f"LLM Error: {e}. Using raw text.")
        return text

# --- 5. EXTRACT & PREPARE TEXT ---
print(f"--- Extracting PDF: {args.filename} ---")
md_text = pymupdf4llm.to_markdown(
    args.filename,
    pages=list(range(args.start_page - 1, args.end_page)) if args.end_page else None
)

# Basic cleanup
md_text = re.sub(r'```[\s\S]*?```', '', md_text)
md_text = re.sub(r'`[^`]+`', '', md_text)

# Chunking
header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[('#', 'h1'), ('##', 'h2')])
header_splits = header_splitter.split_text(md_text)

final_chunks = []
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=650, chunk_overlap=0, separators=["\n\n", ". ", "! ", "? ", "; "]
)

print("--- Processing Text Chunks ---")
for doc in header_splits:
    normalized = normalize_text(doc.page_content)
    if not args.no_llm:
        cleaned = clean_text_with_llm(normalized)
    else:
        cleaned = normalized
    final_chunks.extend(text_splitter.split_text(cleaned))

# --- 6. AUDIO GENERATION ---
print(f"--- Generating Audio ({len(final_chunks)} chunks) ---")

# Load breath sample
try:
    BREATH_SAMPLE = AudioSegment.from_mp3('templates/female-inhale.mp3').set_frame_rate(24000).apply_gain(-32).fade_in(100).fade_out(100)
except:
    print("Warning: Breath sample not found. Skipping breaths.")
    BREATH_SAMPLE = None



def get_jittered_pause(ms):
    return ms + random.randint(-50, 50)

audio_segments = []

# Monkey-patch once before the loop
original_load_voice = pipeline.load_voice
pipeline.load_voice = lambda x: hybrid_voice

for i, chunk_text in enumerate(final_chunks):
    if not chunk_text.strip(): continue
    
    # Decide if we need a breath BEFORE this chunk
    word_count = len(chunk_text.split())
    is_new_paragraph = i > 0 and "\n\n" in final_chunks[i-1]
    is_long_sentence = word_count > 20
    
    if BREATH_SAMPLE:
        breath_chance = 0.85 if is_new_paragraph else (0.65 if is_long_sentence else 0.40)
        if random.random() < breath_chance:
            pre_silence = random.randint(120, 180)
            post_silence = random.randint(80, 120)
            audio_segments.append(AudioSegment.silent(duration=pre_silence, frame_rate=24000))
            audio_segments.append(BREATH_SAMPLE)
            audio_segments.append(AudioSegment.silent(duration=post_silence, frame_rate=24000))
    
    base_speed = random.uniform(0.78, 0.84)
    current_speed = base_speed + random.uniform(-0.02, 0.02)
    generator = pipeline(chunk_text, voice=VOICE_1, speed=current_speed, split_pattern=r'\n+')
    
    for _, _, audio_tensor in generator:
        # Convert Torch/Numpy audio to PyDub AudioSegment
        if isinstance(audio_tensor, torch.Tensor):
            audio_np = audio_tensor.cpu().numpy()
        else:
            audio_np = audio_tensor
            
        # Ensure float32 -> int16 conversion for PyDub
        audio_int16 = (audio_np * 32767).astype(np.int16)
        
        segment = AudioSegment(
            audio_int16.tobytes(), 
            frame_rate=24000,
            sample_width=2, 
            channels=1
        )
        audio_segments.append(segment)

    # Improved Pause Logic with Jitter
    if "\n\n" in chunk_text:
        pause_ms = get_jittered_pause(1000)
    elif chunk_text.rstrip().endswith(('!', '?')):
        pause_ms = get_jittered_pause(800)
    elif chunk_text.rstrip().endswith('.'):
        pause_ms = get_jittered_pause(500)
    elif "," in chunk_text:
        pause_ms = get_jittered_pause(200)
    else:
        pause_ms = get_jittered_pause(150)
    
    audio_segments.append(AudioSegment.silent(duration=pause_ms, frame_rate=24000))
        
    if i % 5 == 0:
        print(f"Generated {i}/{len(final_chunks)} chunks...")

# Restore original method after loop
pipeline.load_voice = original_load_voice

# Combine all segments
full_audio = sum(audio_segments)

# Save extracted markdown if keeping artifacts
if args.keep_artifacts:
    md_output_path = TEMP_DIR / 'extracted_text.md'
    with open(md_output_path, 'w', encoding='utf-8') as f:
        f.write(md_text)
    print(f"Saved extracted markdown to: {md_output_path}")
    
    # Save processed chunks (LLM output)
    chunks_output_path = TEMP_DIR / 'processed_chunks.txt'
    with open(chunks_output_path, 'w', encoding='utf-8') as f:
        for idx, chunk in enumerate(final_chunks):
            f.write(f"--- Chunk {idx+1} ---\n{chunk}\n\n")
    print(f"Saved processed chunks to: {chunks_output_path}")

# Export raw (pre-processing) audio
raw_path = TEMP_DIR / 'raw_speech.wav'
full_audio.export(str(raw_path), format='wav')
if args.keep_artifacts:
    print(f"Saved raw audio to: {raw_path}")

# --- 7. POST-PROCESSING (PEDALBOARD) ---
print("--- Applying Audio Engineering (Pedalboard) ---")

# Define the Signal Chain
board = Pedalboard([
    # Proximity effect: Low-shelf boost
    LowShelfFilter(cutoff_frequency_hz=120, gain_db=2.5, q=0.7),
    
    # EQ: Boost body/warmth
    PeakFilter(cutoff_frequency_hz=200, gain_db=3.0, q=1.0),
    
    # De-esser: Tame harsh sibilance
    PeakFilter(cutoff_frequency_hz=6500, gain_db=-4.0, q=3.0),
    
    # Saturation: Tube-like warmth
    Distortion(drive_db=2.0),
    
    # Fast compressor: Catch peaks
    Compressor(threshold_db=-12.0, ratio=4.0, attack_ms=1.0, release_ms=50.0),
    
    # Slow compressor: Glue compression
    Compressor(threshold_db=-20.0, ratio=1.5, attack_ms=20.0, release_ms=200.0),
    
    # Air band: Add shimmer
    HighShelfFilter(cutoff_frequency_hz=12000, gain_db=1.5),
    
    # Reverb: Very subtle room tone
    Reverb(room_size=0.08, damping=0.9, wet_level=0.03, dry_level=0.97),
    
    # Limiter: Prevent clipping
    Limiter(threshold_db=-1.0, release_ms=100.0),
])

# Read raw file with Pedalboard
with AudioFile(str(raw_path)) as f:
    audio = f.read(f.frames)
    samplerate = f.samplerate

# Process
processed_audio = board(audio, samplerate)

# Save Final
output_file = OUTPUT_DIR / 'final_audiobook.mp3'
with AudioFile(str(output_file), 'w', samplerate, processed_audio.shape[0]) as f:
    f.write(processed_audio)

# Cleanup
if not args.keep_artifacts:
    try:
        os.remove(raw_path)
    except:
        pass
else:
    print(f"Artifacts retained in: {TEMP_DIR}")

print(f"DONE! File saved to: {output_file}")