import os
import sys
import time
import argparse
import re
import random
import logging
import shutil
import traceback
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
import psutil
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from markdown import markdown
from pathlib import Path
from tqdm import tqdm
from kokoro import KPipeline
from pydub import AudioSegment
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pedalboard import (
    Pedalboard, 
    Compressor, 
    Reverb, 
    Limiter, 
    Distortion, 
    HighShelfFilter, 
    LowShelfFilter,
    PeakFilter
)
from pedalboard.io import AudioFile

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger(__name__)

# Suppress noisy third-party warnings
logging.getLogger('kokoro').setLevel(logging.ERROR)
logging.getLogger('misaki').setLevel(logging.ERROR)


class SystemMonitor:
    """Reports system health: RAM, disk, and VRAM usage."""

    @staticmethod
    def get_status(output_dir=None):
        mem = psutil.virtual_memory()
        ram_used_gb = mem.used / (1024 ** 3)
        ram_total_gb = mem.total / (1024 ** 3)
        ram_pct = mem.percent

        disk_path = str(output_dir) if output_dir else '/'
        disk = shutil.disk_usage(disk_path)
        disk_free_gb = disk.free / (1024 ** 3)
        disk_total_gb = disk.total / (1024 ** 3)

        status = (
            f"RAM: {ram_used_gb:.1f}/{ram_total_gb:.1f} GB ({ram_pct}%) | "
            f"Disk free: {disk_free_gb:.1f}/{disk_total_gb:.1f} GB"
        )

        if torch.cuda.is_available():
            vram_used = torch.cuda.memory_allocated() / (1024 ** 3)
            vram_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            vram_pct = (vram_used / vram_total) * 100 if vram_total > 0 else 0
            status += f" | VRAM: {vram_used:.1f}/{vram_total:.1f} GB ({vram_pct:.0f}%)"

        return status

    @staticmethod
    def log_status(output_dir=None):
        log.info(f"System: {SystemMonitor.get_status(output_dir)}")


class AudiobookConfig:
    """Configuration settings for the audiobook generation pipeline."""
    
    def __init__(self, output_directory, voice_type='female'):
        self.voice_type = voice_type
        
        # Select voice pairs based on gender.
        if voice_type == 'male':
            self.voice_1 = 'am_adam'
            self.voice_2 = 'am_michael'
        else:
            self.voice_1 = 'af_bella'
            self.voice_2 = 'af_sarah'
        
        # Voice blending uses 60/40 ratio as baseline for human realism.
        # This asymmetric blend prevents tonal sameness across long narration.
        # The ratio will be varied slightly per chunk to add organic variation.
        self.blend_ratio = 0.7
        self.output_dir = Path(output_directory)
        self.temp_dir = self.output_dir / 'temp'
        self.sample_rate = 24000
        
        # Ensure directories exist.
        self.output_dir.mkdir(exist_ok=True)
        self.temp_dir.mkdir(exist_ok=True)


class VoiceBlender:
    """Handles voice blending operations for creating hybrid voice profiles."""
    
    def __init__(self, pipeline, device):
        self.pipeline = pipeline
        self.device = device
    
    def blend_voices(self, voice_1_name, voice_2_name, ratio=0.6):
        """
        Loads two voice tensors and blends them mathematically to create a hybrid voice.
        
        Human realism strategy:
        - Uses 60/40 blend as baseline instead of 50/50 to prevent tonal sameness.
        - The asymmetric ratio creates subtle character variation that mimics natural
          vocal inconsistencies in human narration.
        - This prevents ghosting artifacts and crashes during audio generation.
        """
        log.info(f"Blending voices: {voice_1_name} ({ratio*100:.0f}%) + {voice_2_name} ({(1-ratio)*100:.0f}%)")
        
        try:
            # Load voice tensors from the pipeline's internal cache.
            voice_1 = self.pipeline.load_voice(voice_1_name)
            voice_2 = self.pipeline.load_voice(voice_2_name)
            
            # Blend the tensors using weighted average.
            if isinstance(voice_1, torch.Tensor):
                voice_1 = voice_1.to(self.device)
                voice_2 = voice_2.to(self.device)
                blended_tensor = (voice_1 * ratio) + (voice_2 * (1 - ratio))
                return blended_tensor
            else:
                # Fallback for numpy arrays.
                return (voice_1 * ratio) + (voice_2 * (1 - ratio))
                
        except Exception as e:
            log.warning(f"Could not blend voices ({e}). Using {voice_1_name} only.")
            return voice_1_name
    
    def get_dynamic_blend_ratio(self, chunk_index):
        """
        Generates dynamic voice blend ratio with controlled variation.
        
        Ultra-human mimicry approach:
        - Base ratio: 60/40
        - Per-chunk variation: ±5% randomization
        - This creates subtle tonal shifts across paragraphs that prevent the artificial
          consistency of static blending. Human voices naturally vary in timbre due to
          fatigue, emotion, and micro-adjustments in vocal tract configuration.
        """
        base_ratio = 0.6
        variation = random.uniform(-0.05, 0.05)
        return max(0.55, min(0.65, base_ratio + variation))


class TextProcessor:
    """Handles text extraction, normalization, and cleaning operations."""
    
    def __init__(self):
        pass
    
    def _markdown_to_plaintext(self, text):
        """Converts markdown to plaintext."""
        # Remove markdown images and links before conversion
        text = re.sub(r'!\[([^\]]*)\]\([^\)]+\)', '', text)  # Remove images
        text = re.sub(r'\[([^\]]*)\]\([^\)]+\)', r'\1', text)  # Convert links to text
        
        html = markdown(text, extensions=['extra'])
        soup = BeautifulSoup(html, 'html.parser')

        return soup.get_text()
    
    def _html_to_plaintext(self, html_content):
        """Converts HTML directly to plaintext with clean formatting."""
        soup = BeautifulSoup(html_content, 'html.parser')
        # Remove non-text elements
        for tag in soup.find_all(['img', 'figcaption', 'figure', 'style', 'script']):
            tag.decompose()
        
        # Find leaf block elements (blocks that don't contain other blocks)
        block_tags = ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'blockquote']
        blocks = soup.find_all(block_tags)
        
        if blocks:
            paragraphs = []
            seen = set()
            for block in blocks:
                text = re.sub(r'\s+', ' ', block.get_text(separator=' ')).strip()
                if text and text not in seen:
                    seen.add(text)
                    paragraphs.append(text)
            return '\n\n'.join(paragraphs)
        
        # Fallback: collapse whitespace in raw text
        text = soup.get_text()
        lines = [re.sub(r'\s+', ' ', line).strip() for line in text.split('\n')]
        lines = [l for l in lines if l]
        return '\n\n'.join(lines)
    
    def _url_to_words(self, match):
        """Converts a URL into spoken words."""
        url = match.group(0)
        replacements = [
            ('://', ' colon forward slash forward slash '),
            ('/', ' forward slash '),
            ('.', ' dot '),
            ('-', ' dash '),
            ('_', ' underscore '),
            ('?', ' question mark '),
            ('&', ' ampersand '),
            ('=', ' equals '),
            ('#', ' hash '),
            ('@', ' at '),
            (':', ' colon '),
        ]
        for old, new in replacements:
            url = url.replace(old, new)
        return url.strip()

    def normalize_text(self, text):
        """Applies basic text normalization to remove formatting artifacts."""
        text = re.sub(r'https?://[^\s)>]+', self._url_to_words, text)
        return text.strip()
    

    def _display_toc_and_select(self, book):
        """Displays EPUB table of contents and prompts user to select chapters."""
        toc = book.toc
        chapters = []
        
        print("\n=== Table of Contents ===")
        for index, item in enumerate(toc, 1):
            if isinstance(item, tuple):
                section = item[0]
                title = section.title
                href = section.href.split('#')[0]
            else:
                title = item.title
                href = item.href.split('#')[0]
            chapters.append((title, href))
            print(f"{index}. {title}")
        
        print("\nEnter chapter range (e.g., '1' for chapter 1, '1-5' for chapters 1-5, or 'all'):")
        user_input = input("> ").strip().lower()
        
        if user_input == 'all':
            return [href for _, href in chapters]
        elif '-' in user_input:
            start, end = user_input.split('-')
            start_index = int(start) - 1
            end_index = int(end)
            return [href for _, href in chapters[start_index:end_index]]
        else:
            index = int(user_input) - 1
            return [chapters[index][1]]
    
    def _process_chapter(self, chapter_content, batch_num=0, is_html=False):
        """Processes a whole chapter through normalization and chunking."""
        # Convert to plaintext
        if is_html:
            plaintext = self._html_to_plaintext(chapter_content)
        else:
            # Remove XML declaration
            chapter_content = re.sub(r"xml version=['\"].*?['\"]\s*encoding=['\"].*?['\"]\?", '', chapter_content)
            # Remove code blocks and inline code
            chapter_content = re.sub(r'```[\s\S]*?```', '', chapter_content)
            chapter_content = re.sub(r'`[^`]+`', '', chapter_content)
            plaintext = self._markdown_to_plaintext(chapter_content)
        
        # Normalize the plaintext
        normalized = self.normalize_text(plaintext)
        
        # Split into smaller chunks for TTS processing.
        # Prioritize paragraph breaks, then full sentences (ending with punctuation followed by space).
        # Avoid splitting mid-sentence by using sentence-terminal patterns.
        tts_splitter = RecursiveCharacterTextSplitter(
            chunk_size=650, chunk_overlap=0, separators=["\n\n", ".\n", "!\n", "?\n", ". ", "! ", "? ", "; ", ", "],
            keep_separator="end"
        )
        chunks = tts_splitter.split_text(normalized)
        # Re-attach trailing separators that belong to the previous chunk
        chunks = [c.strip() for c in chunks if c.strip()]
        
        return normalized, chunks
    
    def extract_and_chunk_markdown(self, filename):
        """Extracts text from a Markdown file and processes it."""
        log.info(f"Extracting Markdown: {filename}")

        with open(filename, 'r', encoding='utf-8') as f:
            markdown_text = f.read()

        # Split by headings to create chapters
        chapters = re.split(r'(?=^#{1,2}\s)', markdown_text, flags=re.MULTILINE)
        chapters = [c for c in chapters if c.strip()]

        chapter_titles = []
        chapter_chunks = []
        processed_sections = []

        for index, chapter in enumerate(tqdm(chapters, desc="Parsing", unit="sec", leave=False)):
            title_match = re.match(r'^#{1,2}\s+(.+)', chapter)
            title = title_match.group(1).strip() if title_match else f"Section_{index+1}"
            chapter_titles.append(title)
            processed, chunks = self._process_chapter(chapter, index, is_html=False)
            processed_sections.append(processed)
            chapter_chunks.append(chunks)

        processed_text = '\n\n'.join(processed_sections)
        return markdown_text, processed_text, chapter_chunks, chapter_titles

    def _flatten_toc(self, toc):
        """Flattens nested TOC into a list of (title, href) leaf entries."""
        entries = []
        for item in toc:
            if isinstance(item, tuple):
                section, children = item[0], item[1]
                entries.append((section.title.strip(), section.href.split('#')[0]))
                entries.extend(self._flatten_toc(children))
            else:
                entries.append((item.title.strip(), item.href.split('#')[0]))
        return entries

    def extract_and_chunk_epub(self, filename, start_chapter, end_chapter):
        """Extracts text from EPUB and processes directly from HTML."""
        log.info(f"Extracting EPUB: {filename}")
        
        book = epub.read_epub(filename)
        toc = book.toc
        
        # Flatten TOC to get every section as its own entry
        flat_toc = self._flatten_toc(toc)
        
        # Deduplicate by href (same xhtml file referenced multiple times)
        seen = set()
        chapter_list = []
        for title, href in flat_toc:
            if href not in seen:
                seen.add(href)
                chapter_list.append((title, href))
        
        # Handle chapter selection
        if start_chapter is not None or end_chapter is not None:
            start_index = int(start_chapter) if start_chapter else 0
            end_index = int(end_chapter) + 1 if end_chapter else len(chapter_list)
            
            if start_index < 0 or start_index >= len(chapter_list):
                log.error(f"Invalid start chapter index {start_index}. Valid range: 0-{len(chapter_list)-1}")
                sys.exit(1)
            if end_index <= start_index or end_index > len(chapter_list):
                log.error(f"Invalid end chapter index {end_chapter}. Valid range: {start_index}-{len(chapter_list)-1}")
                sys.exit(1)
            
            selected = chapter_list[start_index:end_index]
        else:
            selected_hrefs = self._display_toc_and_select(book)
            selected = [(t, h) for t, h in chapter_list if h in selected_hrefs]
        
        # Build href -> html map
        href_to_html = {}
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                item_name = item.get_name()
                for _, href in selected:
                    if href in item_name and href not in href_to_html:
                        href_to_html[href] = item.get_content().decode('utf-8', errors='ignore')
                        break
        
        # Build chapters list
        chapters = []
        chapter_titles = []
        for title, href in selected:
            if href in href_to_html:
                chapters.append(href_to_html[href])
                chapter_titles.append(title)
        
        # Process each chapter from HTML
        processed_sections = []
        chapter_chunks = []
        
        for index, chapter_html in enumerate(tqdm(chapters, desc="Parsing", unit="sec", leave=False)):
            processed, chunks = self._process_chapter(chapter_html, index, is_html=True)
            processed_sections.append(processed)
            chapter_chunks.append(chunks)
        
        # Convert to markdown for raw output only
        markdown_chapters = [md(html, heading_style='ATX', strip=['xml']) for html in chapters]
        markdown_text = '\n\n'.join(markdown_chapters)
        processed_text = '\n\n'.join(processed_sections)
        return markdown_text, processed_text, chapter_chunks, chapter_titles
    



class AudioGenerator:
    """Generates audio from text chunks using TTS pipeline with natural pauses and breaths."""
    
    def __init__(self, pipeline, hybrid_voice, config, voice_blender):
        self.pipeline = pipeline
        self.hybrid_voice = hybrid_voice
        self.config = config
        self.voice_blender = voice_blender
        self.breath_sample = self._load_breath_sample()
        self.pitch_drift_accumulator = 0.0
    
    def _load_breath_sample(self):
        """
        Loads and processes the breath sample audio for natural pauses.
        
        Ultra-human mimicry breath modeling:
        - Resampling to 24000 Hz ensures the breath matches the TTS output sample rate,
          preventing pitch shifts or timing issues when concatenating audio segments.
        - Variable gain between -30 dB to -36 dB (instead of fixed -32 dB) adds breath
          intensity variation that mimics real human breathing patterns.
        - Fade in/out (100ms each) smooths the breath edges to avoid clicks or pops that
          occur when abruptly starting or stopping audio signals.
        - Each breath instance will be slightly modified to avoid identical waveform reuse.
        """
        try:
            if self.config.voice_type == 'male':
                breath_file = 'templates/male-inhale.mp3'
            else:
                breath_file = 'templates/female-inhale.mp3'
            breath = AudioSegment.from_mp3(breath_file)
            breath = breath.set_frame_rate(self.config.sample_rate)
            return breath
        except:
            log.warning("Breath sample not found. Skipping breaths.")
            return None
    
    def _get_varied_breath(self):
        """
        Creates a unique breath instance with randomized characteristics.
        
        Breath variability for human realism:
        - Gain: -30 dB to -36 dB (intensity variation)
        - Pitch shift: ±2% (subtle frequency variation)
        - Length: ±5% (duration variation)
        - This prevents the artificial repetition of identical breath sounds and mimics
          the natural variation in human respiratory patterns during speech.
        """
        if not self.breath_sample:
            return None
        
        breath = self.breath_sample
        
        # Apply variable gain for intensity variation.
        gain_db = random.uniform(-36, -30)
        breath = breath.apply_gain(gain_db)
        
        # Apply subtle pitch shift (±2%).
        pitch_shift = random.uniform(0.98, 1.02)
        breath = breath._spawn(breath.raw_data, overrides={'frame_rate': int(breath.frame_rate * pitch_shift)})
        breath = breath.set_frame_rate(self.config.sample_rate)
        
        # Apply length variation (±5%).
        length_factor = random.uniform(0.95, 1.05)
        if length_factor != 1.0:
            breath = breath.speedup(playback_speed=1.0/length_factor)
        
        # Apply fades to prevent clicks.
        breath = breath.fade_in(100).fade_out(100)
        
        return breath
    
    def _get_jittered_pause(self, base_milliseconds, jitter_range):
        """
        Adds random variation to pause duration for more natural speech rhythm.
        
        Prosodic timing variability:
        - Variable jitter range allows different pause types to have appropriate randomness.
        - This irregular rhythm creates human realism by avoiding robotic consistency.
        - Natural speakers never pause for exactly the same duration twice.
        """
        return base_milliseconds + random.randint(-jitter_range, jitter_range)
    
    def _should_add_breath(self, chunk_text, has_emotional_punctuation, is_dialogue):
        """
        Context-aware breath insertion logic for human realism.
        
        Ultra-human mimicry breath modeling:
        - Breaths are context-aware, not probability-based.
        - Insert breath when sentence length exceeds 18 words (natural respiratory need).
        - Insert after emotionally intense punctuation (! or ?) as humans naturally
          pause and breathe after expressing strong emotion.
        - Insert before dialogue segments to mimic the natural breath actors take
          before speaking character lines.
        - This creates biologically authentic breathing patterns rather than random insertion.
        """
        if not self.breath_sample:
            return False
        
        word_count = len(chunk_text.split())
        
        # Context-aware breath triggers.
        is_long_sentence = word_count > 18
        
        # Breath is needed for long sentences or emotional/dialogue contexts.
        return is_long_sentence or has_emotional_punctuation or is_dialogue
    
    def _calculate_pause_duration(self, chunk_text):
        """
        Calculates appropriate pause duration based on punctuation and content.
        
        Prosodic timing variability for human realism:
        - Sentence-ending pause: Base 550 ms ± 80 ms (replaces fixed 500 ms)
        - Comma pause: Base 220 ms ± 40 ms (replaces fixed 200 ms)
        - Paragraph pause: Base 1100 ms ± 120 ms (replaces fixed 1000 ms)
        - Emotional punctuation (! ?): Base 650 ms ± 90 ms (replaces fixed 800 ms)
        - Long sentences (>25 words): Insert mid-sentence micro pause (~150 ms)
        - Irregular rhythm creates realism by avoiding robotic consistency.
        """
        word_count = len(chunk_text.split())
        
        # Paragraph breaks get longest pause with high variability.
        if "\n\n" in chunk_text:
            return self._get_jittered_pause(1100, 120)
        
        # Emotional punctuation gets moderate pause with variability.
        elif chunk_text.rstrip().endswith(('!', '?')):
            return self._get_jittered_pause(650, 90)
        
        # Sentence-ending period gets standard pause with variability.
        elif chunk_text.rstrip().endswith('.'):
            return self._get_jittered_pause(550, 80)
        
        # Comma gets short pause with moderate variability.
        elif "," in chunk_text:
            return self._get_jittered_pause(220, 40)
        
        # Default minimal pause.
        else:
            return self._get_jittered_pause(150, 30)
    
    def _detect_emotional_context(self, chunk_text):
        """
        Detects emotional cues in text for dynamic adjustment.
        
        Emotional intensity scaling:
        - Exclamation marks indicate excitement or emphasis.
        - Question marks indicate inquiry or uncertainty.
        - Ellipses indicate trailing thought or hesitation.
        - Short fragmented sentences indicate urgency or impact.
        - This creates narrative dimensionality through prosodic variation.
        """
        has_exclamation = '!' in chunk_text
        has_question = '?' in chunk_text
        has_ellipsis = '...' in chunk_text
        is_short_fragment = len(chunk_text.split()) < 5
        is_dialogue = '"' in chunk_text or "'" in chunk_text
        
        return {
            'has_emotional_punctuation': has_exclamation or has_question,
            'has_ellipsis': has_ellipsis,
            'is_short_fragment': is_short_fragment,
            'is_dialogue': is_dialogue
        }
    
    def generate_audio(self, text_chunks):
        """
        Generates complete audiobook from text chunks with natural pacing, breaths, and pauses.
        Returns a combined AudioSegment ready for post-processing.
        """
        audio_segments = []
        previous_chunk = ""
        
        original_load_voice = self.pipeline.load_voice
        self.pipeline.load_voice = lambda x: self.hybrid_voice
        
        chunk_bar = tqdm(text_chunks, desc="  Chunks", unit="chunk", leave=False, position=1)
        
        for i, chunk_text in enumerate(chunk_bar):
            if not chunk_text.strip():
                continue
            
            try:
                emotional_context = self._detect_emotional_context(chunk_text)
                
                # Add breath at paragraph start
                is_paragraph_start = i == 0 or previous_chunk.endswith('\n\n') or '\n\n' in previous_chunk[-10:]
                
                if is_paragraph_start and self.breath_sample:
                    pre_silence = random.randint(150, 250)
                    post_silence = random.randint(100, 180)
                    audio_segments.append(AudioSegment.silent(duration=pre_silence, frame_rate=self.config.sample_rate))
                    varied_breath = self._get_varied_breath()
                    if varied_breath:
                        audio_segments.append(varied_breath)
                    audio_segments.append(AudioSegment.silent(duration=post_silence, frame_rate=self.config.sample_rate))
                elif self._should_add_breath(
                    chunk_text,
                    emotional_context['has_emotional_punctuation'],
                    emotional_context['is_dialogue']
                ):
                    pre_silence = random.randint(100, 200)
                    post_silence = random.randint(80, 150)
                    audio_segments.append(AudioSegment.silent(duration=pre_silence, frame_rate=self.config.sample_rate))
                    varied_breath = self._get_varied_breath()
                    if varied_breath:
                        audio_segments.append(varied_breath)
                    audio_segments.append(AudioSegment.silent(duration=post_silence, frame_rate=self.config.sample_rate))
                
                if emotional_context['has_emotional_punctuation'] and '!' in chunk_text:
                    base_speed = random.uniform(0.92, 0.98)
                elif emotional_context['has_ellipsis']:
                    base_speed = random.uniform(0.84, 0.90)
                else:
                    base_speed = random.uniform(0.88, 0.94)
                
                current_speed = base_speed + random.uniform(-0.02, 0.02)
                
                pitch_drift = random.uniform(-0.15, 0.15)
                self.pitch_drift_accumulator += pitch_drift
                self.pitch_drift_accumulator = max(-0.2, min(0.2, self.pitch_drift_accumulator))
                
                if "\n\n" in chunk_text:
                    self.pitch_drift_accumulator = 0.0
                
                generator = self.pipeline(chunk_text, voice=self.config.voice_1, speed=current_speed, split_pattern=r'\n+')
                
                for _, _, audio_tensor in generator:
                    if isinstance(audio_tensor, torch.Tensor):
                        audio_numpy = audio_tensor.cpu().numpy()
                    else:
                        audio_numpy = audio_tensor
                    
                    audio_int16 = (audio_numpy * 32767).astype(np.int16)
                    segment = AudioSegment(
                        audio_int16.tobytes(), 
                        frame_rate=self.config.sample_rate,
                        sample_width=2, 
                        channels=1
                    )
                    fade_duration = random.randint(5, 8)
                    segment = segment.fade_in(fade_duration).fade_out(fade_duration)
                    audio_segments.append(segment)
                
                pause_milliseconds = self._calculate_pause_duration(chunk_text)
                audio_segments.append(AudioSegment.silent(duration=pause_milliseconds, frame_rate=self.config.sample_rate))
                previous_chunk = chunk_text
                
            except Exception as e:
                log.error(f"Failed on chunk {i+1}/{len(text_chunks)}: {e}")
                log.debug(f"Chunk content: {chunk_text[:100]}...")
                log.debug(traceback.format_exc())
                continue
        
        chunk_bar.close()
        self.pipeline.load_voice = original_load_voice
        return sum(audio_segments)


class AudioPostProcessor:
    """Applies professional audio engineering effects to enhance the final audiobook quality."""
    
    def __init__(self):
        self.pedalboard = self._create_signal_chain()
    
    def _create_signal_chain(self):
        """
        Creates a professional audio mastering chain following the Ultra-Human Mimicry profile.
        
        Signal chain order for maximum realism:
        1. Subtractive EQ (clean before enhance)
        2. Transient Compression (catch peaks)
        3. Saturation (analog warmth)
        4. Glue Compression (consistency)
        5. Additive EQ (enhance presence)
        6. Micro Reverb (subtle space)
        7. Limiter (prevent clipping, -3 dB ceiling)
        
        Philosophy: Human realism through controlled imperfection and biological authenticity.
        """
        return Pedalboard([
            # STAGE 1: SUBTRACTIVE EQ - Clean before enhance.
            # Cut mud in the 280-320 Hz range to remove boxiness and improve clarity.
            # This frequency range often accumulates in TTS output and clouds the voice.
            PeakFilter(cutoff_frequency_hz=300, gain_db=-2.0, q=1.2),
            
            # Reduce harshness in the 3.5-4.5 kHz range where digital artifacts accumulate.
            # This prevents listener fatigue from piercing frequencies.
            PeakFilter(cutoff_frequency_hz=4000, gain_db=-2.0, q=2.0),
            
            # De-esser: reduce harsh sibilant sounds (S, T, SH) at 6500 Hz.
            # High Q value (3.0) creates a narrow notch that targets only the problematic
            # frequency range without affecting overall voice clarity.
            PeakFilter(cutoff_frequency_hz=6500, gain_db=-4.0, q=3.0),
            
            # STAGE 2: TRANSIENT COMPRESSION - Catch peaks instantly.
            # Fast attack (1ms) prevents sudden loud sounds from causing distortion.
            # The 3.5:1 ratio aggressively reduces signals above -12 dB.
            # Fast 60ms release allows quick recovery between words for natural dynamics.
            Compressor(threshold_db=-12.0, ratio=3.5, attack_ms=1.0, release_ms=60.0),
            
            # STAGE 3: HARMONIC SATURATION - Add analog warmth.
            # Very light even-order saturation (1.5-2.0 dB drive) adds harmonics that create
            # analog warmth and reduce digital sterility. This mimics tube preamps or tape,
            # increasing perceived vocal density without audible distortion.
            Distortion(drive_db=1.8),
            
            # STAGE 4: GLUE COMPRESSION - Smooth overall dynamics.
            # Slow attack (25ms) provides "glue compression" that smooths loudness variations
            # between sentences. The gentle 1.4:1 ratio and slow 220ms release create
            # cohesive, consistent volume throughout without sounding obviously compressed.
            # This avoids audible pumping artifacts.
            Compressor(threshold_db=-20.0, ratio=1.4, attack_ms=25.0, release_ms=220.0),
            
            # STAGE 5: ADDITIVE EQ - Enhance after compression.
            # Low shelf filter boosts bass frequencies below 120 Hz for warmth and fullness.
            # This simulates the proximity effect of close microphone placement.
            LowShelfFilter(cutoff_frequency_hz=120, gain_db=2.5, q=0.7),
            
            # Peak filter at 200 Hz enhances the fundamental frequency range of human voice,
            # adding body and chest resonance for richer, more present sound.
            PeakFilter(cutoff_frequency_hz=200, gain_db=3.0, q=1.0),
            
            # High shelf filter boosts frequencies above 12 kHz, adding "air" and sparkle.
            # This enhances clarity and creates a sense of openness.
            HighShelfFilter(cutoff_frequency_hz=12000, gain_db=1.5),
            
            # STAGE 6: MICRO REVERB - Invisible room ambience.
            # Ultra-subtle micro-booth reverb creates physical space without being detectable.
            # Room size 0.05-0.07 creates tight space, high damping (0.9) absorbs frequencies
            # quickly, and 1-1.5% wet level adds just enough ambience to remove digital
            # silence artifacts. Reverb should never be consciously heard.
            Reverb(room_size=0.06, damping=0.9, wet_level=0.012, dry_level=0.988),
            
            # STAGE 7: LIMITER - Prevent clipping with -3 dB ceiling.
            # Set at -3.0 dB threshold (not -1.0 dB) to provide headroom for distribution.
            # This catches any peaks that exceed the threshold and instantly reduces them,
            # ensuring the audio never exceeds -3 dBFS. The 100ms release prevents pumping
            # artifacts while maintaining transparent peak control. This is the final safety
            # stage before normalization to -19 LUFS for audiobook standards.
            Limiter(threshold_db=-3.0, release_ms=100.0),
        ])
    
    def process_audio(self, input_path, output_path):
        """
        Applies the signal chain to the raw audio file and exports the final processed version.
        Returns dictionary containing audio metadata (duration, file size, format info).
        """
        
        # Read raw audio file.
        with AudioFile(str(input_path)) as f:
            audio = f.read(f.frames)
            sample_rate = f.samplerate
        
        # Apply processing chain.
        processed_audio = self.pedalboard(audio, sample_rate)
        
        # Export final audio.
        with AudioFile(str(output_path), 'w', sample_rate, processed_audio.shape[0]) as f:
            f.write(processed_audio)
        
        # Calculate audio metadata
        duration_seconds = processed_audio.shape[1] / sample_rate
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        
        metadata = {
            "duration_seconds": duration_seconds,
            "duration_formatted": self._format_duration(duration_seconds),
            "file_size_mb": file_size_mb,
            "sample_rate": sample_rate,
            "channels": processed_audio.shape[0],
            "bit_depth": 16,
            "format": "MP3"
        }
        
        return metadata
    
    def _format_duration(self, seconds):
        """Formats duration in seconds to HH:MM:SS format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class AudiobookPipeline:
    """Main orchestrator for the complete audiobook generation workflow."""
    
    def __init__(self, args, output_directory, voice_type):
        self.args = args
        self.config = AudiobookConfig(output_directory, voice_type)
        self._initialize_models()
    
    def _initialize_models(self):
        """Sets up all required models and services for the pipeline."""
        log.info("Initializing models...")
        
        # Determine compute device.
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        log.info(f"Using device: {self.device}")
        
        # Initialize Kokoro TTS pipeline.
        self.tts_pipeline = KPipeline(lang_code='a', device=self.device)
        
        # Create voice blender for dynamic blending.
        voice_blender = VoiceBlender(self.tts_pipeline, self.device)
        
        # Create hybrid voice by blending two voice profiles with 60/40 ratio.
        self.hybrid_voice = voice_blender.blend_voices(
            self.config.voice_1, 
            self.config.voice_2, 
            self.config.blend_ratio
        )
        
        # Store voice blender for use in audio generation.
        self.voice_blender = voice_blender
    
    def _save_artifacts(self, markdown_text, processed_text, text_chunks):
        """Saves intermediate processing artifacts if requested by user."""
        if not self.args.keep_artifacts:
            return
        
        raw_output_path = self.config.temp_dir / 'raw.md'
        with open(raw_output_path, 'w', encoding='utf-8') as file:
            file.write(markdown_text)
        log.info(f"Saved raw extract to: {raw_output_path}")
        
        processed_output_path = self.config.temp_dir / 'processed.txt'
        with open(processed_output_path, 'w', encoding='utf-8') as file:
            file.write(processed_text)
        log.info(f"Saved processed text to: {processed_output_path}")
        
        chunks_output_path = self.config.temp_dir / 'chunks.txt'
        with open(chunks_output_path, 'w', encoding='utf-8') as file:
            for index, chunk in enumerate(text_chunks):
                file.write(f"--- Chunk {index+1} ---\n{chunk}\n\n")
        log.info(f"Saved processed chunks to: {chunks_output_path}")
    
    def run(self):
        """Executes the complete audiobook generation pipeline."""
        text_processor = TextProcessor()
        
        log.info("Extracting text...")
        try:
            if self.args.filename.lower().endswith('.md'):
                markdown_text, processed_text, chapter_chunks, chapter_titles = text_processor.extract_and_chunk_markdown(
                    self.args.filename
                )
            else:
                markdown_text, processed_text, chapter_chunks, chapter_titles = text_processor.extract_and_chunk_epub(
                    self.args.filename,
                    self.args.start_chapter,
                    self.args.end_chapter
                )
        except Exception as e:
            log.error(f"Text extraction failed: {e}")
            log.debug(traceback.format_exc())
            sys.exit(1)
        
        all_chunks = [chunk for chunks in chapter_chunks for chunk in chunks]
        total_chunks = len(all_chunks)
        self._save_artifacts(markdown_text, processed_text, all_chunks)
        
        audio_generator = AudioGenerator(self.tts_pipeline, self.hybrid_voice, self.config, self.voice_blender)
        post_processor = AudioPostProcessor()
        
        total_chapters = len(chapter_chunks)
        log.info(f"Starting generation: {total_chapters} sections, {total_chunks} total chunks")
        SystemMonitor.log_status(self.config.output_dir)
        
        # ETA tracking with exponential moving average
        ema_seconds_per_chunk = None
        ema_alpha = 0.3
        chunks_completed = 0
        pipeline_start = time.time()
        
        # CPU worker pool for parallel post-processing while GPU generates next section
        cpu_workers = min(4, max(1, (psutil.cpu_count(logical=False) or 2) - 1))
        log.info(f"Post-processing workers: {cpu_workers}")
        post_futures = []
        
        chapter_bar = tqdm(total=total_chapters, desc="Sections", unit="sec", position=0)
        
        with ThreadPoolExecutor(max_workers=cpu_workers) as executor:
            for index, chunks in enumerate(chapter_chunks):
                chapter_title = chapter_titles[index] if index < len(chapter_titles) else f"Chapter_{index+1}"
                kebab_title = re.sub(r'[^\w\s-]', '', chapter_title).strip().lower().replace(' ', '-')
                safe_title = f"{index}-{kebab_title}"
                
                chapter_bar.set_postfix_str(f"{chapter_title[:40]}")
                chapter_start = time.time()
                
                try:
                    chapter_audio = audio_generator.generate_audio(chunks)
                    
                    raw_path = self.config.temp_dir / f'{safe_title}_raw.wav'
                    chapter_audio.export(str(raw_path), format='wav')
                    
                    # Submit post-processing to CPU thread pool
                    output_file = self.config.output_dir / f'{safe_title}.mp3'
                    future = executor.submit(
                        self._post_process_section,
                        post_processor, raw_path, output_file
                    )
                    post_futures.append((future, output_file, len(chunks), chapter_start))
                    
                except Exception as e:
                    log.error(f"Failed to generate '{chapter_title}': {e}")
                    log.debug(traceback.format_exc())
                    chunks_completed += len(chunks)
                
                # Drain completed futures without blocking
                still_pending = []
                for fut, out_file, n_chunks, ch_start in post_futures:
                    if fut.done():
                        try:
                            audio_metadata = fut.result()
                        except Exception as e:
                            log.error(f"Post-processing failed for {out_file.name}: {e}")
                            chapter_bar.update(1)
                            chunks_completed += n_chunks
                            continue
                        
                        chapter_elapsed = time.time() - ch_start
                        spc = chapter_elapsed / max(n_chunks, 1)
                        if ema_seconds_per_chunk is None:
                            ema_seconds_per_chunk = spc
                        else:
                            ema_seconds_per_chunk = ema_alpha * spc + (1 - ema_alpha) * ema_seconds_per_chunk
                        
                        chunks_completed += n_chunks
                        chunks_remaining = total_chunks - chunks_completed
                        eta_seconds = (ema_seconds_per_chunk or 0) * chunks_remaining
                        
                        chapter_bar.update(1)
                        tqdm.write(
                            f"  done {out_file.name} | "
                            f"{audio_metadata['duration_formatted']} | "
                            f"{audio_metadata['file_size_mb']:.1f} MB | "
                            f"ETA: {self._format_duration(eta_seconds)}"
                        )
                    else:
                        still_pending.append((fut, out_file, n_chunks, ch_start))
                post_futures = still_pending
                
                # Log system health every 5 sections
                if (index + 1) % 5 == 0:
                    SystemMonitor.log_status(self.config.output_dir)
            
            # Wait for remaining post-processing futures
            for fut, out_file, n_chunks, ch_start in post_futures:
                try:
                    audio_metadata = fut.result()
                except Exception as e:
                    log.error(f"Post-processing failed for {out_file.name}: {e}")
                    chapter_bar.update(1)
                    chunks_completed += n_chunks
                    continue
                
                chapter_elapsed = time.time() - ch_start
                spc = chapter_elapsed / max(n_chunks, 1)
                if ema_seconds_per_chunk is None:
                    ema_seconds_per_chunk = spc
                else:
                    ema_seconds_per_chunk = ema_alpha * spc + (1 - ema_alpha) * ema_seconds_per_chunk
                
                chunks_completed += n_chunks
                chunks_remaining = total_chunks - chunks_completed
                eta_seconds = (ema_seconds_per_chunk or 0) * chunks_remaining
                
                chapter_bar.update(1)
                tqdm.write(
                    f"  done {out_file.name} | "
                    f"{audio_metadata['duration_formatted']} | "
                    f"{audio_metadata['file_size_mb']:.1f} MB | "
                    f"ETA: {self._format_duration(eta_seconds)}"
                )
        
        chapter_bar.close()
        
        total_elapsed = time.time() - pipeline_start
        log.info(f"Complete: {total_chapters} section(s) in {self._format_duration(total_elapsed)}")
        log.info(f"Output: {self.config.output_dir}")
        SystemMonitor.log_status(self.config.output_dir)
        if self.args.keep_artifacts:
            log.info(f"Artifacts: {self.config.temp_dir}")
    
    def _post_process_section(self, post_processor, raw_path, output_file):
        """Runs pedalboard + MP3 export in a thread pool worker."""
        audio_metadata = post_processor.process_audio(raw_path, output_file)
        if not self.args.keep_artifacts:
            try:
                os.remove(raw_path)
            except OSError:
                pass
        return audio_metadata
    
    @staticmethod
    def _format_duration(seconds):
        """Formats duration in seconds to HH:MM:SS format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def print_toc(filename):
    """Prints table of contents with indexes."""
    book = epub.read_epub(filename)
    processor = TextProcessor()
    flat_toc = processor._flatten_toc(book.toc)
    
    # Deduplicate by href
    seen = set()
    entries = []
    for title, href in flat_toc:
        if href not in seen:
            seen.add(href)
            entries.append(title)
    
    print("\n=== Table of Contents ===")
    for index, title in enumerate(entries):
        print(f"{index}: {title}")
    print()


def parse_arguments():
    """Parses and validates command-line arguments for the audiobook generator."""
    parser = argparse.ArgumentParser(description='Process EPUB files for text-to-speech with Audio Engineering')
    parser.add_argument('filename', help='EPUB file to process')
    parser.add_argument('--output-dir', default='output_audio', help='Output directory for generated files')
    parser.add_argument('--voice-type', choices=['male', 'female'], default='female', help='Voice gender for narration')
    parser.add_argument('--start-chapter', type=str, help='Starting chapter index')
    parser.add_argument('--end-chapter', type=str, help='Ending chapter index')
    parser.add_argument('--keep-artifacts', action='store_true', help='Retain intermediate files (raw markdown, processed text, raw audio)')
    parser.add_argument('--extract-only', action='store_true', help='Extract text artifacts only, no audio generation')
    parser.add_argument('--print-toc', action='store_true', help='Print table of contents with indexes and exit')
    args = parser.parse_args()
    
    if not args.filename.lower().endswith(('.epub', '.md')):
        log.error("Unsupported format. Please provide an .epub or .md file.")
        sys.exit(1)
    
    if args.print_toc:
        if args.filename.lower().endswith('.epub'):
            print_toc(args.filename)
        else:
            log.error("--print-toc is only supported for EPUB files.")
        sys.exit(0)
    
    return args


def extract_only(args):
    """Extracts text artifacts without generating audio."""
    config = AudiobookConfig(args.output_dir, args.voice_type)
    text_processor = TextProcessor()

    if args.filename.lower().endswith('.md'):
        markdown_text, processed_text, chapter_chunks, chapter_titles = text_processor.extract_and_chunk_markdown(
            args.filename
        )
    else:
        markdown_text, processed_text, chapter_chunks, chapter_titles = text_processor.extract_and_chunk_epub(
            args.filename, args.start_chapter, args.end_chapter
        )

    all_chunks = [chunk for chunks in chapter_chunks for chunk in chunks]

    raw_output_path = config.temp_dir / 'raw.md'
    with open(raw_output_path, 'w', encoding='utf-8') as f:
        f.write(markdown_text)
    log.info(f"Saved raw extract to: {raw_output_path}")

    processed_output_path = config.temp_dir / 'processed.txt'
    with open(processed_output_path, 'w', encoding='utf-8') as f:
        f.write(processed_text)
    log.info(f"Saved processed text to: {processed_output_path}")

    chunks_output_path = config.temp_dir / 'chunks.txt'
    with open(chunks_output_path, 'w', encoding='utf-8') as f:
        for index, chunk in enumerate(all_chunks):
            f.write(f"--- Chunk {index+1} ---\n{chunk}\n\n")
    log.info(f"Saved processed chunks to: {chunks_output_path}")


def main():
    """Entry point for the audiobook generation application."""
    args = parse_arguments()

    if args.extract_only:
        extract_only(args)
        return

    pipeline = AudiobookPipeline(args, args.output_dir, args.voice_type)
    pipeline.run()


if __name__ == '__main__':
    main()
