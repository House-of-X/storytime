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
import chevron
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
        
        self.blend_ratio = 0.5
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
    
    def blend_voices(self, voice_1_name, voice_2_name, ratio=0.5):
        """
        Loads two voice tensors and blends them mathematically to create a hybrid voice.
        This prevents ghosting artifacts and crashes during audio generation.
        """
        print(f"--- Blending Voices: {voice_1_name} ({ratio*100}%) + {voice_2_name} ({(1-ratio)*100}%) ---")
        
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
            print(f"Warning: Could not blend voices automatically ({e}). Using {voice_1_name} only.")
            return voice_1_name


class TextProcessor:
    """Handles text extraction, normalization, and cleaning operations."""
    
    def __init__(self, bedrock_client=None):
        self.bedrock_client = bedrock_client
        self.prompt_template = self._load_prompt_template()
    
    def _load_prompt_template(self):
        """Loads the LLM prompt template from file."""
        template_path = Path('prompts/clean_text.prompt')
        with open(template_path, 'r', encoding='utf-8') as file:
            return file.read()
    
    def normalize_text(self, text):
        """
        Applies basic text normalization to remove formatting artifacts and improve readability.
        Removes hyphenated line breaks, page numbers, figures, and excessive whitespace.
        """
        text = re.sub(r'-\n', '', text)
        text = re.sub(r'(?<![.!?:])\n(?!\n)', ' ', text)
        text = re.sub(r'¡\s*', '', text)
        text = re.sub(r'~~[^~]+~~', '', text)
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'Figure \d+\.\d+[^\n]*', '', text)
        text = re.sub(r' +', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()
    
    def clean_text_with_llm(self, text):
        """
        Uses Claude LLM to enhance text for audiobook narration by expanding abbreviations,
        adding transitions, and removing print-only artifacts while maintaining structure.
        """
        if len(text.strip()) < 10:
            return text
        
        prompt = chevron.render(self.prompt_template, {'text': text})
        
        try:
            response = self.bedrock_client.invoke_model(
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
    
    def extract_and_chunk_pdf(self, filename, start_page, end_page, use_language_model=True):
        """
        Extracts text from PDF, splits into manageable chunks, and optionally cleans with LLM.
        Returns a list of processed text chunks ready for audio generation.
        """
        print(f"--- Extracting PDF: {filename} ---")
        
        # Extract markdown from PDF using specified page range.
        markdown_text = pymupdf4llm.to_markdown(
            filename,
            pages=list(range(start_page - 1, end_page)) if end_page else None
        )
        
        # Remove code blocks and inline code that shouldn't be narrated.
        markdown_text = re.sub(r'```[\s\S]*?```', '', markdown_text)
        markdown_text = re.sub(r'`[^`]+`', '', markdown_text)
        
        # Split by markdown headers to preserve document structure.
        header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[('#', 'h1'), ('##', 'h2')])
        header_splits = header_splitter.split_text(markdown_text)
        
        # Further split into smaller chunks for better TTS processing.
        final_chunks = []
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=650, chunk_overlap=0, separators=["\n\n", ". ", "! ", "? ", "; "]
        )
        
        print("--- Processing Text Chunks ---")
        for document in header_splits:
            normalized = self.normalize_text(document.page_content)
            if use_language_model and self.bedrock_client:
                cleaned = self.clean_text_with_llm(normalized)
            else:
                cleaned = normalized
            final_chunks.extend(text_splitter.split_text(cleaned))
        
        return markdown_text, final_chunks


class AudioGenerator:
    """Generates audio from text chunks using TTS pipeline with natural pauses and breaths."""
    
    def __init__(self, pipeline, hybrid_voice, config):
        self.pipeline = pipeline
        self.hybrid_voice = hybrid_voice
        self.config = config
        self.breath_sample = self._load_breath_sample()
    
    def _load_breath_sample(self):
        """
        Loads and processes the breath sample audio for natural pauses.
        
        Audio processing steps:
        - Resampling to 24000 Hz ensures the breath matches the TTS output sample rate,
          preventing pitch shifts or timing issues when concatenating audio segments.
        - Applying -32 dB gain reduces the breath volume to a subtle, natural level that
          doesn't overpower the speech. This mimics real human breathing during narration.
        - Fade in/out (100ms each) smooths the breath edges to avoid clicks or pops that
          occur when abruptly starting or stopping audio signals.
        """
        try:
            breath_file = 'templates/male-inhale.mp3' if self.config.voice_type == 'male' else 'templates/female-inhale.mp3'
            breath = AudioSegment.from_mp3(breath_file)
            breath = breath.set_frame_rate(self.config.sample_rate).apply_gain(-32).fade_in(100).fade_out(100)
            return breath
        except:
            print("Warning: Breath sample not found. Skipping breaths.")
            return None
    
    def _get_jittered_pause(self, base_milliseconds):
        """Adds random variation to pause duration for more natural speech rhythm."""
        return base_milliseconds + random.randint(-50, 50)
    
    def _should_add_breath(self, chunk_index, chunk_text, previous_chunk):
        """
        Determines whether to insert a breath sound based on context.
        Higher probability for paragraph breaks and long sentences.
        """
        if not self.breath_sample:
            return False
        
        word_count = len(chunk_text.split())
        is_new_paragraph = chunk_index > 0 and "\n\n" in previous_chunk
        is_long_sentence = word_count > 20
        
        breath_chance = 0.85 if is_new_paragraph else (0.65 if is_long_sentence else 0.40)
        return random.random() < breath_chance
    
    def _calculate_pause_duration(self, chunk_text):
        """Calculates appropriate pause duration based on punctuation and content."""
        if "\n\n" in chunk_text:
            return self._get_jittered_pause(1000)
        elif chunk_text.rstrip().endswith(('!', '?')):
            return self._get_jittered_pause(800)
        elif chunk_text.rstrip().endswith('.'):
            return self._get_jittered_pause(500)
        elif "," in chunk_text:
            return self._get_jittered_pause(200)
        else:
            return self._get_jittered_pause(150)
    
    def generate_audio(self, text_chunks):
        """
        Generates complete audiobook from text chunks with natural pacing, breaths, and pauses.
        Returns a combined AudioSegment ready for post-processing.
        """
        print(f"--- Generating Audio ({len(text_chunks)} chunks) ---")
        
        audio_segments = []
        
        # Temporarily override the pipeline's voice loading to use our hybrid voice.
        original_load_voice = self.pipeline.load_voice
        self.pipeline.load_voice = lambda x: self.hybrid_voice
        
        for i, chunk_text in enumerate(text_chunks):
            if not chunk_text.strip():
                continue
            
            # Add breath sound before chunk if appropriate.
            if self._should_add_breath(i, chunk_text, text_chunks[i-1] if i > 0 else ""):
                pre_silence = random.randint(120, 180)
                post_silence = random.randint(80, 120)
                audio_segments.append(AudioSegment.silent(duration=pre_silence, frame_rate=self.config.sample_rate))
                audio_segments.append(self.breath_sample)
                audio_segments.append(AudioSegment.silent(duration=post_silence, frame_rate=self.config.sample_rate))
            
            # Generate speech with slight speed variation for natural delivery.
            base_speed = random.uniform(0.78, 0.84)
            current_speed = base_speed + random.uniform(-0.02, 0.02)
            generator = self.pipeline(chunk_text, voice=self.config.voice_1, speed=current_speed, split_pattern=r'\n+')
            
            # Convert generated audio tensors to AudioSegment format.
            for _, _, audio_tensor in generator:
                if isinstance(audio_tensor, torch.Tensor):
                    audio_numpy = audio_tensor.cpu().numpy()
                else:
                    audio_numpy = audio_tensor
                
                # Convert float32 normalized audio (-1.0 to 1.0) to int16 PCM format.
                # Multiplying by 32767 scales the float range to the int16 range (-32768 to 32767).
                # This is the standard PCM audio format used by most audio libraries and hardware.
                audio_int16 = (audio_numpy * 32767).astype(np.int16)
                
                # Create AudioSegment with mono channel (1) and 16-bit sample width (2 bytes).
                # Sample rate of 24000 Hz provides good quality for speech while keeping file sizes reasonable.
                segment = AudioSegment(
                    audio_int16.tobytes(), 
                    frame_rate=self.config.sample_rate,
                    sample_width=2, 
                    channels=1
                )
                audio_segments.append(segment)
            
            # Add contextual pause after chunk.
            pause_milliseconds = self._calculate_pause_duration(chunk_text)
            audio_segments.append(AudioSegment.silent(duration=pause_milliseconds, frame_rate=self.config.sample_rate))
            
            if i % 5 == 0:
                print(f"Generated {i}/{len(text_chunks)} chunks...")
        
        # Restore original voice loading method.
        self.pipeline.load_voice = original_load_voice
        
        # Combine all audio segments into single track.
        return sum(audio_segments)


class AudioPostProcessor:
    """Applies professional audio engineering effects to enhance the final audiobook quality."""
    
    def __init__(self):
        self.pedalboard = self._create_signal_chain()
    
    def _create_signal_chain(self):
        """
        Creates a professional audio mastering chain that processes the raw TTS output
        through multiple stages to achieve broadcast-quality sound.
        
        The signal flow follows standard audio engineering practices:
        EQ → Saturation → Compression → Spatial Effects → Limiting
        """
        return Pedalboard([
            # Low shelf filter boosts bass frequencies below 120 Hz by 2.5 dB.
            # This simulates the proximity effect (bass boost when close to a microphone)
            # and adds warmth and fullness to the voice. The Q factor of 0.7 creates
            # a gentle, natural-sounding slope rather than an abrupt frequency change.
            LowShelfFilter(cutoff_frequency_hz=120, gain_db=2.5, q=0.7),
            
            # Peak filter at 200 Hz enhances the fundamental frequency range of human voice.
            # This adds body and chest resonance, making the voice sound richer and more present.
            # A Q of 1.0 creates a moderate bandwidth boost centered at 200 Hz.
            PeakFilter(cutoff_frequency_hz=200, gain_db=3.0, q=1.0),
            
            # De-esser reduces harsh sibilant sounds (S, T, SH) at 6500 Hz by 4 dB.
            # High Q value (3.0) creates a narrow notch that targets only the problematic
            # frequency range without affecting overall voice clarity. This prevents listener
            # fatigue from piercing high frequencies.
            PeakFilter(cutoff_frequency_hz=6500, gain_db=-4.0, q=3.0),
            
            # Subtle harmonic distortion adds even-order harmonics that create analog warmth.
            # At 2 dB drive, this mimics the pleasant saturation of tube preamps or tape,
            # adding richness without audible distortion. This makes digital TTS sound
            # less sterile and more organic.
            Distortion(drive_db=2.0),
            
            # Fast attack compressor (1ms) catches transient peaks instantly, preventing
            # sudden loud sounds from causing distortion. The 4:1 ratio aggressively reduces
            # signals above -12 dB, while the fast 50ms release allows the compressor to
            # recover quickly between words, maintaining natural dynamics.
            Compressor(threshold_db=-12.0, ratio=4.0, attack_ms=1.0, release_ms=50.0),
            
            # Slow attack compressor (20ms) provides "glue compression" that smooths overall
            # loudness variations between sentences. The gentle 1.5:1 ratio and slow 200ms
            # release create cohesive, consistent volume throughout the audiobook without
            # sounding obviously compressed. This is the secret to professional-sounding audio.
            Compressor(threshold_db=-20.0, ratio=1.5, attack_ms=20.0, release_ms=200.0),
            
            # High shelf filter boosts frequencies above 12 kHz by 1.5 dB, adding "air"
            # and sparkle to the voice. This enhances clarity and creates a sense of openness,
            # making the audio sound more expensive and professionally recorded.
            HighShelfFilter(cutoff_frequency_hz=12000, gain_db=1.5),
            
            # Reverb simulates a small recording booth with minimal reflections.
            # Room size of 0.08 creates a tight space, damping of 0.9 absorbs high frequencies
            # quickly (like acoustic treatment), and 3% wet level adds just enough ambience
            # to prevent the "in your head" feeling of completely dry audio while maintaining
            # clarity and intelligibility.
            Reverb(room_size=0.08, damping=0.9, wet_level=0.03, dry_level=0.97),
            
            # Limiter is the final safety stage that prevents clipping (digital distortion).
            # Set at -1.0 dB threshold, it catches any peaks that exceed this level and
            # instantly reduces them, ensuring the audio never exceeds 0 dBFS (digital maximum).
            # This also maximizes perceived loudness by allowing the entire mix to be pushed
            # closer to the digital ceiling without distortion. The 100ms release prevents
            # pumping artifacts while maintaining transparent peak control.
            Limiter(threshold_db=-1.0, release_ms=100.0),
        ])
    
    def process_audio(self, input_path, output_path):
        """
        Applies the signal chain to the raw audio file and exports the final processed version.
        Returns the path to the processed audio file.
        """
        print("--- Applying Audio Engineering (Pedalboard) ---")
        
        # Read raw audio file.
        with AudioFile(str(input_path)) as f:
            audio = f.read(f.frames)
            sample_rate = f.samplerate
        
        # Apply processing chain.
        processed_audio = self.pedalboard(audio, sample_rate)
        
        # Export final audio.
        with AudioFile(str(output_path), 'w', sample_rate, processed_audio.shape[0]) as f:
            f.write(processed_audio)
        
        return output_path


class AudiobookPipeline:
    """Main orchestrator for the complete audiobook generation workflow."""
    
    def __init__(self, args, output_directory, voice_type):
        self.args = args
        self.config = AudiobookConfig(output_directory, voice_type)
        self._initialize_models()
    
    def _initialize_models(self):
        """Sets up all required models and services for the pipeline."""
        print("--- Initializing Models ---")
        
        # Determine compute device.
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Initialize AWS Bedrock client for LLM processing.
        self.bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")
        
        # Initialize Kokoro TTS pipeline.
        self.tts_pipeline = KPipeline(lang_code='a', device=self.device)
        
        # Create hybrid voice by blending two voice profiles.
        voice_blender = VoiceBlender(self.tts_pipeline, self.device)
        self.hybrid_voice = voice_blender.blend_voices(
            self.config.voice_1, 
            self.config.voice_2, 
            self.config.blend_ratio
        )
    
    def _save_artifacts(self, markdown_text, text_chunks):
        """Saves intermediate processing artifacts if requested by user."""
        if not self.args.keep_artifacts:
            return
        
        # Save extracted markdown.
        markdown_output_path = self.config.temp_dir / 'extracted_text.md'
        with open(markdown_output_path, 'w', encoding='utf-8') as file:
            file.write(markdown_text)
        print(f"Saved extracted markdown to: {markdown_output_path}")
        
        # Save processed chunks.
        chunks_output_path = self.config.temp_dir / 'processed_chunks.txt'
        with open(chunks_output_path, 'w', encoding='utf-8') as file:
            for index, chunk in enumerate(text_chunks):
                file.write(f"--- Chunk {index+1} ---\n{chunk}\n\n")
        print(f"Saved processed chunks to: {chunks_output_path}")
    
    def run(self):
        """Executes the complete audiobook generation pipeline from PDF to final audio."""
        # Extract and process text from PDF.
        text_processor = TextProcessor(self.bedrock_client if not self.args.no_llm else None)
        markdown_text, text_chunks = text_processor.extract_and_chunk_pdf(
            self.args.filename,
            self.args.start_page,
            self.args.end_page,
            use_language_model=not self.args.no_llm
        )
        
        # Save intermediate artifacts if requested.
        self._save_artifacts(markdown_text, text_chunks)
        
        # Generate raw audio from text chunks.
        audio_generator = AudioGenerator(self.tts_pipeline, self.hybrid_voice, self.config)
        full_audio = audio_generator.generate_audio(text_chunks)
        
        # Export raw audio to temporary file.
        raw_path = self.config.temp_dir / 'raw_speech.wav'
        full_audio.export(str(raw_path), format='wav')
        if self.args.keep_artifacts:
            print(f"Saved raw audio to: {raw_path}")
        
        # Apply post-processing effects.
        post_processor = AudioPostProcessor()
        output_file = self.config.output_dir / 'final_audiobook.mp3'
        post_processor.process_audio(raw_path, output_file)
        
        # Clean up temporary files unless artifacts are requested.
        if not self.args.keep_artifacts:
            try:
                os.remove(raw_path)
            except:
                pass
        else:
            print(f"Artifacts retained in: {self.config.temp_dir}")
        
        print(f"DONE! File saved to: {output_file}")


def parse_arguments():
    """Parses and validates command-line arguments for the audiobook generator."""
    parser = argparse.ArgumentParser(description='Process PDF files for text-to-speech with Audio Engineering')
    parser.add_argument('filename', help='PDF file to process')
    parser.add_argument('--output-dir', default='output_audio', help='Output directory for generated files')
    parser.add_argument('--voice-type', choices=['male', 'female'], default='female', help='Voice gender for narration')
    parser.add_argument('--start-page', type=int, default=1, help='Starting page number')
    parser.add_argument('--end-page', type=int, help='Ending page number')
    parser.add_argument('--no-llm', action='store_true', help='Skip Claude cleaning (faster, less expensive)')
    parser.add_argument('--keep-artifacts', action='store_true', help='Retain intermediate files (PDF markdown, LLM output, raw audio)')
    args = parser.parse_args()
    
    # Validate file format.
    if not args.filename.lower().endswith('.pdf'):
        print(f"Error: Unsupported format. Please provide a .pdf file.", file=sys.stderr)
        sys.exit(1)
    
    return args


def main():
    """Entry point for the audiobook generation application."""
    args = parse_arguments()
    pipeline = AudiobookPipeline(args, args.output_dir, args.voice_type)
    pipeline.run()


if __name__ == '__main__':
    main()
