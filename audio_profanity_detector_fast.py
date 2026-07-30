"""
Audio Profanity Detector (Faster-Whisper) - Detects profanity in audio using faster-whisper
Optimized for GPU acceleration with automatic fallback to CPU on CUDA errors.
"""

import subprocess
import tempfile
import shutil
import sys
import time
from pathlib import Path
from typing import List, Tuple

from profanity_words import PROFANITY_WORDS


class MissingBinaryError(RuntimeError):
    """Raised when required system binaries (ffmpeg/ffprobe) are not available."""


class AudioProfanityDetectorFast:
    """Detects profanity in audio using faster-whisper with optional enhancements"""

    PROFANITY_WORDS_SET = set(PROFANITY_WORDS)
    PROFANITY_PHRASES = [p for p in PROFANITY_WORDS if ' ' in p]

    _MODEL_ORDER = ['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3']

    def __init__(self,
                 model_size: str = 'base',
                 phrase_gap: float = 1.5,
                 dialog_enhance: bool = False,
                 dump_transcript_path: str = None,
                 min_wpm: float = 40.0,
                 auto_upgrade: bool = False):
        self.model_size = model_size
        self.phrase_gap = phrase_gap
        self.dialog_enhance = dialog_enhance
        self.dump_transcript_path = dump_transcript_path
        self.min_wpm = min_wpm
        self.auto_upgrade = auto_upgrade
        self._upgraded_once = False
        self._cuda_failed = False
        self.whisper_model = None
        self._device = self._detect_best_device()
        self._init_whisper()

    def _detect_best_device(self) -> str:
        """Detecta el mejor dispositivo disponible: CUDA > DirectML > CPU"""
        print("  [INFO] Detecting best available device...")
        
        # 1. Intentar CUDA
        try:
            import torch
            if torch.cuda.is_available():
                print(f"  [OK] CUDA detected: {torch.cuda.get_device_name(0)}")
                return 'cuda'
        except:
            pass
        
        # 2. Intentar DirectML (Windows)
        try:
            import torch_directml
            if torch_directml.is_available():
                print("  [OK] DirectML detected (Windows GPU acceleration)")
                return 'dml'
        except:
            pass
        
        # 3. Fallback a CPU
        print("  [INFO] No GPU acceleration found. Using CPU (slower).")
        return 'cpu'

    def _init_whisper(self):
        """Initialize faster-whisper model with the best available device"""
        try:
            from faster_whisper import WhisperModel
            
            # Si ya marcamos que CUDA falló, forzar CPU
            if self._cuda_failed:
                print("  [WARN] CUDA previously failed. Forcing CPU mode.")
                self._device = 'cpu'
            
            # Lista de compute types a probar
            if self._device == 'cuda':
                compute_types = ['float16', 'int8_float16', 'int8', 'float32']
            elif self._device == 'dml':
                compute_types = ['float16', 'int8_float16', 'int8', 'float32']
            else:  # CPU
                compute_types = ['int8_float16', 'int8', 'float32']
            
            print(f"  Loading faster-whisper model: {self.model_size}...")
            print(f"  Device: {self._device}")
            
            model_loaded = False
            last_error = None
            
            for compute_type in compute_types:
                try:
                    print(f"    Trying compute_type={compute_type}...")
                    self.whisper_model = WhisperModel(
                        self.model_size,
                        device=self._device,
                        compute_type=compute_type,
                        cpu_threads=8 if self._device == 'cpu' else 0,
                        num_workers=4 if self._device == 'cpu' else 1
                    )
                    print(f"  [OK] Faster-whisper loaded! (device={self._device}, compute_type={compute_type})")
                    model_loaded = True
                    break
                except Exception as e:
                    print(f"    [WARN] Error with compute_type={compute_type}: {e}")
                    last_error = e
                    continue
            
            # Si ningún compute_type funcionó, intentar con CPU como último recurso
            if not model_loaded:
                print("  [WARN] GPU initialization failed. Falling back to CPU...")
                self._device = 'cpu'
                for compute_type in ['int8_float16', 'int8', 'float32']:
                    try:
                        self.whisper_model = WhisperModel(
                            self.model_size,
                            device='cpu',
                            compute_type=compute_type,
                            cpu_threads=8,
                            num_workers=4
                        )
                        print(f"  [OK] Faster-whisper loaded on CPU (compute_type={compute_type})")
                        model_loaded = True
                        break
                    except Exception as e:
                        print(f"    [WARN] CPU error: {e}")
                        continue
                
                if not model_loaded:
                    raise RuntimeError(f"Could not initialize faster-whisper. Last error: {last_error}")
                
        except ImportError:
            raise ImportError(
                "faster-whisper not installed. Install with: pip install faster-whisper"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize faster-whisper: {e}")

    def _transcribe_with_fallback(self, audio_path: Path):
        """
        Intenta transcribir con el dispositivo actual. Si falla por CUDA,
        cambia a CPU y reintenta automáticamente.
        """
        try:
            # Primero, intentar con el dispositivo actual
            segments, info = self.whisper_model.transcribe(
                str(audio_path),
                beam_size=5,
                word_timestamps=True,
                language='es'
            )
            return segments, info
        except RuntimeError as e:
            error_msg = str(e).lower()
            # Si el error es por CUDA (cublas, cuda, etc.)
            if "cublas" in error_msg or "cuda" in error_msg or "library" in error_msg:
                print(f"  [WARN] CUDA runtime error detected: {e}")
                print(f"  [INFO] Automatically switching to CPU and retrying...")
                
                # Marcar que CUDA falló
                self._cuda_failed = True
                self._device = 'cpu'
                # Re-inicializar el modelo en CPU
                self._init_whisper()
                
                # Reintentar transcripción en CPU
                print(f"  [WAIT] Retrying transcription on CPU...")
                segments, info = self.whisper_model.transcribe(
                    str(audio_path),
                    beam_size=5,
                    word_timestamps=True,
                    language='es'
                )
                return segments, info
            else:
                # Otro tipo de error, no relacionado con CUDA
                raise

    def detect(self, video_path: Path) -> List[Tuple[float, float, str]]:
        """Detect profanity in audio."""
        if not self.whisper_model:
            return []

        self._ensure_media_binaries()
        
        temp_dir = Path(tempfile.mkdtemp())
        profanity_segments = []
        
        try:
            # Get video duration
            try:
                duration_cmd = [
                    'ffprobe', '-v', 'error',
                    '-show_entries', 'format=duration',
                    '-of', 'default=noprint_wrappers=1:nokey=1',
                    str(video_path)
                ]
                duration_result = subprocess.run(duration_cmd, capture_output=True, text=True, check=True)
                duration = float(duration_result.stdout.strip())
                print(f"  Video duration: {duration/60:.1f} minutes")
            except:
                duration = None
            
            # Extract audio
            print(f"  Extracting audio from video{' (dialog enhance)' if self.dialog_enhance else ''}...")
            audio_path = temp_dir / 'audio.wav'
            filter_chain = None
            if self.dialog_enhance:
                filter_chain = 'highpass=f=200,lowpass=f=3500,dynaudnorm=f=75,volume=1.3'
            cmd = ['ffmpeg', '-i', str(video_path)]
            if filter_chain:
                cmd += ['-af', filter_chain]
            cmd += [
                '-ar', '16000',
                '-ac', '1',
                '-loglevel', 'error',
                '-y', str(audio_path)
            ]
            subprocess.run(cmd, capture_output=True, check=True)
            print(f"  [OK] Audio extracted")
            
            # Transcribe with fallback
            print(f"  Transcribing audio with faster-whisper ({self.model_size} model)...")
            if duration:
                # Estimar tiempo según dispositivo
                if self._device in ['cuda', 'dml']:
                    est_time = duration / 30
                else:
                    est_time = duration / 5
                print(f"  [WAIT] Estimated time: ~{est_time:.1f} seconds for {duration/60:.1f} min video")
            
            start_time = time.time()
            
            # Usar el método con fallback
            segments, info = self._transcribe_with_fallback(audio_path)
            
            all_words = []
            for segment in segments:
                for word in segment.words:
                    all_words.append(word)
            
            elapsed = time.time() - start_time
            print(f"  [OK] Transcription complete in {elapsed:.1f}s ({info.duration/elapsed:.1f}x real-time)")

            # WPM diagnostic
            if info.duration and info.duration > 0:
                wpm = len(all_words) / (info.duration / 60.0)
                print(f"  Transcript stats: {len(all_words)} words, {wpm:.1f} WPM")
                if wpm < self.min_wpm and self.auto_upgrade and not self._upgraded_once:
                    next_model = self._next_model(self.model_size)
                    if next_model:
                        print(f"  [INFO] Auto-upgrade enabled: retrying with larger model '{next_model}'...")
                        self.model_size = next_model
                        self._upgraded_once = True
                        self._init_whisper()
                        return self._retry_transcribe(audio_path)

            # Dump transcript if requested
            if self.dump_transcript_path:
                try:
                    with open(self.dump_transcript_path, 'w') as f:
                        for w in all_words:
                            f.write(f"{w.start:.3f}\t{w.end:.3f}\t{w.word.strip()}\n")
                    print(f"  [OK] Raw transcript dumped to: {self.dump_transcript_path}")
                except Exception as e:
                    print(f"  [WARN] Failed to dump transcript: {e}")
            
            # Detect profanity
            print(f"  Searching {len(all_words)} words for profanity...")
            profanity_segments = self._detect_profanity_in_words(all_words)
            
            if profanity_segments:
                print(f"  [OK] Profanity search complete: {len(profanity_segments)} segment(s) found")
                print(f"  Merging nearby profanity segments...")
                profanity_segments = self._merge_nearby(profanity_segments)
                print(f"  [OK] Merged into {len(profanity_segments)} segment(s)")
            else:
                print(f"  [WARN] No profanity segments detected")
            
        except FileNotFoundError as e:
            raise MissingBinaryError(
                "Required media tool not found. Please install FFmpeg (ffmpeg and ffprobe) and ensure both are in PATH."
            ) from e
        except Exception as e:
            print(f"  [ERROR] Error during audio profanity detection: {e}")
            import traceback
            traceback.print_exc()
            profanity_segments = []
        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
        
        return profanity_segments

    def _ensure_media_binaries(self) -> None:
        """Verify ffmpeg and ffprobe are available."""
        missing = []
        if shutil.which('ffmpeg') is None:
            missing.append('ffmpeg')
        if shutil.which('ffprobe') is None:
            missing.append('ffprobe')
        if missing:
            raise MissingBinaryError(
                f"Missing required binary/binaries: {', '.join(missing)}. "
                "Install FFmpeg and make sure ffmpeg/ffprobe are in your PATH."
            )

    def _next_model(self, current: str):
        """Return next larger model name or None."""
        if current not in self._MODEL_ORDER:
            return None
        idx = self._MODEL_ORDER.index(current)
        if idx + 1 < len(self._MODEL_ORDER):
            return self._MODEL_ORDER[idx + 1]
        return None

    def _retry_transcribe(self, audio_path: Path):
        """Retry transcription after model upgrade."""
        profanity_segments = []
        try:
            print(f"  [INFO] Retranscribing with upgraded model '{self.model_size}'...")
            start_time = time.time()
            # Usar fallback también aquí
            segments, info = self._transcribe_with_fallback(audio_path)
            all_words = []
            for segment in segments:
                for word in segment.words:
                    all_words.append(word)
            elapsed = time.time() - start_time
            print(f"  [OK] Upgrade transcription complete in {elapsed:.1f}s ({info.duration/elapsed:.1f}x real-time)")
            wpm = len(all_words) / (info.duration / 60.0) if info.duration else 0
            print(f"  Transcript stats (upgraded): {len(all_words)} words, {wpm:.1f} WPM")
            if self.dump_transcript_path:
                try:
                    with open(self.dump_transcript_path, 'w') as f:
                        for w in all_words:
                            f.write(f"{w.start:.3f}\t{w.end:.3f}\t{w.word.strip()}\n")
                    print(f"  [OK] Raw transcript dumped to: {self.dump_transcript_path}")
                except Exception as e:
                    print(f"  [WARN] Failed to dump transcript: {e}")
            print(f"  Searching {len(all_words)} words for profanity...")
            profanity_segments = self._detect_profanity_in_words(all_words)
        except Exception as e:
            print(f"  [ERROR] Error during upgraded transcription: {e}")
        return profanity_segments
    
    def _merge_nearby(self, segments: List[Tuple[float, float, str]]) -> List[Tuple[float, float, str]]:
        """Merge profanity segments that are close together."""
        if not segments:
            return []
        
        segments.sort(key=lambda x: x[0])
        merged = []
        current_start, current_end, current_words = segments[0]
        current_words_set = {current_words}
        
        for start, end, word in segments[1:]:
            if start <= current_end + self.phrase_gap:
                current_end = max(current_end, end)
                current_words_set.add(word)
            else:
                merged.append((current_start, current_end, ', '.join(sorted(current_words_set))))
                current_start, current_end, current_words = start, end, word
                current_words_set = {word}
        
        merged.append((current_start, current_end, ', '.join(sorted(current_words_set))))
        return merged

    def _detect_profanity_in_words(self, all_words: List) -> List[Tuple[float, float, str]]:
        """Detect profanity by scanning through all words and grouping them into segments."""
        if not all_words:
            return []
        
        all_words.sort(key=lambda w: w.start)
        profanity_segments = []
        i = 0
        n = len(all_words)
        
        while i < n:
            word_info = all_words[i]
            word = word_info.word.strip().lower().rstrip('.,!?;:')
            
            phrase_found = False
            max_lookahead = min(5, n - i)
            for j in range(i, i + max_lookahead):
                phrase_words = []
                for k in range(i, j + 1):
                    phrase_words.append(all_words[k].word.strip().lower().rstrip('.,!?;:'))
                phrase = ' '.join(phrase_words)
                if phrase in self.PROFANITY_PHRASES:
                    start = all_words[i].start
                    end = all_words[j].end
                    profanity_segments.append((start, end, phrase))
                    i = j + 1
                    phrase_found = True
                    break
            
            if not phrase_found:
                if word in self.PROFANITY_WORDS_SET:
                    profanity_segments.append((word_info.start, word_info.end, word))
                i += 1
        
        return profanity_segments