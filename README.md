# 🎙️ Profanity Filter Español

Filtro de groserías para audio/video usando **faster-whisper** con aceleración GPU por Fieneko735.  
Ideal para streams, podcasts y grabaciones, esta es una version modificada con IA gracias a la ayuda de Deepseek, no se nada sobre codificacion por lo que no podria dar soporte o algo por el estilo. 

Simplemente es un pequeño proyecto para filtrar grocerias en español, principalmente español mexicano. Aunque no llega a procesar todas asi que si alguien que sepa codificar correctamente quiere modificarlo adelante jeje despues de todo esta version para filtrar en español fue generada con ayuda de la IA por lo que no se que tan bien codificado este. Funciona correctamente pero se le llegan a escapar algunas palabras asi que el contenido quedara filtrado y libre de groserias hasta cierto punto, por lo que aun necesitara alguna edicion extra para estar completamente limpio

Quien quiera aportar para tener un mejor filtrado en español puede hacerlo modificando el archivo profanity_words.py, es totalmente bienvenida cualquier ayuda para mejorar la exactitud del filtro.

## ⚡ Características
- Transcripción con `faster-whisper` (GPU). (agregado por Fieneko735, no existia la posibilidad de procesar con gpu en la version original)
- Detección de groserías en español mexicano. (agregado por Fieneko735, igualmente no existia el español solo filtraba contenido en ingles)
- Silencio o corte de segmentos ofensivos.
- Soporte para cualquier formato de video (MP4, MKV, AVI, etc.).

## 📦 Requisitos
- Python 3.11
- NVIDIA GPU con CUDA 12.4
- **FFmpeg**: Necesario para extraer audio de los videos.  
  Descárgalo desde [ffmpeg.org](https://ffmpeg.org/download.html) y asegúrate de que esté en el PATH del sistema.
  
## Notas:
- Esto fue probado en una pc con Ryzen 5 5600x, 32gb Ram, RTX 4070 12gb.
- La mayoria del trabajo de codificacion y demas fue generada con deepseek por lo que no sabria decir que tan bien codificado este todo pero funciona correctamente al procesar videos en español.
- Quien quiera modificarlo puede hacerlo jeje ayudaria bastante si se logra mejorar para que filtre la mayoria de grocerias de videos en español.
- Aun se mantiene el trabajo original de filtrado para palabras en ingles asi que aun funciona para ello pero ahora con la aceleracion por GPU.
- Esta desactivada la generacion de subtitulos para este proyecto asi que si alguien ocupa esa generacion debera modificar el codigo para reactivarlo en el archivo clean.py.

## SALIDA DE PRUEBA
-Se probo con un video del youtuber Darkar que suele tener un contenido bastante grosero.

-Especificamente este: https://www.youtube.com/watch?v=S-pFvPx_sqM por si alguien quiere probar su funcionalidad.

```bash
(venv) PS C:\Users\black\Desktop\prueba-final\profanity-filter-spanish> python clean.py C:\Users\black\Downloads\prueba.mp4 C:\Users\black\Downloads\prueba_limpia.mp4 --model small --mute-only
============================================================
AUTOMATED MOVIE CLEANER - PROFANITY FILTER
============================================================
Step 1: Transcribing audio and detecting profanity (faster-whisper)
------------------------------------------------------------
  🔍 Detecting best available device...
  ✅ CUDA detected: NVIDIA GeForce RTX 4070
  Loading faster-whisper model: small...
  Device: cuda
    Trying compute_type=float16...
  ✅ Faster-whisper loaded! (device=cuda, compute_type=float16)
  Video duration: 1.8 minutes
  Extracting audio from video (dialog enhance)...
  ✓ Audio extracted
  Transcribing audio with faster-whisper (small model)...
  ⏳ Estimated time: ~3.6 seconds for 1.8 min video
  ✓ Transcription complete in 8.9s (12.2x real-time)
  Transcript stats: 214 words, 118.2 WPM
  Searching 214 words for profanity...
  ✓ Profanity search complete: 16 segment(s) found
  Merging nearby profanity segments...
  ✓ Merged into 12 segment(s)
------------------------------------------------------------
Step 1 Summary: Found 12 profanity segment(s) in audio
    - 20.84s to 21.26s (0.42s): 'pendejada'
    - 27.26s to 30.42s (3.16s): 'pinche salchicha'
    - 33.18s to 33.58s (0.40s): 'farsa'
    - 36.78s to 38.24s (1.46s): 'a la puta chingada'
    - 43.62s to 44.18s (0.56s): 'pendeja'
    - 47.32s to 49.66s (2.34s): 'a la verga, pinche'
    - 57.04s to 57.44s (0.40s): 'carajo'
    - 60.60s to 61.76s (1.16s): 'farsa, pinche'
    - 68.26s to 68.92s (0.66s): 'pedo'
    - 72.28s to 73.34s (1.06s): 'imbécil, mierda'
    - 90.40s to 91.12s (0.72s): 'pendejada'
    - 94.92s to 95.30s (0.38s): 'asesina'

Step 2: Merging segments...
  Audio segments: 12
  Merged into 12 segment(s) to remove
    1. 20.84s to 21.26s (0.42s)
    2. 27.26s to 30.42s (3.16s)
    3. 33.18s to 33.58s (0.40s)
    4. 36.78s to 38.24s (1.46s)
    5. 43.62s to 44.18s (0.56s)
    6. 47.32s to 49.66s (2.34s)
    7. 57.04s to 57.44s (0.40s)
    8. 60.60s to 61.76s (1.16s)
    9. 68.26s to 68.92s (0.66s)
    10. 72.28s to 73.34s (1.06s)
    11. 90.40s to 91.12s (0.72s)
    12. 94.92s to 95.30s (0.38s)

Step 2b: Expanding segments to catch clipped syllables...
    1. 20.84s to 21.26s (0.42s)
    2. 27.26s to 30.42s (3.16s)
    3. 33.18s to 33.58s (0.40s)
    4. 36.78s to 38.24s (1.46s)
    5. 43.62s to 44.18s (0.56s)
    6. 47.32s to 49.66s (2.34s)
    7. 57.04s to 57.44s (0.40s)
    8. 60.60s to 61.76s (1.16s)
    9. 68.26s to 68.92s (0.66s)
    10. 72.28s to 73.34s (1.06s)
    11. 90.40s to 91.12s (0.72s)
    12. 94.92s to 95.30s (0.38s)

Step 3: Processing detected segments in video...
------------------------------------------------------------
  Muting 12 segment(s) from video...
  Total affected time: 12.72 seconds (0.21 minutes)
  Processing 12 segment(s) to remove...
  Total time to remove: 12.72 seconds (0.21 minutes)
  Mute-only mode enabled: preserving timeline and muting audio in detected intervals
  ✓ Audio muting complete
------------------------------------------------------------
Total video cutting time: 1.77 seconds
Cutting time written to: C:\Users\black\Downloads\prueba_limpia.time.txt
============================================================
SUCCESS!
============================================================
Cleaned video saved to: C:\Users\black\Downloads\prueba_limpia.mp4
No subtitles were generated or embedded.
Removed 12 segment(s)
Total time removed: 12.72 seconds
Total end-to-end processing time: 20.09 seconds
Overall processing time written to: C:\Users\black\Downloads\prueba_limpia.total_time.txt

```
--------
  
## 🚀 Instalación rápida

```bash
git clone https://github.com/Fieneko735/profanity-filter-spanish.git
cd profanity-filter-spanish
python -m venv venv
# En Windows: venv\Scripts\activate
# En Linux/Mac: source venv/bin/activate

# 1. INSTALA PRIMERO ESTO (PyTorch con GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 2. LUEGO ESTO (resto de dependencias)
pip install -r requirements.txt

# 3. Verifica que detecte la GPU correctamente
python -c "import torch; print('CUDA disponible:', torch.cuda.is_available())"

# 4. Prueba el filtro
python clean.py "TU/RUTA/VIDEO.mp4" "TU/RUTA/VIDEO_SALIDA.mp4" --model small --mute-only
