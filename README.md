# 🎙️ Profanity Filter Español

Filtro de groserías para audio/video usando **faster-whisper** con aceleración GPU por Fieneko735.  
Ideal para streams, podcasts y grabaciones, esta es una version modificada con IA gracias a la ayuda de Deepseek, no se nada sobre codificacion por lo que no podria dar soporte o algo por el estilo. 

Simplemente es un pequeño proyecto para filtrar grocerias en español, principalmente español mexicano. Aunque no llega a procesar todas asi que si alguien que sepa codificar correctamente quiere modificarlo adelante jeje despues de todo esta version para filtrar en español fue generada con ayuda de la IA por lo que no se que tan bien codificado este, aunque funciono correctamente en los videos con los que probe a filtrar se le llega a ir alguna que otra groseria. La verdad mejora bastante el contenido para que no contenga lenguaje demasiado grosero y se pueda usar en redes o en el lugar donde lo quieran subir.

Quien quiera aportar para tener un mejor filtrado en español puede hacerlo modificando el archivo profanity_words.py, es totalmente bienvenida cualquier ayuda para mejorar la exactitud.

## ⚡ Características
- Transcripción con `faster-whisper` (GPU). (agregado por Fieneko735, no existia la posibilidad de procesar con gpu en la version original)
- Detección de groserías en español mexicano. (agregado por Fieneko735, igualmente no existia el español solo filtraba contenido en ingles)
- Silencio o corte de segmentos ofensivos.
- Soporte para cualquier formato de video (MP4, MKV, AVI, etc.).

## 📦 Requisitos
- Python 3.11
- NVIDIA GPU con CUDA 12.4
- FFmpeg instalado en el PATH
  
## Notas:
- Esto fue probado en una pc con Ryzen 5 5600x, 32gb Ram, RTX 4070 12gb.
- La mayoria del trabajo de codificacion y demas fue generada con deepseek por lo que no sabria decir que tan bien codificado este todo pero funciona correctamente al procesar videos en español.
- Quien quiera modificarlo puede hacerlo jeje ayudaria bastante si se logra mejorar para que filtre la mayoria de grocerias de videos en español.
- Aun se mantiene el trabajo original de filtrado para palabras en ingles asi que aun funciona para ello pero ahora con la aceleracion por GPU.
- Esta desactivada la generacion de subtitulos para este proyecto asi que si alguien ocupa esa generacion debera modificar el codigo para reactivarlo en el archivo clean.py.
  
## 🚀 Instalación rápida

```bash
git clone https://github.com/Fieneko735/profanity-filter-spanish.git
cd profanity-filter-spanish
python -m venv venv
# En Windows: venv\Scripts\activate
# En Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
