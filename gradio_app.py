"""
Gradio Web Interface for Profanity Filter Español
Adapted for Fieneko735's project
"""

import gradio as gr
import subprocess
import os
from pathlib import Path
import sys

def clean_video(video_file, model_size, mute_only):
    """
    Procesa un video llamando a clean.py con los parámetros seleccionados.
    """
    if video_file is None:
        return None, "⚠️ Por favor, sube un video."

    input_path = Path(video_file)
    output_dir_path = Path.home() / "Videos"
    output_dir_path.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "clean.py",
        video_file,
        "--model", model_size,
        "--output-dir", str(output_dir_path)
    ]
    if mute_only:
        cmd.append("--mute-only")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            env=os.environ.copy()
        )
        logs = result.stdout + result.stderr

        if result.returncode != 0:
            return None, f"❌ Error al procesar:\n{logs}"

        output_path = output_dir_path / f"{input_path.stem}_clean{input_path.suffix}"
        if output_path.exists():
            return str(output_path), f"✅ Procesado con éxito:\n{logs}"
        else:
            posibles = list(output_dir_path.glob(f"{input_path.stem}_clean*"))
            if posibles:
                return str(posibles[0]), f"✅ Procesado con éxito:\n{logs}"
            else:
                return None, f"⚠️ No se encontró el archivo de salida:\n{logs}"

    except Exception as e:
        return None, f"❌ Excepción: {str(e)}"


def create_interface():
    """Crea y devuelve la interfaz Gradio"""
    
    with gr.Blocks(title="Profanity Filter Español") as demo:
        gr.Markdown("""
        # 🎙️ Profanity Filter Español
        **Filtra groserías de tus videos usando IA con aceleración GPU.**
        Ideal para streams, podcasts y grabaciones. Adaptado al español mexicano por [Fieneko735](https://github.com/Fieneko735).
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                video_input = gr.Video(label="Sube tu video", interactive=True)
                model_selector = gr.Dropdown(
                    choices=["tiny", "base", "small", "medium", "large-v3"],
                    value="small",
                    label="Modelo de transcripción (small = equilibrio velocidad/precisión)"
                )
                mute_checkbox = gr.Checkbox(
                    label="Silenciar groserías (mute-only)",
                    value=True
                )
                submit_btn = gr.Button("🚀 Procesar video", variant="primary")
                
            with gr.Column(scale=2):
                video_output = gr.File(label="📥 Video procesado (descargar)", interactive=False)
                log_output = gr.Textbox(
                    label="Logs del proceso",
                    lines=20,
                    max_lines=30
                )
        
        submit_btn.click(
            fn=clean_video,
            inputs=[video_input, model_selector, mute_checkbox],
            outputs=[video_output, log_output]
        )
        
        gr.Markdown("""
        ---
        ### 📌 Notas importantes
        - El video procesado se guardará automáticamente en **tu carpeta de Videos** (`C:/Users/TuUsuario/Videos/`) con el nombre `nombre_original_clean.mp4`.
        - El modelo **small** es el recomendado por su equilibrio entre velocidad y precisión.
        - **Mute-only** silencia las groserías en lugar de cortar el video (mantiene la duración).
        - Asegúrate de tener **FFmpeg** instalado y en el PATH del sistema.
        
        ### 🙏 Créditos
        - Proyecto original: [Adeel Raza](https://github.com/adeel-raza/profanity-filter)
        - Adaptación al español y soporte GPU: **Fieneko735**
        - Asistencia con IA: **DeepSeek**
        """)
    
    return demo