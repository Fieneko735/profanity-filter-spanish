from gradio_app import create_interface
from pathlib import Path
import tempfile
import webbrowser  # <-- AÑADIR ESTA LÍNEA

demo = create_interface()

if __name__ == "__main__":
    videos_dir = str(Path.home() / "Videos")
    temp_dir = tempfile.gettempdir()
    
    # Abrir el navegador automáticamente
    webbrowser.open("http://127.0.0.1:7860")
    
    demo.launch(
        server_name="127.0.0.1",
        share=False,
        allowed_paths=[videos_dir, temp_dir]
    )