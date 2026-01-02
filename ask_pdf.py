import sys
import torch
from pdf2image import convert_from_path
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image

# Nota: La biblioteca `pdf2image` depende de la herramienta de sistema `poppler`.
# Asegúrate de tenerla instalada. Por ejemplo, en Debian/Ubuntu:
# sudo apt-get install poppler-utils

def ask_pdf_processor(pdf_path, question):
    """
    Hace una pregunta a cada página de un documento PDF utilizando un modelo de visión y lenguaje.

    Args:
        pdf_path (str): La ruta al archivo PDF.
        question (str): La pregunta a realizar.
    """
    try:
        # Determinar si se usará GPU o CPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Usando dispositivo: {device}")

        # 1. Cargar el procesador y el modelo
        print("Cargando el modelo y el procesador... Esto puede tardar un momento.")
        processor = AutoProcessor.from_pretrained("docling-project/SmolDocling-256M-preview")
        model = AutoModelForVision2Seq.from_pretrained("docling-project/SmolDocling-256M-preview").to(device)
        print("Modelo y procesador cargados exitosamente.")

        # 2. Convertir el PDF en una lista de imágenes
        print(f"Convirtiendo el PDF '{pdf_path}' a imágenes...")
        images = convert_from_path(pdf_path)
        print(f"PDF convertido. Se encontraron {len(images)} páginas.")

        # 3. Iterar sobre cada página (imagen)
        for i, page_image in enumerate(images):
            print(f"--- Procesando Página {i + 1} ---")

            # Preparar la plantilla de mensajes para el modelo
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},  # Marcador de posición para la imagen
                        {"type": "text", "text": question}
                    ]
                },
            ]

            # Paso 1: Crear el prompt de texto a partir de la plantilla.
            prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

            # Paso 2: Procesar el texto y la imagen juntos para obtener las entradas del modelo.
            inputs = processor(
                text=[prompt],
                images=[page_image],
                return_tensors="pt"
            ).to(device)

            # Generar la salida del modelo
            outputs = model.generate(**inputs, max_new_tokens=256)

            # Decodificar y mostrar la respuesta
            input_token_len = inputs["input_ids"].shape[-1]
            generated_text = processor.decode(outputs[0][input_token_len:], skip_special_tokens=True)

            print(f"Respuesta: {generated_text.strip()}")

    except FileNotFoundError:
        print(f"Error: El archivo PDF '{pdf_path}' no fue encontrado.", file=sys.stderr)
    except Exception as e:
        print(f"Ha ocurrido un error inesperado: {e}", file=sys.stderr)

def main():
    """
    Función principal para analizar los argumentos de la línea de comandos y ejecutar el script.
    """
    if len(sys.argv) != 3:
        print("Uso: python ask_pdf.py <ruta_al_pdf> \"<pregunta>\"", file=sys.stderr)
        print("Ejemplo: python ask_pdf.py documento.pdf \"¿Cuál es el tema principal?\"", file=sys.stderr)
        sys.exit(1)

    pdf_path = sys.argv[1]
    question = sys.argv[2]

    ask_pdf_processor(pdf_path, question)

if __name__ == "__main__":
    main()
