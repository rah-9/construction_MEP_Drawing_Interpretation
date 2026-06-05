import importlib
import traceback


MODULES = [
    ("google.genai", "Gemini SDK"),
    ("google.generativeai", "Google GenerativeAI"),
    ("openai", "OpenAI SDK"),
    ("torch", "PyTorch"),
    ("torchvision", "TorchVision"),
    ("torchaudio", "TorchAudio"),
    ("tensorflow", "TensorFlow"),
    ("mediapipe", "MediaPipe"),
    ("gradio", "Gradio"),
    ("paddle", "PaddlePaddle"),
    ("paddleocr", "PaddleOCR"),
    ("transformers", "Transformers"),
    ("huggingface_hub", "HuggingFace Hub"),
    ("PIL", "Pillow"),
    ("cv2", "OpenCV"),
    ("fitz", "PyMuPDF"),
    ("numpy", "NumPy"),
    ("pandas", "Pandas"),
    ("requests", "Requests"),
    ("httpx", "HTTPX"),
]


print("=" * 80)
print("ENVIRONMENT HEALTH CHECK")
print("=" * 80)

working = []
failed = []

for module_name, display_name in MODULES:
    try:
        module = importlib.import_module(module_name)

        version = getattr(module, "__version__", "Version Not Available")

        print(f"[OK] {display_name:<25} | {version}")

        working.append(display_name)

    except Exception as e:
        print(f"[FAIL] {display_name:<25} | {str(e)}")

        failed.append((display_name, str(e)))

print("\n")
print("=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"Working Modules : {len(working)}")
print(f"Failed Modules  : {len(failed)}")

if failed:
    print("\nFAILED MODULES:")
    print("-" * 80)

    for name, error in failed:
        print(f"{name}")
        print(f"Error: {error}")
        print("-" * 80)

print("\n")
print("=" * 80)
print("PYTORCH DETAILS")
print("=" * 80)

try:
    import torch

    print("Torch Version:", torch.__version__)
    print("CUDA Available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU Name:", torch.cuda.get_device_name(0))
        print("GPU Count:", torch.cuda.device_count())

except Exception:
    traceback.print_exc()

print("\n")
print("=" * 80)
print("PADDLEOCR TEST")
print("=" * 80)

try:
    from paddleocr import PaddleOCR

    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False
    )

    print("[OK] PaddleOCR Initialized Successfully")

except Exception:
    traceback.print_exc()

print("\n")
print("=" * 80)
print("OLLAMA TEST")
print("=" * 80)

try:
    import requests

    response = requests.get(
        "http://localhost:11434/api/tags",
        timeout=5
    )

    if response.status_code == 200:
        print("[OK] Ollama Server Running")

        models = response.json().get("models", [])

        print(f"Installed Models: {len(models)}")

        for model in models:
            print(" -", model.get("name"))

    else:
        print(
            f"[FAIL] Ollama Returned Status Code {response.status_code}"
        )

except Exception as e:
    print(f"[FAIL] Ollama Not Reachable: {e}")

print("\n")
print("=" * 80)
print("CHECK COMPLETE")
print("=" * 80)