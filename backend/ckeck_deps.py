import sys

required = {
    "fastapi": "FastAPI",
    "uvicorn": "Uvicorn",
    "PIL": "Pillow (Image handling)",
    "tensorflow": "TensorFlow (ML Model)" 
}

print("Checking dependencies...\n")

all_good = True

for module, name in required.items():
    try:
        __import__(module)
        print(f"✅ {name} ({module}) - FOUND")
    except ImportError:
        print(f"❌ {name} ({module}) - MISSING OR ERROR")
        all_good = False

if all_good:
    print("\n✨ All dependencies confirmed!")
else:
    print("\n⚠️  Some dependencies are missing. Run 'pip install -r requirements.txt'")
    sys.exit(1)