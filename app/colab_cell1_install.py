# ════════════════════════════════════════════════════════════
# LEUKEMIA HARMONISER — CELL 1: INSTALL DEPENDENCIES
# Run this once at the start of each Colab session
# ════════════════════════════════════════════════════════════

!pip install -q fastapi "uvicorn[standard]" nest-asyncio pyngrok \
    rapidfuzz pandas numpy joblib scikit-learn xgboost \
    imbalanced-learn python-multipart aiofiles reportlab

print("✓ All dependencies installed. Run Cell 2.")