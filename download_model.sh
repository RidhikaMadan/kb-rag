
MODEL_REPO="bartowski/Llama-3.2-3B-Instruct-GGUF"
MODEL_FILE="Llama-3.2-3B-Instruct-Q4_K_M.gguf"

mkdir -p models

URL="https://huggingface.co/$MODEL_REPO/resolve/main/$MODEL_FILE"

echo "Downloading $MODEL_FILE ..."
echo "From: $URL"
echo ""

TARGET="models/$MODEL_FILE"

if command -v aria2c >/dev/null 2>&1; then
    echo "⚡ aria2c detected → Using accelerated download"
    aria2c -x 16 -s 16 -k 1M "$URL" -o "$TARGET"
else
    echo "aria2c not found → Using curl fallback"
    curl -L "$URL" -o "$TARGET"
fi

echo ""
echo "Download complete!"
echo "Saved to: $TARGET"
