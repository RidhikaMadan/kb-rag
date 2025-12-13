from backend.rag_engine import RAGEngine
import sys

# Set stdout to UTF-8 encoding
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

if sys.stdin.encoding != 'utf-8':
    import io
    sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8', errors='replace')


def ensure_utf8(text: str) -> str:
    """Ensure text is UTF-8 safe by encoding and decoding with error handling"""
    if not isinstance(text, str):
        text = str(text)
    try:
        return text.encode('utf-8', errors='replace').decode('utf-8')
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')


def load_rag():
    return RAGEngine(kb_folder="KB")

rag = load_rag()

def main():
    print("Welcome to the Knowledge Base Assistant CLI!")
    print("You can ask questions related to the knowledge base.")
    print("Type 'exit' to quit the application.\n")

    while True:
        try:
            user_query = input("Ask your question: ")
        except (UnicodeDecodeError, UnicodeEncodeError):
            print("⚠️ Error: Invalid input encoding. Please try again.")
            continue

        if not user_query:
            continue
            
        user_query = ensure_utf8(user_query)

        if user_query.lower() == 'exit':
            print("Goodbye!")
            break

        try:
            answer, sources, chunks = rag.answer(user_query)

            answer = (answer if isinstance(answer, str) else answer.text).replace("```", "").strip()
            answer = ensure_utf8(answer)

            print(f"\nAssistant: {answer}\n")
            if sources:
                print(f"Sources: {', '.join(sources)}\n")
            if chunks:
                print(f"Retrieved {len(chunks)} chunks from knowledge base.\n")

        except Exception as e:
            error_msg = ensure_utf8(str(e))
            print(f"⚠️ Error: {error_msg}\n")

if __name__ == "__main__":
    main()
