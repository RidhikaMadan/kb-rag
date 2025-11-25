from backend.rag_engine import RAGEngine

def load_rag():
    return RAGEngine(kb_folder="kb")

rag = load_rag()

def main():
    print("Welcome to the Knowledge Base Assistant CLI!")
    print("You can ask questions related to the knowledge base.")
    print("Type 'exit' to quit the application.\n")

    while True:
        user_query = input("Ask your question: ")

        if user_query.lower() == 'exit':
            print("Goodbye!")
            break

        try:
            answer = rag.answer(user_query)

            answer = (answer if isinstance(answer, str) else answer.text).replace("```", "").strip()

            print(f"\nAssistant: {answer}\n")

        except Exception as e:
            print(f"⚠️ Error: {e}\n")

if __name__ == "__main__":
    main()
