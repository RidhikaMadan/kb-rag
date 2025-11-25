import os
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from backend.local_llm import LocalLLM
from sentence_transformers import CrossEncoder, SentenceTransformer
import numpy as np

MAX_CONTEXT_TOKENS = 8192 
AVG_CHARS_PER_TOKEN = 4 

class ZeroShotIntentClassifier:
    def __init__(self):
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.intents = {
            "casual": "User is casually chatting, greeting, or making informal statements.",
            "informational": (
                "User is asking about technical, procedural, or actionable topics, "
                "including IT, HR, workflows, company policies, or work setup such as remote work."
            )
        }
        self.intent_embeddings = self.model.encode(list(self.intents.values()), convert_to_numpy=True)

    def predict(self, query: str):
        query_emb = self.model.encode([query], convert_to_numpy=True)[0]
        cos_sims = self.intent_embeddings @ query_emb / (
            np.linalg.norm(self.intent_embeddings, axis=1) * np.linalg.norm(query_emb) + 1e-8
        )
        predicted_index = np.argmax(cos_sims)
        confidence = cos_sims[predicted_index]
        predicted_intent = list(self.intents.keys())[predicted_index]
        return predicted_intent, confidence


class RAGEngine:
    def __init__(
        self,
        kb_folder="kb",
        index_path="index.faiss",
        top_k=3,
        chunk_size=500,
        chunk_overlap=50,
        embedding_model_name="sentence-transformers/all-mpnet-base-v2",
        reranker_model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        casual_threshold=0.6
    ):
        self.kb_folder = kb_folder
        self.index_path = index_path
        self.top_k = top_k
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.casual_threshold = casual_threshold

        self.model = LocalLLM(n_ctx=MAX_CONTEXT_TOKENS)
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
        self.reranker = CrossEncoder(reranker_model_name)
        self.intent_classifier = ZeroShotIntentClassifier()
        self.vectorstore = None

        self._load_or_create_index()


    def _load_texts(self):
        docs = []
        for root, _, files in os.walk(self.kb_folder):
            for fname in files:
                if fname.endswith(".txt"):
                    fpath = os.path.join(root, fname)
                    with open(fpath, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                        if content:
                            rel_path = os.path.relpath(fpath, self.kb_folder)
                            docs.append(Document(page_content=content, metadata={"source": rel_path}))
        return docs

    def _load_or_create_index(self):
        if os.path.exists(self.index_path):
            print("Loading FAISS index...")
            self.vectorstore = FAISS.load_local(
                self.index_path,
                self.embedding_model,
                allow_dangerous_deserialization=True
            )
        else:
            print("Creating FAISS index from KB...")
            documents = self._load_texts()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len
            )
            split_docs = text_splitter.split_documents(documents)
            self.vectorstore = FAISS.from_documents(split_docs, self.embedding_model)
            self.vectorstore.save_local(self.index_path)
            print("Index created.")


    def retrieve(self, query, min_score: float = 0.5):
        results_with_scores = self.vectorstore.similarity_search_with_score(query, k=self.top_k)
        filtered_docs = [(doc, score) for doc, score in results_with_scores if score >= min_score]
        return filtered_docs

    def rerank_with_cross_encoder(self, query, docs_with_scores):
        if not docs_with_scores or len(docs_with_scores) == 1:
            return [doc for doc, _ in docs_with_scores]
        pairs = [[query, doc.page_content] for doc, _ in docs_with_scores]
        scores = self.reranker.predict(pairs)
        scored = list(zip([doc for doc, _ in docs_with_scores], scores))
        ranked = sorted(scored, key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked]

    def clean_response(self, text: str) -> str:
        match = re.search(r"(?<=Answer:)(.*)", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        cleaned = re.sub(r"^\s*\(Assuming.*?\)\s*", "", text, flags=re.MULTILINE)
        cleaned = re.sub(r"^\s*Answer:\s*", "", cleaned, flags=re.MULTILINE)
        return cleaned.strip()

 
    def _generate_casual_reply(self, query):
        system_prompt = "You are a friendly assistant. Respond briefly and naturally to casual user messages."
        user_prompt = f'User message: "{query}"\nResponse:'
        response = self.model.generate(system_prompt, user_prompt, max_tokens=50)
        return response if isinstance(response, str) else response.text

   
    def answer(self, query, max_tokens=1000, min_score=0.5, debug=False):
        # Zero-shot intent classification
        intent, confidence = self.intent_classifier.predict(query)

        # Handle casual queries (only if high confidence)
        if intent == "casual" and confidence >= self.casual_threshold:
            return self._generate_casual_reply(query)

        # Informational query → retrieve + rerank
        retrieved_docs_with_scores = self.retrieve(query, min_score=min_score)
        if not retrieved_docs_with_scores:
            return "I couldn't find this information in the knowledge base."

        retrieved_docs = self.rerank_with_cross_encoder(query, retrieved_docs_with_scores)

        total_chars = 0
        chunks_for_prompt = []
        for doc in retrieved_docs:
            doc_len = len(doc.page_content)
            if total_chars + doc_len > MAX_CONTEXT_TOKENS * AVG_CHARS_PER_TOKEN:
                break
            chunks_for_prompt.append(doc.page_content)
            total_chars += doc_len
        context = "\n".join(chunks_for_prompt)

        system_prompt = (
            "You are a knowledge base assistant. Answer using ONLY the context below. "
            "Do NOT invent information. Be helpful and give full steps if needed."
        )

        user_prompt = f"Documents:\n{context}\n\nMessage:\n{query}\n\nAnswer:"

        response = self.model.generate(
            system_prompt,
            user_prompt,
            max_tokens=max_tokens
        )

        cleaned_text = self.clean_response(response)

        if debug:
            print(f"Intent classification: {intent} (confidence: {confidence:.3f})")
            print("=== Retrieved Chunks ===")
            for doc, score in retrieved_docs_with_scores:
                print(f"Score: {score:.4f} | Source: {doc.metadata.get('source','N/A')}")
                print(doc.page_content[:300].replace('\n', ' ') + "...\n")
            print("=== LLM Response ===")
            print(cleaned_text)

        return cleaned_text
