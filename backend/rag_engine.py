"""
RAG Engine with vector search, reranking, and intent classification
"""
import os
import re
import sys
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder, SentenceTransformer
import numpy as np
from typing import List, Dict, Optional, Tuple, Any

from backend.llm_provider import get_llm_provider, LLMProvider


MAX_CONTEXT_TOKENS = 8192
AVG_CHARS_PER_TOKEN = 4


def ensure_utf8(text: str) -> str:
    """Ensure text is UTF-8 safe by encoding and decoding with error handling"""
    if not isinstance(text, str):
        text = str(text)
    try:
        return text.encode('utf-8', errors='replace').decode('utf-8')
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')


class ZeroShotIntentClassifier:
    """Classify user queries as casual or informational"""
    
    def __init__(self):
        print("    Loading sentence transformer model...")
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
        """Predict intent and confidence"""
        try:
            if not query or not query.strip():
                return "informational", 0.5
            
            # Ensure query is UTF-8 safe
            query = ensure_utf8(query)
            
            query_emb = self.model.encode([query], convert_to_numpy=True)[0]
            cos_sims = self.intent_embeddings @ query_emb / (
                np.linalg.norm(self.intent_embeddings, axis=1) * np.linalg.norm(query_emb) + 1e-8
            )
            predicted_index = np.argmax(cos_sims)
            confidence = float(cos_sims[predicted_index])
            predicted_intent = list(self.intents.keys())[predicted_index]
            return predicted_intent, confidence
        except Exception as e:
            try:
                safe_error = ensure_utf8(str(e))
                print(f"Error in intent classifier predict: {safe_error}")
            except Exception:
                print("Error in intent classifier predict")
            return "informational", 0.5


class RAGEngine:
    """RAG Engine with vector search, reranking, and LLM generation"""
    
    def __init__(
        self,
        kb_folder="KB",
        index_path="index.faiss",
        top_k=5,
        chunk_size=500,
        chunk_overlap=50,
        embedding_model_name="sentence-transformers/all-mpnet-base-v2",
        reranker_model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        casual_threshold=0.6,
        use_local_llm: bool = False,
        openai_api_key: Optional[str] = None
    ):
        self.kb_folder = kb_folder
        self.index_path = index_path
        self.top_k = top_k
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.casual_threshold = casual_threshold
        
        print("  Loading LLM provider...")
        self.llm_provider: LLMProvider = get_llm_provider(use_local=use_local_llm, api_key=openai_api_key)
        self.use_local_llm = use_local_llm
        
        print("  Loading embedding model...")
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
        
        print("  Loading reranker model...")
        self.reranker = CrossEncoder(reranker_model_name)
        
        print("  Loading intent classifier...")
        self.intent_classifier = ZeroShotIntentClassifier()
        
        self.vectorstore = None
        print("  Loading or creating vector index...")
        self._load_or_create_index()
    
    def _load_texts(self):
        """Load all text files from knowledge base folder"""
        from backend.file_processor import extract_text_from_file
        docs = []
        supported_extensions = {'.txt', '.md', '.markdown', '.pdf'}
        
        for root, _, files in os.walk(self.kb_folder):
            for fname in files:
                fpath = Path(root) / fname
                if fpath.suffix.lower() in supported_extensions:
                    try:
                        content = extract_text_from_file(fpath)
                        if content and content.strip():
                            rel_path = os.path.relpath(fpath, self.kb_folder)
                            docs.append(Document(
                                page_content=content.strip(),
                                metadata={"source": rel_path, "file_type": fpath.suffix.lower()}
                            ))
                    except Exception as e:
                        try:
                            safe_path = ensure_utf8(str(fpath))
                            safe_error = ensure_utf8(str(e))
                            print(f"Warning: Could not load {safe_path}: {safe_error}")
                        except Exception:
                            print(f"Warning: Could not load file")
                        continue
        return docs
    
    def _load_or_create_index(self):
        """Load existing FAISS index or create new one"""
        if os.path.exists(self.index_path):
            print("Loading FAISS index...")
            try:
                self.vectorstore = FAISS.load_local(
                    self.index_path,
                    self.embedding_model,
                    allow_dangerous_deserialization=True
                )
                print("FAISS index loaded successfully.")
            except Exception as e:
                try:
                    safe_error = ensure_utf8(str(e))
                    print(f"Error loading FAISS index: {safe_error}")
                except Exception:
                    print("Error loading FAISS index")
                print("Attempting to recreate index from KB...")
                self.vectorstore = None
        
        if self.vectorstore is None:
            print("Creating FAISS index from KB...")
            documents = self._load_texts()
            if documents:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    length_function=len
                )
                split_docs = text_splitter.split_documents(documents)
                self.vectorstore = FAISS.from_documents(split_docs, self.embedding_model)
                self.vectorstore.save_local(self.index_path)
                print("Index created.")
            else:
                print("No documents found in KB folder.")
    
    def add_documents_to_index(self, documents: List[Document]):
        """Add new documents to the existing index"""
        if not documents:
            return
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len
        )
        split_docs = text_splitter.split_documents(documents)
        
        if self.vectorstore is None:
            self.vectorstore = FAISS.from_documents(split_docs, self.embedding_model)
        else:
            self.vectorstore.add_documents(split_docs)
        
        self.vectorstore.save_local(self.index_path)
        try:
            print(f"Added {len(split_docs)} document chunks to index.")
        except Exception:
            print("Added document chunks to index.")
    
    def retrieve(self, query: str, min_score: float = 0.5, session_id: Optional[str] = None):
        """Retrieve relevant documents with similarity search"""
        if self.vectorstore is None:
            return []
        
        # Ensure query is UTF-8 safe
        query = ensure_utf8(query)
        
        k = self.top_k * 3 if session_id else self.top_k
        results_with_scores = self.vectorstore.similarity_search_with_score(query, k=k)
        filtered_docs = [(doc, score) for doc, score in results_with_scores if score >= min_score]
        
        if session_id:
            filtered_docs = [
                (doc, score) for doc, score in filtered_docs
                if not doc.metadata.get("is_session_upload", False) or doc.metadata.get("session_id") == session_id
            ]
            filtered_docs = filtered_docs[:self.top_k]
        
        return filtered_docs
    
    def rerank_with_cross_encoder(self, query: str, docs_with_scores):
        """Rerank documents using cross-encoder"""
        if not docs_with_scores or len(docs_with_scores) == 1:
            return [doc for doc, _ in docs_with_scores]
        
        # Ensure query and document content are UTF-8 safe
        query = ensure_utf8(query)
        pairs = [[query, ensure_utf8(doc.page_content)] for doc, _ in docs_with_scores]
        scores = self.reranker.predict(pairs)
        scored = list(zip([doc for doc, _ in docs_with_scores], scores))
        ranked = sorted(scored, key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked]
    
    def clean_response(self, text: str) -> str:
        """Clean LLM response by removing markdown formatting"""
        # Ensure text is UTF-8 safe at the start
        text = ensure_utf8(text)
        try:
            cleaned = re.sub(r"^#+\s+", "", text, flags=re.MULTILINE)
            cleaned = re.sub(r"\*\*([^\*]+)\*\*", r"\1", cleaned)
            cleaned = re.sub(r"\*([^\*]+)\*", r"\1", cleaned)
            cleaned = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", cleaned)
            cleaned = re.sub(r"^\s*[-*+]\s+", "", cleaned, flags=re.MULTILINE)
            match = re.search(r"(?<=Answer:)(.*)", cleaned, re.DOTALL)
            if match:
                cleaned = match.group(1).strip()
            cleaned = re.sub(r"^\s*\(Assuming.*?\)\s*", "", cleaned, flags=re.MULTILINE)
            cleaned = re.sub(r"^\s*Answer:\s*", "", cleaned, flags=re.MULTILINE)
            cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
            cleaned = re.sub(r" {2,}", " ", cleaned)
            result = cleaned.strip()
            # Ensure final result is UTF-8 safe
            return ensure_utf8(result)
        except Exception:
            # If cleaning fails, just return UTF-8 safe version of original
            return ensure_utf8(text.strip())
    
    def _generate_casual_reply(self, query: str, chat_history: Optional[List[Dict[str, str]]] = None):
        """Generate a casual reply for non-informational queries"""
        # Ensure query is UTF-8 safe
        query = ensure_utf8(query)
        
        messages = [
            {"role": "system", "content": "You are a friendly assistant. Respond briefly and naturally to casual user messages."},
            {"role": "user", "content": query}
        ]
        
        if chat_history:
            recent_history = chat_history[-6:] if len(chat_history) > 6 else chat_history
            # Ensure all chat history content is UTF-8 safe
            clean_history = []
            for msg in recent_history:
                role = msg.get("role", "user")
                content = ensure_utf8(msg.get("content", ""))
                clean_history.append({"role": role, "content": content})
            messages = [messages[0]] + clean_history + [messages[1]]
        
        response = self.llm_provider.generate(messages, max_tokens=150, temperature=0.7)
        # Clean and ensure UTF-8 safety for casual replies too
        cleaned_response = self.clean_response(response)
        return cleaned_response
    
    def _build_agentic_prompt(self, query: str, context: str, chat_history: Optional[List[Dict[str, str]]] = None) -> List[Dict[str, str]]:
        """Build prompt for informational queries with context"""
        # Ensure query and context are UTF-8 safe
        query = ensure_utf8(query)
        context = ensure_utf8(context)
        
        system_prompt = """You are an intelligent knowledge base assistant with reasoning capabilities. Your role is to:

1. Analyze user queries carefully, considering conversation context
2. Use the provided knowledge base context to answer questions accurately
3. Reason step-by-step when needed
4. Ask clarifying questions if the query is ambiguous
5. Provide actionable, detailed answers when possible
6. Reference specific documents or sources when relevant

You have access to:
- A knowledge base with relevant documents
- Previous conversation history for context
- The ability to reason through complex queries

Guidelines:
- Use ONLY information from the provided context
- If information is not in the context, say so clearly
- Break down complex questions into steps
- Consider the conversation history to understand follow-up questions
- Be helpful, accurate, and concise
- IMPORTANT: Do NOT use markdown formatting (no # headers, no **bold**, no markdown links). Write in plain text only."""

        user_prompt = f"""Context from Knowledge Base:
{context}

Current Query: {query}

Please provide a helpful, accurate answer based on the context above. If the query relates to previous messages, use the conversation history to provide context-aware responses."""

        messages = [{"role": "system", "content": system_prompt}]
        
        if chat_history:
            recent_history = chat_history[-10:] if len(chat_history) > 10 else chat_history
            # Ensure all chat history content is UTF-8 safe
            clean_history = []
            for msg in recent_history:
                role = msg.get("role", "user")
                content = ensure_utf8(msg.get("content", ""))
                clean_history.append({"role": role, "content": content})
            messages.extend(clean_history)
        
        messages.append({"role": "user", "content": user_prompt})
        return messages
    
    def answer(
        self, 
        query: str, 
        chat_history: Optional[List[Dict[str, str]]] = None,
        session_id: Optional[str] = None,
        max_tokens: int = 1000, 
        min_score: float = 0.5, 
        debug: bool = False
    ) -> Tuple[str, List[str], List[Dict[str, Any]]]:
        """Answer a query with multi-turn reasoning support
        
        Returns:
            tuple: (answer_text, list_of_source_document_names, list_of_chunks_with_metadata)
        """
        if chat_history is None:
            chat_history = []
        
        # Ensure query is UTF-8 safe at the start
        query = ensure_utf8(query)
        
        intent, confidence = self.intent_classifier.predict(query)
        
        if intent == "casual" and confidence >= self.casual_threshold:
            casual_answer = self._generate_casual_reply(query, chat_history)
            return casual_answer, [], []  # No sources for casual replies
        
        retrieved_docs_with_scores = self.retrieve(query, min_score=min_score, session_id=session_id)
        
        if not retrieved_docs_with_scores:
            messages = [
                {"role": "system", "content": "You are a helpful assistant. The user's query doesn't match any documents in the knowledge base, but you can still help based on general knowledge and conversation context."}
            ]
            if chat_history:
                # Ensure all chat history content is UTF-8 safe
                clean_history = []
                for msg in chat_history[-6:]:
                    role = msg.get("role", "user")
                    content = ensure_utf8(msg.get("content", ""))
                    clean_history.append({"role": role, "content": content})
                messages.extend(clean_history)
            messages.append({"role": "user", "content": query})
            
            response = self.llm_provider.generate(messages, max_tokens=max_tokens, temperature=0.7)
            # Clean response and ensure UTF-8 safety
            cleaned_response = self.clean_response(response)
            # Ensure the final string is UTF-8 safe
            final_response = f"I couldn't find specific information about this in the knowledge base. However, {cleaned_response.lower()}"
            return ensure_utf8(final_response), [], []  # No sources when no docs retrieved
        
        retrieved_docs = self.rerank_with_cross_encoder(query, retrieved_docs_with_scores)
        
        # Extract unique source document names and prepare chunks with metadata
        source_docs = set()
        chunks_with_metadata = []
        
        # Create a mapping of (content_hash, source) to score for later use
        # Use a hash of content + source to match documents after reranking
        doc_to_score = {}
        for doc, score in retrieved_docs_with_scores:
            doc_content = ensure_utf8(doc.page_content)
            source = ensure_utf8(str(doc.metadata.get('source', 'Unknown')))
            # Use first 100 chars + source as key (sufficient for matching)
            key = (doc_content[:100], source)
            doc_to_score[key] = float(score)
        
        total_chars = 0
        chunks_for_prompt = []
        for doc in retrieved_docs:
            # Ensure document content is UTF-8 safe
            doc_content = ensure_utf8(doc.page_content)
            source = doc.metadata.get('source', 'Unknown')
            if source and source != 'Unknown':
                source_docs.add(ensure_utf8(str(source)))
            
            # Match score using content hash
            source_str = ensure_utf8(str(source)) if source else "Unknown"
            key = (doc_content[:100], source_str)
            matched_score = doc_to_score.get(key, 0.0)
            
            # Store chunk metadata
            chunk_info = {
                "source": source_str,
                "content": doc_content[:500] + "..." if len(doc_content) > 500 else doc_content,  # Truncate for display
                "score": matched_score,
                "full_content": doc_content  # Keep full content for detailed view
            }
            chunks_with_metadata.append(chunk_info)
            
            doc_len = len(doc_content)
            if total_chars + doc_len > MAX_CONTEXT_TOKENS * AVG_CHARS_PER_TOKEN:
                break
            chunks_for_prompt.append(doc_content)
            total_chars += doc_len
        
        context = "\n\n".join([f"[Document: {i+1}]\n{chunk}" for i, chunk in enumerate(chunks_for_prompt)])
        
        messages = self._build_agentic_prompt(query, context, chat_history)
        
        response = self.llm_provider.generate(messages, max_tokens=max_tokens, temperature=0.7)
        cleaned_text = self.clean_response(response)
        # Ensure final return value is UTF-8 safe
        cleaned_text = ensure_utf8(cleaned_text)
        
        if debug:
            try:
                print(f"Intent classification: {intent} (confidence: {confidence:.3f})")
                print("=== Retrieved Chunks ===")
                for doc, score in retrieved_docs_with_scores:
                    safe_content = ensure_utf8(doc.page_content[:300].replace('\n', ' '))
                    safe_source = ensure_utf8(str(doc.metadata.get('source','N/A')))
                    print(f"Score: {score:.4f} | Source: {safe_source}")
                    print(safe_content + "...\n")
                print("=== LLM Response ===")
                print(cleaned_text)
            except Exception:
                pass  # Ignore print errors in debug mode
        
        return cleaned_text, list(source_docs), chunks_with_metadata
