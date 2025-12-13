"""
File processing utilities for multiple file formats
"""
import os
import re
from pathlib import Path
from typing import List
from langchain.docstore.document import Document


def ensure_utf8(text: str) -> str:
    """Ensure text is UTF-8 safe by encoding and decoding with error handling"""
    if not isinstance(text, str):
        text = str(text)
    try:
        return text.encode('utf-8', errors='replace').decode('utf-8')
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')


def extract_text_from_file(file_path: Path) -> str:
    """Extract text content from various file formats"""
    file_extension = file_path.suffix.lower()
    
    if file_extension == '.txt':
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
            return ensure_utf8(content)
    
    elif file_extension in ('.md', '.markdown'):
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
            try:
                import markdown
                html = markdown.markdown(content)
                text = re.sub(r'<[^>]+>', '', html)
                return ensure_utf8(text)
            except Exception:
                return ensure_utf8(content)
    
    elif file_extension == '.pdf':
        try:
            import PyPDF2
            text_parts = []
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        # Ensure PDF text is UTF-8 safe
                        page_text = ensure_utf8(page_text)
                        text_parts.append(page_text)
            result = '\n'.join(text_parts)
            return ensure_utf8(result)
        except Exception as e:
            error_msg = str(e)
            safe_error = ensure_utf8(error_msg)
            raise Exception(f"Error reading PDF: {safe_error}")
    
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")


def process_uploaded_file(file_path: Path, original_filename: str, kb_folder: str = "KB") -> Document:
    """Process a single uploaded file and return a Document"""
    try:
        content = extract_text_from_file(file_path)
        if not content or not content.strip():
            safe_filename = ensure_utf8(original_filename)
            raise ValueError(f"File {safe_filename} is empty or could not be processed")
        
        # Ensure content is UTF-8 safe
        content = ensure_utf8(content.strip())
        rel_path = ensure_utf8(os.path.relpath(file_path, kb_folder))
        safe_filename = ensure_utf8(original_filename)
        
        return Document(
            page_content=content,
            metadata={
                "source": rel_path,
                "filename": safe_filename,
                "file_type": file_path.suffix.lower()
            }
        )
    except Exception as e:
        safe_filename = ensure_utf8(original_filename)
        error_msg = ensure_utf8(str(e))
        raise Exception(f"Error processing file {safe_filename}: {error_msg}")


def process_folder(folder_path: Path, kb_folder: str = "KB") -> List[Document]:
    """Recursively process all supported files in a folder"""
    documents = []
    supported_extensions = {'.txt', '.md', '.markdown', '.pdf'}
    
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = Path(root) / file
            if file_path.suffix.lower() in supported_extensions:
                try:
                    doc = process_uploaded_file(file_path, file, kb_folder)
                    documents.append(doc)
                except Exception as e:
                    try:
                        safe_path = ensure_utf8(str(file_path))
                        safe_error = ensure_utf8(str(e))
                        print(f"Warning: Skipping {safe_path}: {safe_error}")
                    except Exception:
                        print("Warning: Skipping file")
                    continue
    
    return documents
