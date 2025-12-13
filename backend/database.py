"""
MongoDB database connection and models for session tracking and analytics
"""
from pymongo import MongoClient
from datetime import datetime
from typing import Optional, List, Dict, Any
import os
from bson import ObjectId


def ensure_utf8(text: str) -> str:
    """Ensure text is UTF-8 safe by encoding and decoding with error handling"""
    if not isinstance(text, str):
        text = str(text)
    try:
        return text.encode('utf-8', errors='replace').decode('utf-8')
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')


class Database:
    """MongoDB database wrapper for session and message management"""
    
    def __init__(self, connection_string: Optional[str] = None):
        self.connection_string = connection_string or os.getenv(
            "MONGODB_URI", 
            "mongodb://localhost:27017/"
        )
        self.client = MongoClient(self.connection_string)
        self.db = self.client.get_database(os.getenv("MONGODB_DB", "rag_chatbot"))
        self.sessions = self.db.sessions
        self.messages = self.db.messages
        self.analytics = self.db.analytics
        self.knowledge_bases = self.db.knowledge_bases
        
        # Create indexes
        self.sessions.create_index("session_id")
        self.sessions.create_index("created_at")
        self.messages.create_index("session_id")
        self.messages.create_index("timestamp")
        self.analytics.create_index("timestamp")
        self.knowledge_bases.create_index("user_id")
    
    def create_session(self, user_id: Optional[str] = None, metadata: Optional[Dict] = None) -> str:
        """Create a new chat session"""
        session = {
            "session_id": str(ObjectId()),
            "user_id": user_id,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "message_count": 0,
            "metadata": metadata or {}
        }
        self.sessions.insert_one(session)
        return session["session_id"]
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get session by ID"""
        return self.sessions.find_one({"session_id": session_id})
    
    def update_session(self, session_id: str, **kwargs):
        """Update session fields"""
        kwargs["updated_at"] = datetime.utcnow()
        self.sessions.update_one(
            {"session_id": session_id},
            {"$set": kwargs}
        )
    
    def add_message(self, session_id: str, role: str, content: str, metadata: Optional[Dict] = None):
        """Add a message to the database"""
        # Ensure content is UTF-8 safe
        content = ensure_utf8(content)
        session_id = ensure_utf8(session_id)
        role = ensure_utf8(role)
        
        # Ensure metadata values are UTF-8 safe
        safe_metadata = {}
        if metadata:
            for key, value in metadata.items():
                if isinstance(value, str):
                    safe_metadata[key] = ensure_utf8(value)
                else:
                    safe_metadata[key] = value
        
        message = {
            "session_id": session_id,
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow(),
            "metadata": safe_metadata
        }
        self.messages.insert_one(message)
        
        self.sessions.update_one(
            {"session_id": session_id},
            {"$inc": {"message_count": 1}, "$set": {"updated_at": datetime.utcnow()}}
        )
    
    def get_session_messages(self, session_id: str, limit: int = 50) -> List[Dict]:
        """Get messages for a session"""
        # Ensure session_id is UTF-8 safe
        session_id = ensure_utf8(session_id)
        
        messages = list(self.messages.find(
            {"session_id": session_id}
        ).sort("timestamp", 1).limit(limit))
        
        # Ensure all message content is UTF-8 safe
        safe_messages = []
        for msg in messages:
            safe_msg = {}
            for key, value in msg.items():
                if isinstance(value, str):
                    safe_msg[key] = ensure_utf8(value)
                elif isinstance(value, dict):
                    safe_dict = {}
                    for k, v in value.items():
                        if isinstance(v, str):
                            safe_dict[k] = ensure_utf8(v)
                        else:
                            safe_dict[k] = v
                    safe_msg[key] = safe_dict
                else:
                    safe_msg[key] = value
            safe_messages.append(safe_msg)
        
        return safe_messages
    
    def log_analytics(self, event_type: str, data: Dict[str, Any]):
        """Log analytics event"""
        event = {
            "event_type": event_type,
            "timestamp": datetime.utcnow(),
            "data": data
        }
        self.analytics.insert_one(event)
    
    def get_analytics(self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> List[Dict]:
        """Get analytics data with optional date filtering"""
        query = {}
        if start_date:
            query["timestamp"] = {"$gte": start_date}
        if end_date:
            if "timestamp" in query:
                query["timestamp"]["$lte"] = end_date
            else:
                query["timestamp"] = {"$lte": end_date}
        
        return list(self.analytics.find(query).sort("timestamp", -1))
    
    def save_knowledge_base(self, user_id: str, kb_name: str, file_path: str, metadata: Optional[Dict] = None):
        """Save knowledge base information"""
        # Ensure all string fields are UTF-8 safe
        user_id = ensure_utf8(user_id)
        kb_name = ensure_utf8(kb_name)
        file_path = ensure_utf8(file_path)
        
        # Ensure metadata values are UTF-8 safe
        safe_metadata = {}
        if metadata:
            for key, value in metadata.items():
                if isinstance(value, str):
                    safe_metadata[key] = ensure_utf8(value)
                elif isinstance(value, list):
                    safe_metadata[key] = [ensure_utf8(item) if isinstance(item, str) else item for item in value]
                else:
                    safe_metadata[key] = value
        
        kb = {
            "user_id": user_id,
            "kb_name": kb_name,
            "file_path": file_path,
            "created_at": datetime.utcnow(),
            "metadata": safe_metadata
        }
        self.knowledge_bases.insert_one(kb)
        return kb
    
    def get_user_knowledge_bases(self, user_id: str) -> List[Dict]:
        """Get all knowledge bases for a user"""
        return list(self.knowledge_bases.find({"user_id": user_id}).sort("created_at", -1))
