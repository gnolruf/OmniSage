import psycopg2
from psycopg2.extras import RealDictCursor
from typing import List, Dict, Optional

class DatabaseManager:
    def __init__(self, connection_string: str):
        """Initialize database connection and ensure tables exist."""
        self.conn_string = connection_string
        self.conn = psycopg2.connect(connection_string)
        self._create_tables()
    
    def _create_tables(self):
        """Create necessary database tables if they don't exist."""
        with self.conn.cursor() as cur:
            # Create chats table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chats (
                    id SERIAL PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create messages table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id SERIAL PRIMARY KEY,
                    chat_id INTEGER REFERENCES chats(id) ON DELETE CASCADE,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    model_group TEXT,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    CONSTRAINT valid_role CHECK (role IN ('user', 'assistant'))
                )
            """)
            
            self.conn.commit()
    
    def create_chat(self, title: str) -> int:
        """Create a new chat session and return its ID."""
        with self.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO chats (title) VALUES (%s) RETURNING id",
                (title,)
            )
            chat_id = cur.fetchone()[0]
            self.conn.commit()
            return chat_id
    
    def save_message(self, chat_id: int, role: str, content: str, model_group: Optional[str] = None):
        """Save a new message to a chat session."""
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO messages (chat_id, role, content, model_group)
                VALUES (%s, %s, %s, %s)
            """, (chat_id, role, content, model_group))
            
            # Update chat's updated_at timestamp
            cur.execute("""
                UPDATE chats 
                SET updated_at = CURRENT_TIMESTAMP 
                WHERE id = %s
            """, (chat_id,))
            
            self.conn.commit()
    
    def get_chats(self) -> List[Dict]:
        """Get all chat sessions with their latest message."""
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT 
                    c.id,
                    c.title,
                    c.created_at,
                    c.updated_at,
                    COALESCE(m.content, '') as latest_message
                FROM chats c
                LEFT JOIN (
                    SELECT DISTINCT ON (chat_id)
                        chat_id,
                        content
                    FROM messages
                    ORDER BY chat_id, created_at DESC
                ) m ON m.chat_id = c.id
                ORDER BY c.updated_at DESC
            """)
            return cur.fetchall()
    
    def get_chat_messages(self, chat_id: int) -> List[Dict]:
        """Get all messages for a specific chat."""
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT role, content, model_group, created_at
                FROM messages
                WHERE chat_id = %s
                ORDER BY created_at
            """, (chat_id,))
            return cur.fetchall()
    
    def delete_chat(self, chat_id: int):
        """Delete a chat and all its messages."""
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM chats WHERE id = %s", (chat_id,))
            self.conn.commit()
    
    def close(self):
        """Close the database connection."""
        self.conn.close()