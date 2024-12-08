-- Note: This must be run as a postgres superuser
CREATE DATABASE llamachat;

-- Connect to the database
\c llamachat

-- Create tables
CREATE TABLE IF NOT EXISTS chats (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS messages (
    id SERIAL PRIMARY KEY,
    chat_id INTEGER REFERENCES chats(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    model_group TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT valid_role CHECK (role IN ('user', 'assistant'))
);

-- Create index for faster chat message retrieval
CREATE INDEX idx_messages_chat_id ON messages(chat_id);

-- Create index for timestamp-based queries
CREATE INDEX idx_chats_updated_at ON chats(updated_at DESC);

-- Create a function to update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create a trigger to automatically update the updated_at column
CREATE TRIGGER update_chats_updated_at
    BEFORE UPDATE ON chats
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();