-- Migration 003: Assistant conversation persistence
-- Enables multi-turn Clinical Assistant with conversation history

CREATE TABLE IF NOT EXISTS assistant_conversations (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    encounter_id    TEXT,                       -- links to Supabase encounter (optional)
    patient_context JSONB DEFAULT '{}',         -- snapshot of patient at conversation start
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS assistant_messages (
    id              SERIAL PRIMARY KEY,
    conversation_id UUID NOT NULL REFERENCES assistant_conversations(id) ON DELETE CASCADE,
    role            TEXT NOT NULL,              -- 'user' | 'assistant'
    content         TEXT NOT NULL,              -- the message text
    sources         JSONB DEFAULT '[]',         -- [{stg_code, condition_name, section_role, excerpt}]
    tools_used      JSONB DEFAULT '[]',         -- ["search_guidelines", "lookup_condition"]
    tool_calls      JSONB DEFAULT '[]',         -- full tool call/result pairs for Anthropic replay
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_assistant_messages_conversation
    ON assistant_messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_assistant_conversations_encounter
    ON assistant_conversations(encounter_id);
