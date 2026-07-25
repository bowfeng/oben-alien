use anyhow::{anyhow, Result};
use rusqlite::{params, types::Value, Connection, OptionalExtension};
use tracing::info;

use oben_models::{
    Message, MessageRole, Session, SessionManager, SessionMetadata, SessionSource,
};

fn now_ts() -> f64 {
    chrono::Utc::now().timestamp_millis() as f64 / 1000.0
}

fn ts_to_datetime(ts: f64) -> chrono::DateTime<chrono::Utc> {
    let millis = (ts * 1000.0) as i64;
    chrono::DateTime::<chrono::Utc>::from_timestamp_millis(millis).unwrap_or(chrono::Utc::now())
}

fn parse_source(s: &str) -> SessionSource {
    match s {
        "cli" => SessionSource::Cli,
        "gateway" => SessionSource::Gateway,
        "telegram" => SessionSource::Telegram,
        "discord" => SessionSource::Discord,
        "slack" => SessionSource::Slack,
        "web" => SessionSource::Web,
        "tool" => SessionSource::Tool,
        "cron" => SessionSource::Cron,
        "batch" => SessionSource::Batch,
        _ => SessionSource::Cli,
    }
}

const SCHEMA_SQL: &str = "
CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL DEFAULT 'cli',
    model TEXT,
    model_config TEXT,
    system_prompt TEXT,
    parent_session_id TEXT,
    started_at REAL NOT NULL,
    ended_at REAL,
    end_reason TEXT,
    title TEXT UNIQUE,
    preview TEXT,
    message_count INTEGER DEFAULT 0,
    tool_call_count INTEGER DEFAULT 0,
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    cache_read_tokens INTEGER DEFAULT 0,
    cache_write_tokens INTEGER DEFAULT 0,
    reasoning_tokens INTEGER DEFAULT 0,
    api_call_count INTEGER DEFAULT 0,
    user_id TEXT,
    estimated_cost_usd REAL DEFAULT 0,
    actual_cost_usd REAL DEFAULT 0,
    cost_status TEXT,
    cost_source TEXT,
    pricing_version TEXT,
    billing_provider TEXT,
    billing_base_url TEXT,
    billing_mode TEXT,
    handoff_state TEXT,
    handoff_platform TEXT,
    handoff_error TEXT,
    compaction_summary TEXT,
    turn_count INTEGER DEFAULT 0,
    FOREIGN KEY (parent_session_id) REFERENCES sessions(id)
);

-- Goals — independent entities, not scoped to sessions.
-- A goal can be owned by one session and referenced by multiple sessions
-- via session_goal_refs. Plan state is stored as JSON on the goal itself.
CREATE TABLE IF NOT EXISTS goals (
    id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',  -- pending, active, done, failed
    owner_session TEXT,                      -- session that created this goal
    max_turns INTEGER DEFAULT 20,
    turns_used INTEGER DEFAULT 0,
    last_verdict TEXT,                       -- last judge verdict
    last_reason TEXT,                        -- last judge reason
    paused_reason TEXT,                      -- why paused (if paused)
    consecutive_parse_failures INTEGER DEFAULT 0,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

-- Goal plan nodes — tree structure stored as JSON in the plan field.
-- This is a denormalized version for status tracking/search.
CREATE TABLE IF NOT EXISTS goal_nodes (
    row_id INTEGER PRIMARY KEY AUTOINCREMENT,
    goal_id TEXT NOT NULL REFERENCES goals(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',  -- pending, in_progress, done, failed
    completion_message TEXT,
    created_at REAL NOT NULL
);

-- Session-goal references — a session can participate in multiple goals.
CREATE TABLE IF NOT EXISTS session_goal_refs (
    session_id TEXT NOT NULL,
    goal_id TEXT NOT NULL REFERENCES goals(id) ON DELETE CASCADE,
    assigned_at REAL,
    PRIMARY KEY (session_id, goal_id)
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    role TEXT NOT NULL,
    content TEXT,
    tool_call_id TEXT,
    tool_calls TEXT,
    tool_name TEXT,
    timestamp REAL NOT NULL,
    token_count INTEGER,
    finish_reason TEXT,
    reasoning TEXT,
    reasoning_content TEXT,
    reasoning_details TEXT,
    codex_reasoning_items TEXT,
    codex_message_items TEXT,
    delegation_id INTEGER,
    tool_error INTEGER NOT NULL DEFAULT 0,
    include_in_prompt INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_sessions_source ON sessions(source);
CREATE INDEX IF NOT EXISTS idx_sessions_parent ON sessions(parent_session_id);
CREATE INDEX IF NOT EXISTS idx_sessions_started ON sessions(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_sessions_title ON sessions(title);
";

const FTS_SQL: &str = "
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(content);

-- Trigram FTS5 table for CJK/Thai/other non-space-delimited scripts.
-- The default unicode61 tokenizer splits CJK characters into individual
-- tokens, breaking phrase matching. The trigram tokenizer creates
-- overlapping 3-byte sequences so substring queries work natively.
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts_trigram USING fts5(
    content,
    tokenize='trigram'
);

-- Sync triggers for default FTS5
CREATE TRIGGER IF NOT EXISTS messages_fts_insert AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts(rowid, content) VALUES (
        new.id,
        COALESCE(new.content, '') || ' ' || COALESCE(new.tool_name, '') || ' ' || COALESCE(new.tool_calls, '')
    );
END;
CREATE TRIGGER IF NOT EXISTS messages_fts_delete AFTER DELETE ON messages BEGIN
    DELETE FROM messages_fts WHERE rowid = old.id;
END;
CREATE TRIGGER IF NOT EXISTS messages_fts_update AFTER UPDATE ON messages BEGIN
    DELETE FROM messages_fts WHERE rowid = old.id;
    INSERT INTO messages_fts(rowid, content) VALUES (
        new.id,
        COALESCE(new.content, '') || ' ' || COALESCE(new.tool_name, '') || ' ' || COALESCE(new.tool_calls, '')
    );
END;

-- Sync triggers for trigram FTS5
CREATE TRIGGER IF NOT EXISTS messages_fts_trigram_insert AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts_trigram(rowid, content) VALUES (
        new.id,
        COALESCE(new.content, '') || ' ' || COALESCE(new.tool_name, '') || ' ' || COALESCE(new.tool_calls, '')
    );
END;
CREATE TRIGGER IF NOT EXISTS messages_fts_trigram_delete AFTER DELETE ON messages BEGIN
    DELETE FROM messages_fts_trigram WHERE rowid = old.id;
END;
CREATE TRIGGER IF NOT EXISTS messages_fts_trigram_update AFTER UPDATE ON messages BEGIN
    DELETE FROM messages_fts_trigram WHERE rowid = old.id;
    INSERT INTO messages_fts_trigram(rowid, content) VALUES (
        new.id,
        COALESCE(new.content, '') || ' ' || COALESCE(new.tool_name, '') || ' ' || COALESCE(new.tool_calls, '')
    );
END;
";

fn is_retryable_error(error: &anyhow::Error) -> bool {
    let msg = error.to_string().to_lowercase();
    msg.contains("locked") || msg.contains("busy") || msg.contains("database is locked")
}

/// Schema version for data migrations (not column additions).
const SCHEMA_VERSION: u32 = 3;

fn reconcile_schema(conn: &Connection) -> Result<()> {
    // Ensure schema_version table exists
    conn.execute_batch("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)")?;

    // Check current version
    let current_version: u32 = conn
        .query_row("SELECT version FROM schema_version LIMIT 1", [], |row| {
            row.get(0)
        })
        .unwrap_or(0);

    // Declarative column reconciliation: compare SCHEMA_SQL against live tables
    // and ADD any missing columns. This makes adding a column as simple as
    // editing SCHEMA_SQL — no version-gated migrations needed.
    let expected = parse_expected_columns(SCHEMA_SQL)?;
    for (table_name, expected_cols) in &expected {
        let live_cols = get_live_columns(conn, table_name)?;
        for (col_name, col_type) in expected_cols {
            if !live_cols.contains(col_name) {
                let _ = conn.execute(
                    &format!(
                        "ALTER TABLE \"{}\" ADD COLUMN \"{}\" {}",
                        table_name, col_name, col_type
                    ),
                    [],
                );
            }
        }
    }

    // Data migrations (version-gated)
    if current_version < 3 {
        migrate_v2_to_v3(conn)?;
    }

    if current_version < SCHEMA_VERSION {
        conn.execute("UPDATE schema_version SET version = ?", [SCHEMA_VERSION])?;
    }

    Ok(())
}

/// Parse expected columns from SCHEMA_SQL by executing it in memory
/// and using PRAGMA table_info. This handles all SQL syntax correctly
/// (DEFAULT expressions, inline REFERENCES, CHECK constraints, etc.)
fn parse_expected_columns(schema_sql: &str) -> Result<Vec<(String, Vec<(String, String)>)>> {
    let ref_conn = Connection::open_in_memory()?;
    ref_conn.execute_batch(schema_sql)?;

    let mut result = Vec::new();
    let mut stmt = ref_conn.prepare(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'",
    )?;
    let tables: Vec<String> = stmt
        .query_map([], |row| row.get(0))?
        .filter_map(|r| r.ok())
        .collect();

    for table_name in tables {
        let mut stmt2 = ref_conn.prepare(&format!("PRAGMA table_info(\"{}\")", table_name))?;
        let mut cols = Vec::new();
        for row in stmt2.query_map([], |row| {
            Ok((row.get::<_, String>(1)?, row.get::<_, String>(2)?))
        })? {
            if let Ok((name, col_type)) = row {
                cols.push((name, col_type));
            }
        }
        result.push((table_name, cols));
    }

    Ok(result)
}

/// Migrate sessions table v2→v3: add UNIQUE constraint on title column.
///
/// SQLite does not support ALTER TABLE ADD CONSTRAINT, so we rebuild the
/// table: create new with UNIQUE, INSERT deduplicated rows (keep latest
/// `updated_at` when title collides), drop old table, rename new one.
///
/// If `sessions` table doesn't exist yet this is a fresh DB (the schema was
/// already created with UNIQUE by the caller), so we return early.
fn migrate_v2_to_v3(conn: &Connection) -> Result<()> {
    // Skip if already v3+ (sessions table already has UNIQUE on title).
    // Check by looking for any autoindex on sessions (SQLite creates one for UNIQUE).
    let has_autoindex: bool = conn.query_row(
        "SELECT COUNT(*) > 0 FROM sqlite_master WHERE type='index' AND name LIKE 'sqlite_autoindex_sessions%'",
        [],
        |row| row.get(0),
    )?;

    if has_autoindex {
        // Already v3 with UNIQUE, nothing to migrate
        return Ok(());
    }

    conn.execute_batch(
        "DROP TABLE IF EXISTS sessions_new; CREATE TABLE sessions_new (
            id TEXT PRIMARY KEY,
            source TEXT NOT NULL DEFAULT 'cli',
            model TEXT,
            model_config TEXT,
            system_prompt TEXT,
            parent_session_id TEXT,
            started_at REAL NOT NULL,
            ended_at REAL,
            end_reason TEXT,
            title TEXT UNIQUE,
            preview TEXT,
            message_count INTEGER DEFAULT 0,
            tool_call_count INTEGER DEFAULT 0,
            input_tokens INTEGER DEFAULT 0,
            output_tokens INTEGER DEFAULT 0,
            cache_read_tokens INTEGER DEFAULT 0,
            cache_write_tokens INTEGER DEFAULT 0,
            reasoning_tokens INTEGER DEFAULT 0,
            api_call_count INTEGER DEFAULT 0,
            user_id TEXT,
            estimated_cost_usd REAL DEFAULT 0.0,
            actual_cost_usd REAL DEFAULT 0.0,
            cost_status TEXT,
            cost_source TEXT,
            pricing_version TEXT,
            billing_provider TEXT,
            billing_base_url TEXT,
            billing_mode TEXT,
            handoff_state TEXT,
            handoff_platform TEXT,
            handoff_error TEXT,
            FOREIGN KEY (parent_session_id) REFERENCES sessions(id)
        );",
    )?;

    // Deduplicate: for conflicting titles keep the row with the latest
    // `updated_at` (i.e. the newest session).
    conn.execute(
        "INSERT OR IGNORE INTO sessions_new SELECT * FROM sessions WHERE id IN (SELECT s1.id FROM sessions s1 LEFT JOIN sessions s2 ON s1.title = s2.title AND s1.started_at < s2.started_at AND s1.id != s2.id WHERE s2.id IS NULL)",
        [],
    )?;

    let _ = conn.execute_batch("DROP TABLE sessions");
    conn.execute("ALTER TABLE sessions_new RENAME TO sessions", [])?;
    info!("Migrated sessions v2→v3 with UNIQUE title constraint");
    Ok(())
}

/// Get live column names for a table via PRAGMA table_info.
fn get_live_columns(conn: &Connection, table_name: &str) -> Result<Vec<String>> {
    let mut stmt = conn.prepare(&format!("PRAGMA table_info(\"{}\")", table_name))?;
    let mut cols = Vec::new();
    for row in stmt.query_map([], |row| row.get::<_, String>(1))? {
        if let Ok(name) = row {
            cols.push(name);
        }
    }
    Ok(cols)
}

fn row_to_message(row: &rusqlite::Row) -> std::result::Result<Message, rusqlite::Error> {
    let role_str: String = row.get("role")?;
    let role = match role_str.as_str() {
        "system" => MessageRole::System,
        "user" => MessageRole::User,
        "assistant" => MessageRole::Assistant,
        "tool" => MessageRole::Tool,
        _ => MessageRole::User,
    };
    let content: String = row.get("content").unwrap_or_default();
    let tool_calls: Option<String> = row.get("tool_calls").ok();
    let tool_calls = tool_calls.and_then(|tc| serde_json::from_str(&tc).ok());
    let tool_call_id: Option<String> = row.get("tool_call_id").ok();
    let id: Option<i64> = row.get("id").ok();
    let reasoning: Option<String> = row.get("reasoning").ok();
    let delegation_id: Option<i32> = row.get("delegation_id").ok();
    let tool_error: i32 = row.get("tool_error").unwrap_or(0);
    let include_in_prompt: i32 = row.get("include_in_prompt").unwrap_or(1);
    Ok(Message {
        role,
        content: oben_models::MessageContent::Text(content),
        id,
        tool_call_ids: tool_call_id.into_iter().collect(),
        tool_calls,
        reasoning,
        delegation_id: delegation_id.map(|v| v as u32),
        tool_error: tool_error != 0,
        include_in_prompt: include_in_prompt != 0,
    })
}

pub struct SessionDB {
    _db_path: std::path::PathBuf,
    conn: std::sync::Mutex<Option<Connection>>,
    /// Write serialization lock — separates WAL write-lock acquisition
    /// from connection locking, so concurrent async tasks don't hold
    /// the connection mutex across long sleeps (retry jitter).
    write_lock: std::sync::Mutex<()>,
    /// Counter for periodic WAL checkpointing (every N writes).
    write_count: std::sync::atomic::AtomicUsize,
}

// ── Write contention tuning (mirrors hermes_state.py) ─────────────────────
/// Max retries before giving up on a locked DB.
const WRITE_MAX_RETRIES: usize = 15;
/// Min sleep between retries (20ms) — breaks convoy effects that
/// SQLite's deterministic backoff schedule creates.
const WRITE_RETRY_MIN_S: f64 = 0.020;
/// Max sleep between retries (150ms).
const WRITE_RETRY_MAX_S: f64 = 0.150;
/// Checkpoint WAL every N successful writes.
const CHECKPOINT_EVERY_N_WRITES: usize = 50;

impl SessionDB {
    pub fn new<P: AsRef<std::path::Path>>(db_path: P) -> Result<Self> {
        let path = db_path.as_ref().to_path_buf();
        let db_dir = path.parent().unwrap_or(path.as_ref());
        std::fs::create_dir_all(db_dir)?;
        let conn = Connection::open(&path)?;

        // WAL mode with short timeout — application-level retry handles
        // contention instead of sitting in SQLite's busy handler (deterministic
        // schedule causes convoy effects under load). Timeout of 1s means SQLite
        // waits 1s internally before returning DatabaseBusy, much faster than
        // the default 30s.
        //
        // WAL mode requires shared-memory (mmap) coordination and fcntl
        // byte-range locks that don't reliably work on network filesystems
        // (NFS, SMB/CIFS, some FUSE mounts). Fall back to DELETE mode.
        let journal_mode = Self::try_set_journal_mode(&conn);

        conn.execute_batch(&format!(
            "PRAGMA foreign_keys=ON; \
                 PRAGMA journal_mode={}; \
                 PRAGMA journal_size_limit=1000000; \
                 PRAGMA synchronous=NORMAL; \
                 PRAGMA busy_timeout=1000;",
            journal_mode
        ))?;
        conn.execute_batch(SCHEMA_SQL)?;
        conn.execute_batch(FTS_SQL)?;
        reconcile_schema(&conn)?;

        let mode_label = if journal_mode == "wal" {
            "WAL (fast)"
        } else {
            "DELETE (NFS-safe)"
        };
        info!(
            "Opened session DB at {} [journal_mode={}]",
            path.display(),
            mode_label
        );

        Ok(Self {
            _db_path: path,
            conn: std::sync::Mutex::new(Some(conn)),
            write_lock: std::sync::Mutex::new(()),
            write_count: std::sync::atomic::AtomicUsize::new(0),
        })
    }

    /// Try WAL mode first; fall back to DELETE on network filesystems.
    ///
    /// Returns the journal mode actually set ("wal" or "delete").
    /// Try WAL mode first; fall back to DELETE on network filesystems.
    /// Returns the journal mode actually set ("wal" or "delete").
    fn try_set_journal_mode(conn: &Connection) -> &'static str {
        // Attempt WAL mode first
        if conn.execute_batch("PRAGMA journal_mode=WAL").is_ok() {
            // Verify it actually took effect
            if let Ok(mode) = conn.query_row("PRAGMA journal_mode", [], |row: &rusqlite::Row| {
                row.get::<_, String>(0)
            }) {
                if mode != "wal" {
                    info!("WAL mode unavailable, falling back to DELETE");
                    let _ = conn.execute_batch("PRAGMA journal_mode=DELETE");
                    return "delete";
                }
            }
            "wal"
        } else {
            // WAL setup failed — fall back to DELETE mode
            info!(
                "WAL not supported on this filesystem, falling back to DELETE. \
                   Note: journal_mode=DELETE works on NFS/SMB/FUSE but reduces \
                   concurrency — concurrent readers are blocked during writes."
            );
            let _ = conn.execute_batch("PRAGMA journal_mode=DELETE");
            "delete"
        }
    }

    fn with_conn<F, T>(&self, f: F) -> Result<T>
    where
        F: FnOnce(&Connection) -> Result<T>,
    {
        let lock = self.conn.lock().unwrap();
        match lock.as_ref() {
            Some(c) => f(c),
            None => Err(anyhow!("database connection is closed")),
        }
    }

    /// Execute a write transaction with BEGIN IMMEDIATE and jittered retry.
    ///
    /// `BEGIN IMMEDIATE` acquires the WAL write lock at transaction start
    /// (not at commit time), so lock contention surfaces immediately.
    /// On `database is locked` / `DatabaseBusy`, we sleep a random 20-150ms
    /// and retry — breaking the convoy pattern that SQLite's built-in
    /// deterministic backoff schedule creates under high concurrency.
    fn with_conn_mut<F, T>(&self, mut f: F) -> Result<T>
    where
        F: FnMut(&mut Connection) -> Result<T>,
    {
        let mut last_err: Option<String> = None;

        for attempt in 0..WRITE_MAX_RETRIES {
            // Acquire the write serialization lock so only one WAL transaction
            // runs at a time. Other threads wait on this lock, NOT on the
            // connection mutex — keeping the connection mutex scope minimal
            // (only for the actual DB operation, not for the sleep between retries).
            let _wl = self.write_lock.lock().unwrap();

            {
                let mut lock = self.conn.lock().unwrap();
                let conn = lock
                    .as_mut()
                    .ok_or_else(|| anyhow::anyhow!("database connection is closed"))?;

                // BEGIN IMMEDIATE acquires the WAL write lock at transaction start.
                // On contention this fails immediately (not wait 1s for busy_timeout),
                // so we can retry with jitter right away.
                if conn.execute_batch("BEGIN IMMEDIATE").is_err() {
                    last_err = Some("BEGIN IMMEDIATE failed".to_string());
                    continue;
                }

                match f(conn) {
                    Ok(result) => {
                        // Success — commit
                        if conn.execute_batch("COMMIT").is_ok() {
                            return Ok(result);
                        }
                        // COMMIT failed — rollback and retry
                        let _ = conn.execute_batch("ROLLBACK");
                        last_err = Some("COMMIT failed".to_string());
                    }
                    Err(e) => {
                        // Operation failed — rollback
                        let _ = conn.execute_batch("ROLLBACK");
                        // f() returns anyhow::Error — check the message string
                        let err_msg = e.to_string();
                        if !is_retryable_error(&e) {
                            return Err(e); // Re-throw non-lock errors immediately
                        }
                        last_err = Some(err_msg);
                    }
                }
            }

            // On retryable error, sleep with jitter and try again
            if let Some(ref _err_msg) = last_err {
                if attempt < WRITE_MAX_RETRIES - 1 {
                    let jitter = rand::random::<f64>() * (WRITE_RETRY_MAX_S - WRITE_RETRY_MIN_S)
                        + WRITE_RETRY_MIN_S;
                    std::thread::sleep(std::time::Duration::from_secs_f64(jitter));
                    continue;
                }
            }
        }

        Err(anyhow::anyhow!(
            "database is locked after max retries: {:?}",
            last_err
        ))
    }

    /// Record a successful write and checkpoint if needed.
    fn record_write(&self) {
        let count = self
            .write_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            + 1;
        if count % CHECKPOINT_EVERY_N_WRITES == 0 {
            self.try_wal_checkpoint();
        }
    }

    fn try_wal_checkpoint(&self) {
        // Best-effort PASSIVE WAL checkpoint.  Never blocks, never raises.
        // Flushes committed WAL frames back into the main DB file for any
        // frames that no other connection currently needs.  Keeps the WAL
        // from growing unbounded when many processes hold persistent connections.
        if let Ok(lock) = self.conn.lock() {
            if let Some(ref conn) = lock.as_ref() {
                let _ = conn.execute_batch("PRAGMA wal_checkpoint(PASSIVE)");
            }
        }
    }

    /// Sanitize a session title: strip control chars, collapse whitespace,
    /// enforce max length. Matches Hermes' `sanitize_title()`.
    pub fn sanitize_title(title: &str) -> Option<String> {
        let max_len = 100;
        let cleaned: String = title
            .chars()
            .filter(|c| {
                // Allow printable ASCII and common Unicode (strip control chars)
                !matches!(
                    c,
                    '\x00'
                        | '\x01'
                        | '\x02'
                        | '\x03'
                        | '\x04'
                        | '\x05'
                        | '\x06'
                        | '\x07'
                        | '\x08'
                        | '\x0b'
                        | '\x0c'
                        | '\x0e'
                        | '\x0f'
                        | '\x10'
                        | '\x11'
                        | '\x12'
                        | '\x13'
                        | '\x14'
                        | '\x15'
                        | '\x16'
                        | '\x17'
                        | '\x18'
                        | '\x19'
                        | '\x1a'
                        | '\x1b'
                        | '\x1c'
                        | '\x1d'
                        | '\x1e'
                        | '\x1f'
                        | '\x7f'
                )
            })
            .collect();
        let cleaned: String = cleaned
            .chars()
            .filter(|c| {
                // Strip zero-width spaces, line/paragraph separators, BOM, etc.
                !matches!(
                    c,
                    '\u{200b}'
                        | '\u{200c}'
                        | '\u{200d}'
                        | '\u{200e}'
                        | '\u{200f}'
                        | '\u{2028}'
                        | '\u{2029}'
                        | '\u{2060}'
                        | '\u{2061}'
                        | '\u{2062}'
                        | '\u{2063}'
                        | '\u{2064}'
                        | '\u{feff}'
                        | '\u{fffc}'
                        | '\u{fffd}'
                        | '\u{fffe}'
                )
            })
            .collect();
        // Collapse whitespace, remove '#' characters, rejoin with single space
        let cleaned: String = cleaned
            .split(|c: char| c.is_whitespace() || c == '#')
            .filter(|s| !s.is_empty())
            .collect::<Vec<&str>>()
            .join(" ");
        let cleaned = cleaned.trim();

        if cleaned.is_empty() {
            return None;
        }

        // Truncate to max length at character boundary
        let truncated: String = cleaned.chars().take(max_len).collect();
        Some(truncated)
    }

    /// Resolve a title to the latest session in its lineage.
    ///
    /// If the exact title exists, returns that session's ID.
    /// If not, searches for "title #N" variants and returns the latest one.
    pub fn resolve_session_by_title(&self, title: &str) -> Result<Option<String>> {
        self.with_conn(|conn| {
            // First try exact match
            if let Some(id) = conn.query_row(
                "SELECT id FROM sessions WHERE title = ?",
                params![title],
                |row| row.get::<_, String>(0),
            ).optional()? {
                return Ok(Some(id));
            }
            
            // Search for numbered variants: "title #2", "title #3", etc.
            // Escape SQL LIKE wildcards
            let escaped = title
                .replace('\\', "\\\\")
                .replace('%', "\\%")
                .replace('_', "\\_");
            
            match conn.query_row(
                "SELECT id FROM sessions WHERE title LIKE ? ESCAPE '\\\\' ORDER BY started_at DESC LIMIT 1",
                params![format!("{} #%%", escaped)],
                |row| row.get::<_, String>(0),
            ).optional()? {
                Some(id) => Ok(Some(id)),
                None => {
                    // Also try without the #N suffix
                    let escaped2 = escaped.replace("#%", "%");
                    match conn.query_row(
                        "SELECT id FROM sessions WHERE title LIKE ? ESCAPE '\\\\' ORDER BY started_at DESC LIMIT 1",
                        params![format!("{}%", escaped2)],
                        |row| row.get::<_, String>(0),
                    ).map_err(|e| anyhow::anyhow!(e))? {
                        id => Ok(Some(id)),
                    }
                }
            }
        })
    }

    /// Generate the next title in a lineage: "my task" → "my task #2".
    pub fn get_next_title_in_lineage(&self, base_title: &str) -> Result<String> {
        self.with_conn(|conn| {
            // Strip existing #N suffix to find the true base
            let base = if let Some(m) = base_title.rfind(" #") {
                base_title[..m].to_string()
            } else {
                base_title.to_string()
            };

            let escaped = base
                .replace('\\', "\\\\")
                .replace('%', "\\%")
                .replace('_', "\\_");

            let max_num: Option<i32> = conn
                .query_row(
                    "SELECT MAX(CAST(SUBSTR(title, INSTR(title, '# ') + 2) AS INTEGER)) \
                 FROM sessions WHERE title LIKE ? ESCAPE '\\\\'",
                    params![format!("{} #%%", escaped)],
                    |row| row.get::<_, Option<i32>>(0),
                )
                .map_err(|e| anyhow::anyhow!(e))?;

            let next = max_num.map(|n| n + 1).unwrap_or(1);
            if next <= 1 {
                Ok(base) // First instance — no suffix needed
            } else {
                Ok(format!("{} #{}", base, next))
            }
        })
    }

    pub fn get_or_create_session(&self, name: &str) -> Result<Session> {
        let sanitized = Self::sanitize_title(name);
        let title = sanitized.clone().unwrap_or_else(|| name.to_string());

        let session = self.with_conn(|conn| {
            // Try exact title match first
            let id: Option<String> = conn
                .query_row(
                    "SELECT id FROM sessions WHERE COALESCE(title, '') = ? LIMIT 1",
                    params![&title],
                    |row| row.get(0),
                )
                .ok();
            match id {
                Some(id) => self.session_from_id(conn, &id),
                None => self.create_session_record(conn, &title),
            }
        })?;
        Ok(session)
    }

    fn create_session_record(&self, conn: &Connection, name: &str) -> Result<Session> {
        let id = uuid::Uuid::new_v4().to_string();
        let now = chrono::Utc::now();
        let ts = now_ts();
        conn.execute(
            "INSERT INTO sessions (id, title, started_at, source, message_count) VALUES (?, ?, ?, 'cli', 0)",
            params![id.clone(), name, ts],
        )?;
        let metadata = SessionMetadata {
            id: id.clone(),
            name: name.to_string(),
            source: SessionSource::Cli,
            title: Some(name.to_string()),
            started_at: now,
            message_count: 0,
            ..Default::default()
        };
        Ok(Session {
            id: id.clone(),
            name: name.to_string(),
            created_at: now,
            updated_at: now,
            messages: Vec::new(),
            memory_context: None,
            summary_chunks: Vec::new(),
            persisted_message_count: 0,
            compaction_summary: None,
            metadata,
        })
    }

    pub fn get_session(&self, session_id: &str) -> Result<Option<Session>> {
        self.with_conn(|conn| {
            let mut stmt = conn.prepare(
                "SELECT id, COALESCE(title, ''), source, model, system_prompt, parent_session_id, started_at, ended_at, end_reason, title, preview, handoff_state, message_count, tool_call_count, input_tokens, output_tokens, handoff_platform, handoff_error, COALESCE(turn_count, 0) AS turn_count FROM sessions WHERE id = ?"
            )?;
            match stmt.query_row([session_id], |row| {
                let source_str: String = row.get("source")?;
                let started_at_f64: f64 = row.get("started_at")?;
                let title: String = row.get("title")?;
                let ended_at: Option<f64> = row.get("ended_at")?;
                Ok(SessionMetadata {
                    id: row.get("id")?, name: title.clone(), source: parse_source(&source_str),
                    model: row.get("model")?, system_prompt: row.get("system_prompt")?,
                    parent_session_id: row.get("parent_session_id")?,
                    started_at: ts_to_datetime(started_at_f64),
                    ended_at: ended_at.map(|v| ts_to_datetime(v)),
                    end_reason: row.get("end_reason")?, title: Some(title),
                    preview: row.get("preview")?,
                    message_count: row.get("message_count")?,
                    tool_call_count: row.get("tool_call_count")?,
                    input_tokens: row.get("input_tokens")?,
                    output_tokens: row.get("output_tokens")?,
                    total_tokens: (row.get::<_, i32>("input_tokens")? as usize) + (row.get::<_, i32>("output_tokens")? as usize),
                    estimated_cost_usd: 0.0,
                    cost_status: "unknown".to_string(),
                    handoff_state: row.get("handoff_state")?,
                    handoff_platform: row.get("handoff_platform")?,
                    handoff_error: row.get("handoff_error")?,
                    origin: None,
                    last_prompt_tokens: 0,
                    is_fresh_reset: false,
                    suspended: false,
                resume_pending: false,
                resume_reason: None,
                last_resume_marked_at: None,
                compaction_summary: None,
                turn_count: row.get("turn_count")?,
            })
            }) {
                Ok(metadata) => Ok(Some(Session {
                    id: metadata.id.clone(), name: metadata.name.clone(),
                    created_at: metadata.started_at, updated_at: metadata.started_at,
                    messages: Vec::new(), memory_context: None, summary_chunks: Vec::new(),
                    persisted_message_count: 0,
                    compaction_summary: None,
                    metadata,
                })),
                Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
                Err(e) => Err(anyhow!(e)),
            }
        })
    }

    fn session_from_id(&self, conn: &Connection, session_id: &str) -> Result<Session> {
        let mut stmt = conn.prepare(
            "SELECT id, COALESCE(title, ''), source, model, system_prompt, parent_session_id, \
             started_at, ended_at, end_reason, title, preview, handoff_state, \
               message_count, tool_call_count, input_tokens, output_tokens, handoff_platform, handoff_error, COALESCE(turn_count, 0) AS turn_count \
               FROM sessions WHERE id = ?"
        )?;
        let row: (
            String,
            String,
            String,
            Option<String>,
            Option<String>,
            Option<String>,
            f64,
            Option<f64>,
            Option<String>,
            Option<String>,
            Option<String>,
            Option<String>,
            usize,
            usize,
            usize,
            usize,
            Option<String>,
            Option<String>,
            u32,
        ) = match stmt.query_row([session_id], |row| {
            Ok((
                row.get(0)?,
                row.get(1)?,
                row.get(2)?,
                row.get(3)?,
                row.get(4)?,
                row.get(5)?,
                row.get(6)?,
                row.get(7)?,
                row.get(8)?,
                row.get(9)?,
                row.get(10)?,
                row.get(11)?,
                row.get(12)?,
                row.get(13)?,
                row.get(14)?,
                row.get(15)?,
                row.get(16)?,
                row.get(17)?,
                row.get(18)?,
            ))
        }) {
            Ok(r) => r,
            Err(rusqlite::Error::QueryReturnedNoRows) => {
                return Err(anyhow!("session not found: {}", session_id))
            }
            Err(e) => return Err(anyhow!(e)),
        };

        let source = parse_source(&row.2);
        let title_str = row.1.clone();
        let started_at = ts_to_datetime(row.6);
        let ended_at = row.7.map(ts_to_datetime);

        let metadata = SessionMetadata {
            id: row.0,
            name: title_str.clone(),
            source,
            model: row.3,
            system_prompt: row.4,
            parent_session_id: row.5,
            started_at,
            ended_at,
            end_reason: row.8,
            title: Some(title_str.clone()),
            preview: row.10,
            handoff_state: row.11,
            handoff_platform: row.16,
            handoff_error: row.17,
            turn_count: row.18,
            message_count: row.12,
            tool_call_count: row.13,
            input_tokens: row.14,
            output_tokens: row.15,
            total_tokens: row.14.saturating_add(row.15),
            estimated_cost_usd: 0.0,
            cost_status: "unknown".to_string(),
            origin: None,
            last_prompt_tokens: 0,
            is_fresh_reset: false,
            suspended: false,
            resume_pending: false,
            resume_reason: None,
            last_resume_marked_at: None,
            compaction_summary: None,
        };

        Ok(Session {
            id: metadata.id.clone(),
            name: metadata.name.clone(),
            created_at: started_at,
            updated_at: started_at,
            messages: Vec::new(),
            memory_context: None,
            summary_chunks: Vec::new(),
            persisted_message_count: 0,
            compaction_summary: None,
            metadata,
        })
    }

    pub fn save_messages(&self, session_id: &str, messages: &mut [Message]) -> Result<()> {
        if messages.is_empty() {
            return Ok(());
        }
        // Append-only strategy: never delete existing messages.
        // For compaction, the caller should first clear messages via
        // `delete_session_messages()` or a dedicated clear path.
        self.save_new_messages(session_id, messages)
    }

    /// Clear all messages for a session (used by compaction).
    /// Unlike `save_messages`, this does NOT delete the session itself.
    pub fn clear_messages(&self, session_id: &str) -> Result<()> {
        self.with_conn_mut(|conn| {
            conn.execute(
                "DELETE FROM messages WHERE session_id = ?",
                params![session_id],
            )?;
            conn.execute(
                "UPDATE sessions SET message_count = 0, ended_at = ? WHERE id = ?",
                params![now_ts(), session_id],
            )?;
            Ok(())
        })
    }

    pub fn save_new_messages(&self, session_id: &str, messages: &mut [Message]) -> Result<()> {
        if messages.is_empty() {
            return Ok(());
        }
        self.record_write();
        self.with_conn_mut(|conn| {
            // No nested transaction — with_conn_mut already manages BEGIN IMMEDIATE / COMMIT
            let mut stmt = conn.prepare(
                "INSERT INTO messages (session_id, role, content, tool_calls, tool_call_id, timestamp, tool_name, reasoning, delegation_id, tool_error) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10) RETURNING id"
            )?;
            for msg in messages.iter_mut() {
                let role = match msg.role {
                    MessageRole::System => "system",
                    MessageRole::User => "user",
                    MessageRole::Assistant => "assistant",
                    MessageRole::Tool => "tool",
                };
                let content = msg.content.to_text();
                let tool_calls = msg.tool_calls.as_ref().map(|c| serde_json::to_string(c).unwrap_or_default());
                let tool_call_id = if msg.tool_call_ids.len() > 0 {
                    Some(msg.tool_call_ids.join(","))
                } else {
                    None
                };
                let reasoning = &msg.reasoning;
                let delegation_id = msg.delegation_id;
                let tool_error = if msg.tool_error { 1 } else { 0 };
                let mut rows = stmt.query(params![session_id, role, content, tool_calls, tool_call_id, now_ts(), msg.tool_calls.as_ref().map(|_| "unknown"), reasoning, delegation_id, tool_error])?;
                if let Some(row) = rows.next()? {
                    msg.id = Some(row.get(0)?);
                }
            }
            drop(stmt);
            // Update session message_count after inserting
            let total: i64 = conn.query_row(
                "SELECT COUNT(*) FROM messages WHERE session_id = ?",
                params![session_id],
                |row| row.get(0),
            )?;
            tracing::debug!("DB::save_new_messages: session_id={} total_msgs={}", session_id, total);
            conn.execute(
                "UPDATE sessions SET message_count = ? WHERE id = ?",
                params![total, session_id],
            )?;
            tracing::debug!("DB::save_new_messages: updated sessions.message_count to {}", total);
            Ok(())
        })
    }

    /// Ensure the session record exists in the sessions table.
    /// Returns immediately if the session already exists.
    pub fn ensure_session_in_db(&self, session_id: &str, metadata: &SessionMetadata) -> Result<()> {
        self.with_conn_mut(|conn| {
            let exists: bool = conn.query_row(
                "SELECT 1 FROM sessions WHERE id = ?",
                params![session_id],
                |row| row.get(0),
            ).unwrap_or(false);
            if exists {
                return Ok(());
            }
            let started_at_ts = metadata.started_at.timestamp_millis() as f64 / 1000.0;
            conn.execute(
                "INSERT INTO sessions (id, name, title, source, started_at, message_count) VALUES (?1, ?2, ?3, ?4, ?5, 0)",
                params![
                    &metadata.id,
                    &metadata.name,
                    metadata.title.as_deref().unwrap_or(&metadata.name),
                    match metadata.source {
                        SessionSource::Cli => "cli",
                        SessionSource::Gateway => "gateway",
                        SessionSource::Telegram => "telegram",
                        SessionSource::Discord => "discord",
                        SessionSource::Slack => "slack",
                        SessionSource::Web => "web",
                        SessionSource::Tool => "tool",
                        SessionSource::Cron => "cron",
                        SessionSource::Batch => "batch",
                        _ => "cli",
                    },
                    started_at_ts,
                ],
            )?;
            Ok(())
        })
    }

    pub fn load_messages(&self, session_id: &str) -> Result<Vec<Message>> {
        self.with_conn(|conn| {
            let mut stmt =
                conn.prepare("SELECT * FROM messages WHERE session_id = ? ORDER BY id")?;
            let mut rows = stmt.query_map(params![session_id], row_to_message)?;
            let mut messages = Vec::new();
            while let Some(row_result) = rows.next() {
                messages.push(row_result?);
            }
            Ok(messages)
        })
    }

    pub fn get_messages_around(
        &self,
        session_id: &str,
        around_message_id: i64,
        window: usize,
    ) -> Result<MessagesAround> {
        if window == 0 {
            return Ok(MessagesAround {
                window: Vec::new(),
                messages_before: 0,
                messages_after: 0,
            });
        }
        self.with_conn(|conn| {
            let anchor_exists = conn
                .query_row(
                    "SELECT 1 FROM messages WHERE id = ? AND session_id = ? LIMIT 1",
                    params![around_message_id, session_id],
                    |row| row.get::<_, i32>(0),
                )
                .optional()?
                .map(|v| v == 1)
                .unwrap_or(false);

            if !anchor_exists {
                return Ok(MessagesAround {
                    window: Vec::new(),
                    messages_before: 0,
                    messages_after: 0,
                });
            }

            let mut stmt = conn.prepare(
                "SELECT * FROM messages WHERE session_id = ? AND id <= ? ORDER BY id DESC LIMIT ?",
            )?;
            let before: Vec<Message> = stmt
                .query_map(
                    params![session_id, around_message_id, window + 1],
                    row_to_message,
                )?
                .filter_map(|r| r.ok())
                .collect::<Vec<_>>();

            let mut stmt = conn.prepare(
                "SELECT * FROM messages WHERE session_id = ? AND id > ? ORDER BY id ASC LIMIT ?",
            )?;
            let after: Vec<Message> = stmt
                .query_map(
                    params![session_id, around_message_id, window],
                    row_to_message,
                )?
                .filter_map(|r| r.ok())
                .collect::<Vec<_>>();

            let after_len = after.len();
            let mut all = before
                .into_iter()
                .rev()
                .chain(after.into_iter())
                .collect::<Vec<_>>();
            if !all.iter().any(|m| m.id == Some(around_message_id)) {
                if let Ok(msg) = conn.query_row(
                    "SELECT * FROM messages WHERE id = ? AND session_id = ?",
                    params![around_message_id, session_id],
                    row_to_message,
                ) {
                    all.push(msg);
                    all.sort_by(|a, b| {
                        let aid = a.id.unwrap_or(0);
                        let bid = b.id.unwrap_or(0);
                        aid.cmp(&bid)
                    });
                }
            }
            let window_len = all.len();
            Ok(MessagesAround {
                window: all,
                messages_before: window_len.saturating_sub(after_len + 1),
                messages_after: after_len,
            })
        })
    }

    pub fn get_anchored_view(
        &self,
        session_id: &str,
        around_message_id: i64,
        window: usize,
        bookend: usize,
    ) -> Result<AnchoredView> {
        let primitive = self.get_messages_around(session_id, around_message_id, window)?;
        if primitive.window.is_empty() {
            return Ok(AnchoredView {
                window: Vec::new(),
                messages_before: 0,
                messages_after: 0,
                bookend_start: Vec::new(),
                bookend_end: Vec::new(),
            });
        }
        let filtered: Vec<Message> = primitive
            .window
            .into_iter()
            .filter(|m| {
                m.id == Some(around_message_id)
                    || matches!(m.role, MessageRole::User | MessageRole::Assistant)
            })
            .collect();
        let first_id = filtered.first().and_then(|m| m.id).unwrap_or(-1);
        let last_id = filtered.last().and_then(|m| m.id).unwrap_or(-1);

        let bookend_start = if first_id > 0 {
            self._load_bookend(session_id, first_id, bookend, true)?
        } else {
            Vec::new()
        };
        let bookend_end = if last_id > 0 {
            self._load_bookend(session_id, last_id, bookend, false)?
        } else {
            Vec::new()
        };

        Ok(AnchoredView {
            window: filtered,
            messages_before: primitive.messages_before,
            messages_after: primitive.messages_after,
            bookend_start,
            bookend_end,
        })
    }

    fn _load_bookend(
        &self,
        session_id: &str,
        id: i64,
        bookend: usize,
        forward: bool,
    ) -> Result<Vec<Message>> {
        if bookend == 0 {
            return Ok(Vec::new());
        }
        let clause = if forward { "id < ?" } else { "id > ?" };
        let order = if forward {
            "ORDER BY id ASC"
        } else {
            "ORDER BY id DESC"
        };
        let sql = format!("SELECT * FROM messages WHERE session_id = ? AND {} AND role IN ('user', 'assistant') AND length(content) > 0 {} LIMIT ?", clause, order);
        self.with_conn(|conn| {
            let mut stmt = conn.prepare(&sql)?;
            let mut rows = stmt.query_map(params![session_id, id, bookend], row_to_message)?;
            let mut result = Vec::new();
            while let Some(row_result) = rows.next() {
                result.push(row_result?);
            }
            if !forward {
                result.reverse();
            }
            Ok(result)
        })
    }

    pub fn list_sessions(
        &self,
        source: Option<&str>,
        exclude_sources: &[&str],
        limit: usize,
        offset: usize,
        include_children: bool,
    ) -> Result<Vec<SessionMetadata>> {
        self.with_conn(|conn| {
            let mut conditions = Vec::new();
            let mut bind: Vec<Value> = Vec::new();
            if !include_children { conditions.push("(s.parent_session_id IS NULL)".to_string()); }
            if let Some(src) = source { conditions.push("s.source = ?".to_string()); bind.push(Value::from(src.to_string())); }
            if !exclude_sources.is_empty() {
                let placeholders: Vec<String> = (0..exclude_sources.len()).map(|_| "?".to_string()).collect();
                conditions.push(format!("s.source NOT IN ({})", placeholders.join(",")));
                for src in exclude_sources { bind.push(Value::from((*src).to_string())); }
            }
            let where_clause = if conditions.is_empty() { String::new() } else { format!("WHERE {}", conditions.join(" AND ")) };
            let query = format!(
                "SELECT id, COALESCE(title, '') AS title, source, model, COALESCE(system_prompt, '') AS system_prompt, parent_session_id, started_at, ended_at, end_reason, title AS display_title, preview, handoff_state, message_count, tool_call_count, input_tokens, output_tokens, handoff_platform, handoff_error, COALESCE(turn_count, 0) AS turn_count \
                 FROM sessions s {} ORDER BY started_at DESC LIMIT ? OFFSET ?",
                where_clause
            );
            bind.push(Value::from(limit as i64));
            bind.push(Value::from(offset as i64));
            let mut stmt = conn.prepare(&query)?;
            let mut rows = stmt.query_map(rusqlite::params_from_iter(bind.iter()), |row| {
                let source_str: String = row.get("source")?;
                let started_at_f64: f64 = row.get("started_at")?;
                let started_at = ts_to_datetime(started_at_f64);
                let title_str: String = row.get("title")?;
                Ok(SessionMetadata {
                    id: row.get("id")?, name: title_str.clone(), source: parse_source(&source_str),
                    model: row.get("model")?, system_prompt: row.get("system_prompt")?,
                    parent_session_id: row.get("parent_session_id")?,
                    started_at, ended_at: None, end_reason: row.get("end_reason")?,
                    title: Some(title_str), preview: row.get("preview")?,
                    message_count: row.get("message_count")?,
                    tool_call_count: row.get("tool_call_count")?,
                    input_tokens: row.get("input_tokens")?,
                    output_tokens: row.get("output_tokens")?,
                    handoff_state: row.get("handoff_state")?,
                    handoff_platform: row.get("handoff_platform")?,
                    handoff_error: row.get("handoff_error")?,
                    total_tokens: (row.get::<_, i32>("input_tokens")? as usize) + (row.get::<_, i32>("output_tokens")? as usize),
                    estimated_cost_usd: 0.0,
                    cost_status: "unknown".to_string(),
                    origin: None,
                    last_prompt_tokens: 0,
                    is_fresh_reset: false,
                    suspended: false,
                    resume_pending: false,
                    resume_reason: None,
                    last_resume_marked_at: None,
                    compaction_summary: None,
                    turn_count: row.get("turn_count")?,
                })
            })?;
            let mut result = Vec::new();
            while let Some(row_result) = rows.next() {
                result.push(row_result?);
            }
            Ok(result)
        })
    }

    pub fn search_messages(
        &self,
        query: &str,
        limit: usize,
        _role_filter: Option<&[&str]>,
    ) -> Result<Vec<SearchHit>> {
        self.with_conn(|conn| {
            let sanitized = query.replace('%', "\\%").replace('_', "\\_");
            let like_query = format!("%{}%", sanitized);
            let query_str = format!(
                "SELECT m.id, m.session_id, m.role, m.content, m.timestamp, COALESCE(s.title, ''), s.started_at, '' \
                 FROM messages m JOIN sessions s ON m.session_id = s.id \
                 WHERE m.content LIKE ? AND s.parent_session_id IS NULL \
                 ORDER BY m.timestamp DESC LIMIT {}", limit
            );
            let mut stmt = conn.prepare(&query_str)?;
            let mut rows = stmt.query_map(params![Value::from(like_query)], |row| {
                let id_int: i64 = row.get(0)?;
                Ok(SearchHit {
                    id: id_int.to_string(), session_id: row.get(1)?,
                    session_title: row.get(5)?, role: row.get(2)?,
                    content: row.get(3)?, timestamp: row.get(4)?,
                    snippet: String::new(),
                })
            })?;
            let mut result = Vec::new();
            while let Some(row_result) = rows.next() {
                result.push(row_result?);
            }
            Ok(result)
        })
    }

    /// Walk the compression-continuation chain and return the tip session ID.
    ///
    /// A compression continuation is a child session where:
    /// 1. The parent's `end_reason = 'compression'`
    /// 2. The child was created AFTER the parent was ended
    ///
    /// This distinguishes compression continuations from delegate subagents
    /// or branch children, which also have `parent_session_id` but were
    /// spawned while the parent was still live.
    pub fn resolve_session_tip(&self, session_id: &str) -> Result<String> {
        self.with_conn(|conn| {
            let mut current = session_id.to_string();
            for _ in 0..100 {
                // Check if current has a parent
                let parent_id: Option<String> = conn.query_row(
                    "SELECT parent_session_id FROM sessions WHERE id = ?",
                    params![&current],
                    |row| row.get::<_, Option<String>>(0),
                )?;

                let parent_id = match parent_id {
                    Some(pid) => pid,
                    None => return Ok(current), // No parent — this is the tip
                };

                // Check if parent ended with compression and child started after
                let parent_info: (Option<String>, Option<f64>) = conn.query_row(
                    "SELECT end_reason, ended_at FROM sessions WHERE id = ?",
                    params![&parent_id],
                    |row| {
                        Ok((
                            row.get::<_, Option<String>>(0)?,
                            row.get::<_, Option<f64>>(1)?,
                        ))
                    },
                )?;

                match parent_info {
                    (Some(end_reason), Some(ended_at)) if end_reason == "compression" => {
                        // Parent ended with compression — walk to children
                        // Started_at >= ended_at means child was created as continuation
                        match conn.query_row(
                            "SELECT id FROM sessions \
                             WHERE parent_session_id = ? \
                             AND started_at >= ? \
                             ORDER BY started_at DESC LIMIT 1",
                            params![&parent_id, ended_at],
                            |row| row.get::<_, String>(0),
                        ) {
                            Ok(child_id) => current = child_id,
                            Err(_) => return Ok(current), // No more children
                        }
                    }
                    _ => return Ok(current), // Parent not compression or data issue
                }
            }
            Ok(current)
        })
    }

    /// Mark orphaned compression continuation sessions as ended.
    ///
    /// Targets child sessions where:
    /// - Parent has end_reason='compression' and is ended
    /// - Child has messages but api_call_count=0 and no end_reason/ended_at
    /// - Child is >7 days old
    pub fn finalize_orphaned_compression_sessions(&self) -> Result<usize> {
        self.with_conn_mut(|conn| {
            let now_ts: f64 = now_ts();
            let cutoff_ts = now_ts - 7.0 * 24.0 * 3600.0; // 7 days ago
            let result = conn.execute(
                "UPDATE sessions \
                 SET ended_at = ?, end_reason = 'orphaned_compression' \
                 WHERE api_call_count = 0 \
                   AND end_reason IS NULL \
                   AND ended_at IS NULL \
                   AND started_at < ? \
                   AND parent_session_id IS NOT NULL \
                   AND EXISTS (
                       SELECT 1 FROM sessions p \
                       WHERE p.id = sessions.parent_session_id \
                         AND p.end_reason = 'compression' \
                         AND p.ended_at IS NOT NULL
                   ) \
                   AND EXISTS (
                       SELECT 1 FROM messages m \
                       WHERE m.session_id = sessions.id
                   )",
                params![now_ts, cutoff_ts],
            );
            match result {
                Ok(count) => Ok(count as usize),
                Err(_) => Ok(0), // Best effort
            }
        })
    }

    /// Remove empty TUI ghost sessions (no messages, no title, >24h old).
    pub fn prune_empty_ghost_sessions(&self) -> Result<usize> {
        self.with_conn_mut(|conn| {
            let cutoff = now_ts() - 24.0 * 3600.0; // 24 hours ago
                                                   // Find ghost sessions
            let mut stmt = conn.prepare(
                "SELECT id FROM sessions \
                 WHERE source = 'tui' \
                   AND title IS NULL \
                   AND ended_at IS NOT NULL \
                   AND started_at < ? \
                   AND NOT EXISTS (
                       SELECT 1 FROM messages WHERE messages.session_id = sessions.id
                   )",
            )?;
            let ghosts: Vec<String> = stmt
                .query_map(params![cutoff], |row| row.get::<_, String>(0))?
                .filter_map(|r| r.ok())
                .collect();

            if ghosts.is_empty() {
                return Ok(0);
            }

            let placeholders: String = ghosts.iter().map(|_| "?").collect::<Vec<_>>().join(",");
            let count = conn.execute(
                &format!("DELETE FROM sessions WHERE id IN ({})", placeholders),
                rusqlite::params_from_iter(ghosts.iter()),
            )?;

            Ok(count)
        })
    }

    pub fn resolve_resume_session(&self, session_id: &str) -> Result<String> {
        self.with_conn(|conn| {
            let mut current = session_id.to_string();
            let mut seen = vec![current.clone()];
            for _ in 0..100 {
                let has_messages = conn.query_row(
                    "SELECT 1 FROM messages WHERE session_id = ? LIMIT 1",
                    params![current],
                    |row| row.get::<_, i32>(0),
                ).optional()?.map(|v| v == 1).unwrap_or(false);
                if has_messages { return Ok(current); }
                match conn.query_row(
                    "SELECT id FROM sessions WHERE parent_session_id = ? ORDER BY started_at DESC, id DESC LIMIT 1",
                    params![current], |row| row.get(0))
                {
                    Ok(cid) if !seen.contains(&cid) => { seen.push(cid.clone()); current = cid; }
                    _ => return Ok(current),
                }
            }
            Ok(current)
        })
    }

    pub fn delete_session(&self, session_id: &str) -> Result<()> {
        self.with_conn_mut(|conn| {
            conn.execute(
                "DELETE FROM messages WHERE session_id = ?",
                params![session_id],
            )?;
            conn.execute("DELETE FROM sessions WHERE id = ?", params![session_id])?;
            Ok(())
        })
    }

    /// Mark a session as ended with a reason.
    pub fn end_session(&self, session_id: &str, end_reason: &str) -> Result<()> {
        self.with_conn_mut(|conn| {
            conn.execute(
                "UPDATE sessions SET ended_at = ?, end_reason = ? WHERE id = ?",
                params![now_ts(), end_reason, session_id],
            )?;
            Ok(())
        })
    }

    /// Set the parent_session_id for a session.
    pub fn set_parent_session_id(&self, session_id: &str, parent_id: &str) -> Result<()> {
        self.with_conn_mut(|conn| {
            conn.execute(
                "UPDATE sessions SET parent_session_id = ? WHERE id = ?",
                params![parent_id, session_id],
            )?;
            Ok(())
        })
    }

    /// Update the title of a session in the database.
    pub fn set_title(&self, session_id: &str, new_title: &str) -> Result<()> {
        self.with_conn_mut(|conn| {
            let count: i64 = conn.query_row(
                "SELECT COUNT(*) FROM sessions WHERE title = ? AND id != ?",
                params![new_title, session_id],
                |row| row.get(0),
            )?;
            if count > 0 {
                return Err(anyhow::anyhow!(
                    "Session title must be unique, '{new_title}' is already in use"
                ));
            }
            conn.execute(
                "UPDATE sessions SET title = ? WHERE id = ?",
                params![new_title, session_id],
            )?;
            Ok(())
        })
    }

    // ── Goals CRUD — independent entities, not scoped to sessions ────────

    /// Check if a goal exists.
    pub fn goal_exists(&self, goal_id: &str) -> Result<bool> {
        self.with_conn(|conn| {
            let count: i64 = conn.query_row(
                "SELECT COUNT(*) FROM goals WHERE id = ?",
                params![goal_id],
                |row| row.get(0),
            )?;
            Ok(count > 0)
        })
    }

    /// Create a new goal.
    ///
    /// `owner_session` is the session that created the goal (can be None).
    pub fn create_goal(
        &self,
        goal_id: &str,
        text: &str,
        max_turns: usize,
        owner_session: Option<&str>,
    ) -> Result<()> {
        let now = now_ts();
        self.with_conn_mut(|conn| {
            conn.execute(
                "INSERT INTO goals (id, text, status, owner_session, max_turns, turns_used, created_at, updated_at) \
                 VALUES (?, ?, 'pending', ?, ?, 0, ?, ?)",
                params![goal_id, text, owner_session, max_turns, now, now],
            )?;
            Ok(())
        })
    }

    /// Get goal by ID.
    ///
    /// Returns (text, status, max_turns, turns_used, paused_reason).
    pub fn get_goal(
        &self,
        goal_id: &str,
    ) -> Result<Option<(String, String, usize, usize, Option<String>)>> {
        use rusqlite::OptionalExtension;
        self.with_conn(|conn| {
            conn.query_row(
                "SELECT text, status, max_turns, turns_used, paused_reason FROM goals WHERE id = ?",
                params![goal_id],
                |row| {
                    let text: String = row.get(0)?;
                    let status: String = row.get(1)?;
                    let max_turns: usize = row.get(2)?;
                    let turns_used: usize = row.get(3)?;
                    let paused_reason: Option<String> = row.get(4)?;
                    Ok((text, status, max_turns, turns_used, paused_reason))
                },
            )
            .optional()
            .map_err(|e| anyhow::anyhow!(e))
        })
    }

    /// Update goal status and turns.
    pub fn update_goal(
        &self,
        goal_id: &str,
        status: &str,
        turns_used: usize,
        verdict: Option<&str>,
        reason: Option<&str>,
    ) -> Result<()> {
        self.with_conn_mut(|conn| {
            conn.execute(
                "UPDATE goals SET status = ?, turns_used = ?, last_verdict = ?, last_reason = ?, updated_at = ? WHERE id = ?",
                params![status, turns_used, verdict, reason, now_ts(), goal_id],
            )?;
            Ok(())
        })
    }

    /// Pause a goal with a reason.
    pub fn pause_goal(&self, goal_id: &str, reason: &str) -> Result<()> {
        self.with_conn_mut(|conn| {
            conn.execute(
                "UPDATE goals SET status = 'paused', paused_reason = ?, updated_at = ? WHERE id = ?",
                params![reason, now_ts(), goal_id],
            )?;
            Ok(())
        })
    }

    /// Resume a goal.
    pub fn resume_goal(&self, goal_id: &str, reset_budget: bool) -> Result<()> {
        self.with_conn_mut(|conn| {
            if reset_budget {
                conn.execute(
                    "UPDATE goals SET status = 'active', turns_used = 0, paused_reason = NULL, updated_at = ? WHERE id = ?",
                    params![now_ts(), goal_id],
                )?;
            } else {
                conn.execute(
                    "UPDATE goals SET status = 'active', paused_reason = NULL, updated_at = ? WHERE id = ?",
                    params![now_ts(), goal_id],
                )?;
            }
            Ok(())
        })
    }

    /// Fail (mark done) a goal.
    pub fn complete_goal(&self, goal_id: &str) -> Result<()> {
        self.with_conn_mut(|conn| {
            conn.execute(
                "UPDATE goals SET status = 'done', updated_at = ? WHERE id = ?",
                params![now_ts(), goal_id],
            )?;
            Ok(())
        })
    }

    /// Delete a goal. Cascades to goal_nodes.
    pub fn delete_goal(&self, goal_id: &str) -> Result<()> {
        self.with_conn_mut(|conn| {
            conn.execute("DELETE FROM goals WHERE id = ?", params![goal_id])?;
            Ok(())
        })
    }

    /// List goals by status (for a given session or all).
    pub fn list_goals(
        &self,
        status: Option<&str>,
    ) -> Result<Vec<(String, String, Option<String>)>> {
        self.with_conn(|conn| {
            let sql = match status {
                Some(_s) => "SELECT id, text, owner_session FROM goals WHERE status = ? ORDER BY created_at DESC",
                None => "SELECT id, text, owner_session FROM goals ORDER BY created_at DESC",
            };
            let mut stmt = conn.prepare(sql)?;
            let rows: Vec<(String, String, Option<String>)> = match status {
                Some(s) => stmt.query_map(params![s], |row| {
                    Ok((row.get(0)?, row.get(1)?, row.get(2)?))
                })?.filter_map(|r| r.ok()).collect(),
                None => stmt.query_map([], |row| {
                    Ok((row.get(0)?, row.get(1)?, row.get(2)?))
                })?.filter_map(|r| r.ok()).collect(),
            };
            Ok(rows)
        })
    }

    // ── Goal Nodes CRUD (lightweight status tracking) ─────────────────────

    /// Add a plan node to a goal.
    pub fn create_goal_node(&self, goal_id: &str, title: &str) -> Result<i64> {
        let now = now_ts() as f64;
        self.with_conn_mut(|conn| {
            conn.execute(
                "INSERT INTO goal_nodes (goal_id, title, status, created_at) VALUES (?, ?, 'pending', ?)",
                params![goal_id, title, now],
            )?;
            Ok(conn.last_insert_rowid())
        })
    }

    /// Get all node titles for a goal.
    pub fn list_goal_nodes(&self, goal_id: &str) -> Result<Vec<(String, String)>> {
        self.with_conn(|conn| {
            let mut stmt = conn.prepare(
                "SELECT title, status FROM goal_nodes WHERE goal_id = ? ORDER BY row_id",
            )?;
            let rows: Vec<(String, String)> = stmt
                .query_map(params![goal_id], |row| Ok((row.get(0)?, row.get(1)?)))?
                .filter_map(|r| r.ok())
                .collect();
            Ok(rows)
        })
    }

    /// Update a node's status and completion message.
    pub fn update_goal_node(
        &self,
        goal_id: &str,
        title: &str,
        status: &str,
        msg: &str,
    ) -> Result<()> {
        self.with_conn_mut(|conn| {
            conn.execute(
                "UPDATE goal_nodes SET status = ?, completion_message = ? WHERE goal_id = ? AND title = ?",
                params![status, msg, goal_id, title],
            )?;
            Ok(())
        })
    }

    pub fn close(&self) -> Result<()> {
        let mut lock = self.conn.lock().unwrap();
        if let Some(conn) = lock.take() {
            let _ = conn.execute_batch("PRAGMA wal_checkpoint(PASSIVE)");
            drop(conn);
        }
        Ok(())
    }
}

pub struct MessagesAround {
    pub window: Vec<Message>,
    pub messages_before: usize,
    pub messages_after: usize,
}

pub struct AnchoredView {
    pub window: Vec<Message>,
    pub messages_before: usize,
    pub messages_after: usize,
    pub bookend_start: Vec<Message>,
    pub bookend_end: Vec<Message>,
}

pub struct SearchHit {
    pub id: String,
    pub session_id: String,
    pub session_title: Option<String>,
    pub role: String,
    pub content: String,
    pub timestamp: f64,
    pub snippet: String,
}

pub struct DiscoveryResult {
    pub query: String,
    pub results: Vec<DiscoveryEntry>,
    pub count: usize,
}

#[derive(Clone)]
pub struct DiscoveryEntry {
    pub session_id: String,
    pub title: Option<String>,
    pub when: String,
    pub source: SessionSource,
    pub model: Option<String>,
    pub snippet: String,
    pub matched_role: String,
    pub match_message_id: String,
    pub window: Vec<Message>,
    pub bookend_start: Vec<Message>,
    pub bookend_end: Vec<Message>,
    pub messages_before: usize,
    pub messages_after: usize,
}

pub struct BrowseResult {
    pub results: Vec<BrowseEntry>,
    pub count: usize,
}

#[derive(Clone)]
pub struct BrowseEntry {
    pub session_id: String,
    pub title: Option<String>,
    pub source: SessionSource,
    pub started_at: chrono::DateTime<chrono::Utc>,
    pub last_active: chrono::DateTime<chrono::Utc>,
    pub message_count: usize,
    pub preview: Option<String>,
}

pub fn sanitize_fts5_query(query: &str) -> String {
    query
        .replace('"', "\\\"")
        .replace('(', "\\(")
        .replace(')', "\\)")
}

// ── SessionManager (SQLite-backed, wraps SessionDB) ────────────────────────

/// Lifecycle state of the session manager.
#[derive(Clone, Debug, PartialEq)]
pub enum SessionState {
    /// DB is not yet initialized. SessionManager is idle.
    Off,
    /// DB is open, session metadata loaded (titles, IDs, timestamps).
    /// No messages loaded — messages are loaded on-demand.
    Init,
    /// One specific session's messages are loaded and ready for use.
    /// When `Option` is `None`, only session metadata is loaded (no messages).
    Loaded(Option<String>),
}

/// In-memory session cache with SQLite persistence via SessionDB.
pub struct DBSessionManager {
    db: SessionDB,
    sessions: std::collections::HashMap<String, Session>,
    active_session_id: Option<String>,
    state: SessionState,
}

impl DBSessionManager {
    pub fn new() -> Result<Self> {
        Self::new_with_agent(None)
    }

    /// Create a new DBSessionManager with per-agent isolation.
    ///
    /// `agent_name` determines the subdirectory under `~/.obenmatrix/agents/`:
    /// - `None` or empty string → `~/.obenmatrix/agents/default/`
    /// - `Some("manager")` → `~/.obenmatrix/agents/manager/`
    /// - `Some("worker")` → `~/.obenmatrix/agents/worker/`
    pub fn new_with_agent(agent_name: Option<&str>) -> Result<Self> {
        let base = dirs::home_dir()
            .map(|d| d.join(".obenmatrix").join("agents"))
            .unwrap_or_else(|| std::path::PathBuf::from("~/.obenmatrix/agents"));
        
        // ✅ 必须有 agent_name，否则使用 "default"
        let agent_dir = agent_name
            .and_then(|n| if n.is_empty() { None } else { Some(n) })
            .unwrap_or("default");
        
        let path = base.join(agent_dir);
        std::fs::create_dir_all(&path)?;
        let db_path = path.join("sessions.db");
        
        Self::new_with_db_path(db_path)
    }

    pub fn new_with_path(storage_path: std::path::PathBuf) -> Result<Self> {
        let db_path = storage_path.join("sessions.db");
        Self::new_with_db_path(db_path)
    }

    fn new_with_db_path(db_path: std::path::PathBuf) -> Result<Self> {
        let db = SessionDB::new(&db_path)?;
        Ok(Self {
            db,
            sessions: std::collections::HashMap::new(),
            active_session_id: None,
            state: SessionState::Off,
        })
    }

    /// Return current state.
    pub fn state(&self) -> &SessionState {
        &self.state
    }

    /// Expose the underlying `SessionDB` for direct access.
    ///
    /// Used externally by modules like `oben-goals` that need
    /// to persist data alongside session storage.
    pub fn db(&self) -> &SessionDB {
        &self.db
    }

    /// Set the parent_session_id for a session (DB + in-memory).
    pub fn set_parent_session_id_for_child(
        &mut self,
        child_session_id: &str,
        parent_id: &str,
    ) -> Result<()> {
        self.db.set_parent_session_id(child_session_id, parent_id)
    }

    /// Close the session manager — drops the DB connection and resets to Off.
    pub fn close(&mut self) -> Result<()> {
        self.sessions.clear();
        self.active_session_id = None;
        self.state = SessionState::Off;
        Ok(())
    }

    /// Read-only view of a session's messages — does not mutate state.
    pub fn get_session_messages(&self, session_id: &str) -> Result<Vec<Message>> {
        self.db.load_messages(session_id)
    }

    /// Load a session's messages into the in-memory cache.
    /// Does NOT change `active_session_id` and does NOT change session state.
    pub fn ensure_session_loaded(&mut self, session_id: &str) -> Result<()> {
        // Ensure initialized
        if self.state == SessionState::Off {
            self.init()?;
        }
        self.load_session_into_cache(session_id)?;
        Ok(())
    }

    fn find_session_key_by_name(&self, name: &str) -> Option<String> {
        self.sessions
            .iter()
            .find(|(_, s)| s.name == name)
            .map(|(k, _)| k.clone())
    }

    fn load_session_into_cache(&mut self, session_id: &str) -> Result<()> {
        // Get metadata from DB
        let meta = self
            .db
            .get_session(session_id)?
            .ok_or_else(|| anyhow::anyhow!("Session not found: {}", session_id))?;
        // Load messages from DB
        let messages = self.db.load_messages(session_id)?;
        let msg_count = messages.len();
        let mut session = meta;
        session.messages = messages;
        session.persisted_message_count = msg_count;
        self.sessions.insert(session_id.to_string(), session);
        if self.active_session_id.is_none() {
            self.active_session_id = Some(session_id.to_string());
        }
        Ok(())
    }

    /// Load session metadata from DB without messages.
    ///
    /// **State transition: Off → Init**.
    ///
    /// Populates the in-memory cache with session titles, IDs, and timestamps
    /// — sufficient for listing and searching sessions without paying the cost
    /// of loading all messages.
    pub fn init(&mut self) -> Result<()> {
        // Idempotent — already initialized
        if self.state != SessionState::Off {
            return Ok(());
        }
        let metas = self.db.list_sessions(None, &[], 1000, 0, false)?;
        for meta in metas {
            let s = Session {
                id: meta.id.clone(),
                name: meta.name.clone(),
                created_at: meta.started_at,
                updated_at: meta.started_at,
                messages: Vec::new(), // metadata only — no messages loaded
                memory_context: None,
                summary_chunks: Vec::new(),
                persisted_message_count: meta.message_count,
                compaction_summary: None,
                metadata: meta,
            };
            self.sessions.insert(s.id.clone(), s);
        }
        self.state = SessionState::Init;
        Ok(())
    }

    /// Get or create a session by name, loading its messages.
    ///
    /// **State transition: Init → Loaded**.
    /// Returns `&mut Session` for in-place modification. See also
    /// `get_or_create_session` which returns the session ID.
    pub fn get_or_create_session(&mut self, name: &str) -> &mut Session {
        // Ensure we're initialized
        if self.state == SessionState::Off {
            self.init().ok(); // best effort, continue anyway
        }
        let key = self.find_session_key_by_name(name);
        match key {
            Some(key) => {
                // Session exists in cache but might not have messages loaded
                // Check if we need to load messages
                let session = self.sessions.get(&key).unwrap();
                if session.messages.is_empty() {
                    self.load_session_into_cache(&key).ok();
                }
                self.state = SessionState::Loaded(Some(key.clone()));
                self.sessions.get_mut(&key).unwrap()
            }
            None => {
                let session = self.db.get_or_create_session(name).unwrap();
                let id = session.id.clone();
                let messages = self.db.load_messages(&id).unwrap_or_default();
                let msg_count = messages.len();
                let mut full_session = session;
                full_session.messages = messages;
                full_session.persisted_message_count = msg_count;
                self.sessions.insert(id.clone(), full_session);
                self.active_session_id = Some(id.clone());
                self.state = SessionState::Loaded(Some(id.clone()));
                self.sessions.get_mut(&id).unwrap()
            }
        }
    }

    /// Create a new session (empty messages).
    ///
    /// **State transition: Off → Init → Loaded**.
    /// Returns `&mut Session` for in-place modification. See also
    /// `create_session` which returns the session ID.
    pub fn create_session(&mut self, name: &str) -> &mut Session {
        // Ensure we're initialized
        if self.state == SessionState::Off {
            self.init().ok(); // best effort, continue anyway
        }
        let session = self.db.get_or_create_session(name).unwrap();
        let id = session.id.clone();
        self.sessions.insert(id.clone(), session);
        self.active_session_id = Some(id);
        self.state = SessionState::Loaded(self.active_session_id.clone());
        self.sessions
            .get_mut(self.active_session_id.as_ref().unwrap())
            .unwrap()
    }

    /// Alias for `create_session`.
    pub fn new_session(&mut self, name: &str) -> &mut Session {
        self.create_session(name)
    }

    /// Switch to a session, loading its messages.
    ///
    /// **State transition: Init → Loaded**.
    /// Returns `&mut Session` for in-place modification. See also
    /// `switch_session` which returns the session ID.
    pub fn switch_session(&mut self, session_id: &str) -> Result<&mut Session> {
        // Ensure we're initialized
        if self.state == SessionState::Off {
            self.init()?;
        }
        let current_active = self.active_session_id.clone();
        if let Some(ref active_id) = current_active {
            if active_id != session_id {
                self.incremental_save(Some(active_id))?;
            }
        }
        self.load(Some(session_id))?;
        self.active_session_id = Some(session_id.to_string());
        self.sessions
            .get_mut(session_id)
            .ok_or_else(|| anyhow::anyhow!("Session not found: {}", session_id))
    }

    pub fn switch(&mut self, session_id: &str) -> Result<SwitchResult> {
        let session_id = self
            .find_key(session_id)
            .ok_or_else(|| anyhow::anyhow!("Session not found: {}", session_id))?;
        let session = self.switch_session(&session_id)?;
        Ok(SwitchResult {
            session_id: session_id.clone(),
            name: session.name.clone(),
            message_count: session.message_count(),
            was_new: false,
        })
    }

    pub fn switch_or_create(&mut self, key: &str) -> Result<SwitchResult> {
        if let Ok(sr) = self.switch(key) {
            return Ok(sr);
        }
        let session = self.create_session(key);
        Ok(SwitchResult {
            session_id: session.id.clone(),
            name: session.name.clone(),
            message_count: session.message_count(),
            was_new: true,
        })
    }

    pub fn active_session(&self) -> Option<&Session> {
        self.active_session_id
            .as_ref()
            .and_then(|id| self.sessions.get(id))
    }

    pub fn active_session_mut(&mut self) -> Option<&mut Session> {
        self.active_session_id
            .as_ref()
            .and_then(|id| self.sessions.get_mut(id))
    }

    /// Return all sessions as a `Vec<&Session>`.
    pub fn list_sessions_ref(&self) -> Vec<&Session> {
        self.sessions.values().collect()
    }

    /// Alias for `list_sessions_ref`.
    pub fn list(&self) -> Vec<&Session> {
        self.list_sessions_ref()
    }

    pub fn incremental_save(&mut self, session_id: Option<&str>) -> Result<()> {
        let sid: String = match session_id {
            Some(id) => id.to_string(),
            None => match &self.active_session_id {
                Some(id) => id.clone(),
                None => {
                    info!("No active session to save");
                    return Ok(());
                }
            },
        };
        let session = self
            .sessions
            .get(&sid)
            .ok_or_else(|| anyhow::anyhow!("Session not found: {}", sid))?;
        self.db.ensure_session_in_db(&sid, &session.metadata)?;
        // Clamp: after compression, messages.len() can be < persisted_message_count.
        // Treat this as "nothing new to persist" — the DB state is already ahead of memory.
        let new_count = session
            .messages
            .len()
            .saturating_sub(session.persisted_message_count);

        // No new messages to persist
        if new_count == 0 {
            return Ok(());
        }

        let start = session.messages.len() - new_count;
        let mut new_messages: Vec<Message> = session.messages[start..].iter().cloned().collect();
        let _ = session;

        self.db.save_new_messages(&sid, &mut new_messages)?;
        if let Some(s) = self.sessions.get_mut(&sid) {
            s.persisted_message_count = s.messages.len();
        }
        Ok(())
    }

    /// Save compacted messages: delete all old DB messages, then insert the
    /// compacted set.  The `incremental_save()` path cannot handle compaction
    /// because `persisted_message_count > messages.len()` after compression.
    pub fn save_compacted(&mut self, session_id: &str, messages: &[Message]) -> Result<()> {
        // 1. Delete all old messages from DB
        self.db.clear_messages(session_id)?;
        // 2. Insert compacted messages
        let mut msg_copy: Vec<Message> = messages.iter().cloned().collect();
        self.db.save_new_messages(session_id, &mut msg_copy)?;
        // 3. Update in-memory session
        if let Some(s) = self.sessions.get_mut(session_id) {
            s.messages = msg_copy;
            s.persisted_message_count = s.messages.len();
        }
        // 4. Update session metadata (message_count, ended_at)
        self.db.with_conn_mut(|conn| {
            conn.execute(
                "UPDATE sessions SET message_count = ? WHERE id = ?",
                params![messages.len() as i64, session_id],
            )?;
            Ok(())
        })
    }

    /// Load session messages into the cache.
    ///
    /// **State transition: Init → Loaded**.
    ///
    /// If `session_id` is `Some`, only that session's messages are loaded.
    /// If `None`, the active session's messages are loaded.
    pub fn load(&mut self, session_id: Option<&str>) -> Result<()> {
        match self.state {
            SessionState::Off => {
                self.init()?;
            }
            SessionState::Init => {}
            SessionState::Loaded(_) => return Ok(()), // already loaded
        }
        // If a specific session_id is given, load just that one (messages + metadata)
        if let Some(sid) = session_id {
            self.load_session_into_cache(sid)?;
            self.active_session_id = Some(sid.to_string());
            self.state = SessionState::Loaded(Some(sid.to_string()));
            return Ok(());
        }
        // load(None): only load session metadata (not messages) into cache.
        // Messages are loaded on-demand via load_session_into_cache when
        // the user switches to or selects a specific session.
        let metas = self.db.list_sessions(None, &[], 1000, 0, false)?;
        for meta in metas {
            let session = Session {
                id: meta.id.clone(),
                name: meta.name.clone(),
                created_at: meta.started_at,
                updated_at: meta.started_at,
                messages: Vec::new(),
                memory_context: None,
                summary_chunks: Vec::new(),
                persisted_message_count: meta.message_count,
                compaction_summary: None,
                metadata: meta,
            };
            self.sessions.insert(session.id.clone(), session);
        }

        // Don't change active_session_id — preserve the current one.
        self.state = SessionState::Loaded(self.active_session_id.clone());
        info!(
            "load(None) complete: sessions={}, active={} [state={}]",
            self.sessions.len(),
            self.active_session_id.as_deref().unwrap_or("(none)"),
            match &self.state {
                SessionState::Loaded(Some(id)) => format!("Loaded(Some({id}))"),
                SessionState::Loaded(None) => "Loaded(None)".to_string(),
                SessionState::Off => "Off".to_string(),
                SessionState::Init => "Init".to_string(),
            }
        );
        Ok(())
    }

    pub fn find_key(&self, key: &str) -> Option<String> {
        if self.sessions.contains_key(key) {
            return Some(key.to_string());
        }
        self.find_session_key_by_name(key)
    }

    /// Resolve a name key to a session ID.
    pub fn resolve_session_id(&self, key: &str) -> Option<String> {
        self.find_key(key)
    }

    pub fn delete_session(&mut self, key: &str) -> Result<()> {
        // Resolve name → UUID (like switch() does)
        let resolved = self
            .find_key(key)
            .ok_or_else(|| anyhow::anyhow!("Session not found: {}", key))?;

        self.db.delete_session(&resolved)?;
        self.sessions.remove(&resolved);
        if self.active_session_id.as_deref() == Some(&resolved) {
            self.active_session_id = None;
        }
        Ok(())
    }

    /// Alias for `delete_session`.
    pub fn remove_session(&mut self, key: &str) -> Result<()> {
        self.delete_session(key)
    }

    /// Alias for `delete_session`.
    pub fn delete(&mut self, key: &str) -> Result<()> {
        self.delete_session(key)
    }

    pub fn session_count(&self) -> usize {
        self.sessions.len()
    }

    #[inline]
    pub fn active(&self) -> Option<&Session> {
        self.active_session()
    }

    #[inline]
    pub fn session_mut(&mut self, session_id: &str) -> Option<&mut Session> {
        self.sessions.get_mut(session_id)
    }

    #[inline]
    pub fn session(&self, session_id: &str) -> Option<&Session> {
        self.sessions.get(session_id)
    }

    /// Save pending messages. Accepts `Option<&str>` to match `SessionManagerExt`.
    pub fn save_session(&mut self, session_id: Option<&str>) -> Result<()> {
        self.incremental_save(session_id)
    }

    /// Update the title of the current session in both the in-memory cache and the database.
    pub fn set_title(&mut self, new_title: &str) -> Result<()> {
        let sid = self
            .active_session_id
            .clone()
            .ok_or_else(|| anyhow::anyhow!("No active session to rename"))?;

        // Update in-memory session
        if let Some(session) = self.sessions.get_mut(&sid) {
            session.metadata.title = Some(new_title.to_string());
        }

        // Persist to database
        self.db.set_title(&sid, new_title)?;

        Ok(())
    }

    /// Split a session after compression: end the parent, create a child.
    ///
    /// After context compression, the old session is marked as ended with
    /// `end_reason = "compression"` and a new child session is created with
    /// `parent_session_id` pointing to the old session.
    ///
    /// The child's title is auto-numbered: "chat-123" → "chat-123 (2)".
    pub fn split_after_compression(&mut self, parent_id: &str) -> Result<Session> {
        // 1. Mark parent as ended with compression reason
        self.db.end_session(parent_id, "compression")?;

        // 2. Update in-memory cache to match the DB
        if let Some(parent) = self.sessions.get_mut(parent_id) {
            parent.metadata.end_reason = Some("compression".to_string());
            parent.metadata.ended_at = Some(chrono::Utc::now());
        }

        // 3. Save parent's messages to DB before creating child
        {
            let parent = self
                .sessions
                .get_mut(parent_id)
                .ok_or_else(|| anyhow::anyhow!("Session not found: {}", parent_id))?;
            let mut parent_messages = parent.messages.clone();
            self.db.save_messages(parent_id, &mut parent_messages)?;
        } // parent dropped here

        // 4. Determine child title: use parent's **name** (stable) as base,
        //    scanning ALL sessions for the highest child number.
        //    This ensures "test-chat (2)", "test-chat (3)", etc.
        let parent_session = self.sessions.get(parent_id);
        let child_name = parent_session.map(|p| p.name.as_str()).unwrap_or("unnamed");
        let child_title = self.next_child_title(child_name, parent_id);
        // Keep the child's `name` stable (same as parent).
        let child_name_str = child_name.to_string();

        // 4. Create child session record in DB
        let child_session = self.db.get_or_create_session(&child_title)?;
        let child_id = child_session.id.clone();

        // 5. Set parent_session_id in DB
        self.db.set_parent_session_id(&child_id, parent_id)?;

        // 6. Load child messages and set lineage in in-memory cache
        let child_messages = self.db.load_messages(&child_id)?;
        let mut full_child = child_session;
        full_child.messages = child_messages;
        full_child.metadata.parent_session_id = Some(parent_id.to_string());
        // Keep name stable (use parent's name), only title is auto-numbered.
        full_child.name = child_name_str;
        full_child.metadata.title = Some(child_title.clone());
        self.sessions.insert(child_id.clone(), full_child.clone());

        // 7. Set child as active session
        self.active_session_id = Some(child_id.clone());

        Ok(full_child)
    }

    /// Compute the next auto-numbered title for a child session.
    ///
    /// Finds the highest existing child number and increments it.
    /// Examples: "chat" → "chat (2)", "chat (5)" → "chat (6)".
    fn next_child_title(&self, base: &str, parent_id: &str) -> String {
        let next_num = self.find_next_child_number(base, parent_id);
        format!("{} ({})", base, next_num)
    }

    /// Find the next child number for a given base title.
    ///
    /// Scans **all** sessions (not just children of `parent_id`) to find the
    /// highest existing child number, ensuring unique auto-numbers across
    /// all rotations regardless of lineage depth.
    fn find_next_child_number(&self, base: &str, _parent_id: &str) -> usize {
        // Scan all sessions for any child with the given base prefix
        let mut max_num = 1;
        for s in self.sessions.values() {
            if s.metadata.parent_session_id.is_some() {
                if let Some(title) = &s.metadata.title {
                    // Parse "base (N)" suffix
                    if let Some(suffix) = title.strip_prefix(base) {
                        if let Some(num) =
                            suffix.strip_prefix(" (").and_then(|s| s.strip_suffix(')'))
                        {
                            if let Ok(n) = num.parse::<usize>() {
                                if n > max_num {
                                    max_num = n;
                                }
                            }
                        }
                    }
                }
            }
        }
        max_num + 1
    }

    /// Clone a session: create a new session with the same messages.
    pub fn clone_session(&mut self, name: &str) -> Option<Session> {
        let target = self.find_key(name)?;
        let src = self.sessions.get(&target)?.clone();
        let mut new_session = Session::new(name);
        new_session.messages = src.messages.clone();
        new_session.summary_chunks = src.summary_chunks.clone();
        let new_id = new_session.id.clone();
        self.sessions.insert(new_id.clone(), new_session.clone());
        self.active_session_id = Some(new_id.clone());
        Some(self.sessions.get(&new_id).unwrap().clone())
    }

    // ── SessionManagerExt methods ───────────────────────────────────────

    /// Get or create a session by name, returning the session ID.
    pub fn get_or_create_session_id(&mut self, name: &str) -> String {
        let session = self.get_or_create_session(name);
        session.id.clone()
    }

    /// Create a new empty session, returning the session ID.
    pub fn create_session_id(&mut self, name: &str) -> String {
        let session = self.create_session(name);
        session.id.clone()
    }

    /// Reset the current session — clear messages, keep metadata.
    pub fn reset_current_session(&mut self) -> Result<()> {
        let sid = self
            .active_session_id
            .clone()
            .ok_or_else(|| anyhow::anyhow!("No active session to reset"))?;
        self.reset_session(&sid)
    }

    /// Reset a specific session — clear messages, keep metadata.
    pub fn reset_session(&mut self, key: &str) -> Result<()> {
        let sid = self
            .find_key(key)
            .ok_or_else(|| anyhow::anyhow!("Session not found: {}", key))?;
        let _now_ts = now_ts();
        // End the current session in DB
        self.db.end_session(&sid, "reset")?;
        // Create new session with same metadata
        let old_meta = self.sessions.get(&sid).map(|s| s.metadata.clone());
        let new_id = format!("reset_{}", uuid::Uuid::new_v4().to_string());
        let meta = old_meta.unwrap_or_default();
        let new_meta = SessionMetadata {
            id: new_id.clone(),
            name: meta.name.clone(),
            source: meta.source.clone(),
            model: meta.model.clone(),
            system_prompt: meta.system_prompt.clone(),
            parent_session_id: Some(sid.clone()),
            started_at: chrono::Utc::now(),
            ended_at: None,
            end_reason: Some("reset".to_string()),
            title: meta.title.clone(),
            preview: meta.preview.clone(),
            message_count: 0,
            tool_call_count: 0,
            input_tokens: 0,
            output_tokens: 0,
            total_tokens: 0,
            estimated_cost_usd: 0.0,
            cost_status: "unknown".to_string(),
            handoff_state: None,
            handoff_platform: None,
            handoff_error: None,
            origin: meta.origin.clone(),
            last_prompt_tokens: 0,
            is_fresh_reset: true,
            suspended: false,
            resume_pending: false,
            resume_reason: None,
            last_resume_marked_at: None,
            compaction_summary: None,
            turn_count: 0,
        };
        // Update DB
        self.db.set_parent_session_id(&new_id, &sid)?;
        // Update in-memory cache
        if let Some(s) = self.sessions.get_mut(&sid) {
            s.metadata = meta.clone();
            s.messages.clear();
            s.persisted_message_count = 0;
            s.updated_at = chrono::Utc::now();
        }
        // Insert new session
        let new_session = Session {
            id: new_id.clone(),
            name: meta.name.clone(),
            created_at: meta.started_at,
            updated_at: chrono::Utc::now(),
            messages: Vec::new(),
            memory_context: None,
            summary_chunks: Vec::new(),
            persisted_message_count: 0,
            compaction_summary: None,
            metadata: new_meta,
        };
        self.sessions.insert(new_id.clone(), new_session);
        self.active_session_id = Some(new_id.clone());
        Ok(())
    }

    /// Suspend a session — next access will auto-create a fresh session.
    pub fn suspend_session(&mut self, key: &str) -> bool {
        let sid = match self.find_key(key) {
            Some(id) => id,
            None => return false,
        };
        if let Some(s) = self.sessions.get_mut(&sid) {
            s.metadata.suspended = true;
            self.db.end_session(&sid, "suspended").ok();
            true
        } else {
            false
        }
    }

    /// Mark a session as resumable after a restart interruption.
    pub fn mark_resume_pending(&mut self, key: &str, reason: &str) -> bool {
        let sid = match self.find_key(key) {
            Some(id) => id,
            None => return false,
        };
        if let Some(s) = self.sessions.get_mut(&sid) {
            if !s.metadata.suspended {
                s.metadata.resume_pending = true;
                s.metadata.resume_reason = Some(reason.to_string());
                s.metadata.last_resume_marked_at = Some(chrono::Utc::now());
                true
            } else {
                false
            }
        } else {
            false
        }
    }

    /// Clear the resume-pending flag after a successful resumed turn.
    pub fn clear_resume_pending(&mut self, key: &str) -> bool {
        let sid = match self.find_key(key) {
            Some(id) => id,
            None => return false,
        };
        if let Some(s) = self.sessions.get_mut(&sid) {
            if s.metadata.resume_pending {
                s.metadata.resume_pending = false;
                s.metadata.resume_reason = None;
                s.metadata.last_resume_marked_at = None;
                true
            } else {
                false
            }
        } else {
            false
        }
    }

    /// List all sessions as `SessionListEntry` with metadata.
    pub fn list_sessions(&self, active_minutes: Option<u64>) -> Vec<oben_models::SessionListEntry> {
        let now = chrono::Utc::now();
        let cutoff = active_minutes.map(|m| now - chrono::Duration::minutes(m as i64));
        self.sessions
            .values()
            .filter(|s| cutoff.map(|c| s.updated_at >= c).unwrap_or(true))
            .map(|s| oben_models::SessionListEntry {
                id: s.id.clone(),
                name: s.name.clone(),
                title: s.metadata.title.clone(),
                source: s.metadata.source.clone(),
                model: s.metadata.model.clone(),
                message_count: s.metadata.message_count,
                tool_call_count: s.metadata.tool_call_count,
                input_tokens: s.metadata.input_tokens,
                output_tokens: s.metadata.output_tokens,
                started_at: s.metadata.started_at,
                last_active: s.updated_at,
                ended_at: s.metadata.ended_at,
                end_reason: s.metadata.end_reason.clone(),
                preview: s.metadata.preview.clone(),
                parent_session_id: s.metadata.parent_session_id.clone(),
                suspended: s.metadata.suspended,
            })
            .collect::<Vec<_>>()
    }

    /// List all sessions as full `Session` objects.
    /// Alias for iterating `sessions.values().cloned()`.
    pub fn list_sessions_full(&self) -> Vec<oben_models::Session> {
        self.sessions
            .values()
            .map(|s| {
                let mut cloned = s.clone();
                cloned.persisted_message_count = s.metadata.message_count;
                cloned
            })
            .collect()
    }

    /// Prune sessions older than `max_age_days` (exclude suspended and active).
    pub fn prune_sessions(&mut self, max_age_days: i64) -> usize {
        if max_age_days <= 0 {
            return 0;
        }
        let cutoff = chrono::Utc::now() - chrono::Duration::days(max_age_days);
        let active_sid = self.active_session_id.clone();
        let to_remove: Vec<String> = self
            .sessions
            .iter()
            .filter(|(_, s)| {
                !s.metadata.suspended
                    && s.updated_at < cutoff
                    && s.id != active_sid.as_deref().unwrap_or("")
            })
            .map(|(id, _)| id.clone())
            .collect();
        for sid in &to_remove {
            self.db.delete_session(sid).ok();
            self.sessions.remove(sid);
        }
        if to_remove.contains(&active_sid.clone().unwrap_or_default()) {
            self.active_session_id = None;
        }
        to_remove.len()
    }

    /// Get the active session ID.
    pub fn active_session_id(&self) -> Option<String> {
        self.active_session_id.clone()
    }

    /// Update token tracking for a session.
    pub fn update_token_tracking(
        &mut self,
        session_id: &str,
        input_tokens: usize,
        output_tokens: usize,
        total_tokens: usize,
        estimated_cost_usd: f64,
    ) {
        if let Some(s) = self.sessions.get_mut(session_id) {
            s.metadata.input_tokens += input_tokens;
            s.metadata.output_tokens += output_tokens;
            s.metadata.total_tokens += total_tokens;
            s.metadata.estimated_cost_usd += estimated_cost_usd;
            s.metadata.cost_status = "tracked".to_string();
            s.metadata.last_prompt_tokens = input_tokens;
        }
    }
}

// ── SessionManager impl for DBSessionManager ───────────────────────────
// Uses `Self::method(self)` to call inherent methods without
// ambiguity — avoids infinite recursion vs `self.method()`.

impl SessionManager for DBSessionManager {
    fn init(&mut self) -> Result<()> {
        Self::init(self)
    }
    fn get_or_create_session(&mut self, name: &str) -> &mut Session {
        Self::get_or_create_session(self, name)
    }
    fn create_session(&mut self, name: &str) -> &mut Session {
        Self::create_session(self, name)
    }
    fn switch_session(&mut self, key: &str) -> Result<&mut Session, anyhow::Error> {
        Self::switch_session(self, key)
    }
    fn reset_current_session(&mut self) -> Result<()> {
        Self::reset_current_session(self)
    }
    fn reset_session(&mut self, key: &str) -> Result<()> {
        Self::reset_session(self, key)
    }
    fn suspend_session(&mut self, key: &str) -> bool {
        Self::suspend_session(self, key)
    }
    fn mark_resume_pending(&mut self, key: &str, reason: &str) -> bool {
        Self::mark_resume_pending(self, key, reason)
    }
    fn clear_resume_pending(&mut self, key: &str) -> bool {
        Self::clear_resume_pending(self, key)
    }
    fn list_sessions(&self, active_minutes: Option<u64>) -> Vec<oben_models::SessionListEntry> {
        Self::list_sessions(self, active_minutes)
    }
    fn delete_session(&mut self, key: &str) -> Result<()> {
        Self::remove_session(self, key)
    }
    fn prune_sessions(&mut self, max_age_days: i64) -> usize {
        Self::prune_sessions(self, max_age_days)
    }
    fn save_session(&mut self, session_id: Option<&str>) -> Result<()> {
        Self::incremental_save(self, session_id)
    }
    fn resolve_session_id(&self, key: &str) -> Option<String> {
        Self::resolve_session_id(self, key)
    }
    fn update_token_tracking(
        &mut self,
        session_id: &str,
        input_tokens: usize,
        output_tokens: usize,
        total_tokens: usize,
        estimated_cost_usd: f64,
    ) {
        Self::update_token_tracking(
            self,
            session_id,
            input_tokens,
            output_tokens,
            total_tokens,
            estimated_cost_usd,
        )
    }
    fn split_after_compression(&mut self, parent_id: &str) -> Result<Session, anyhow::Error> {
        Self::split_after_compression(self, parent_id)
    }
    fn session_mut(&mut self, session_id: &str) -> Option<&mut Session> {
        Self::session_mut(self, session_id)
    }
    fn session(&self, session_id: &str) -> Option<&Session> {
        Self::session(self, session_id)
    }
    fn save_compacted(&mut self, session_id: &str, messages: &[Message]) -> Result<(), anyhow::Error> {
        Self::save_compacted(self, session_id, messages)
    }
    fn incremental_save(&mut self, session_id: Option<&str>) -> Result<(), anyhow::Error> {
        Self::incremental_save(self, session_id)
    }
    fn new_session(&mut self, name: &str) -> Result<&mut Session, anyhow::Error> {
        Ok(Self::new_session(self, name))
    }
    fn find_key(&self, key: &str) -> Option<String> {
        Self::find_key(self, key)
    }
    fn list_sessions_full(&self) -> Vec<Session> {
        Self::list_sessions_full(self)
    }
    fn get_session_messages(&self, session_id: &str) -> Result<Vec<crate::Message>, anyhow::Error> {
        Self::get_session_messages(self, session_id)
    }
    fn ensure_session_loaded(&mut self, session_id: &str) -> Result<(), anyhow::Error> {
        Self::ensure_session_loaded(self, session_id)
    }
    fn close(&mut self) -> Result<(), anyhow::Error> {
        Self::close(self)
    }
    fn set_compaction_summary(&mut self, session_id: &str, summary: String) -> Result<(), anyhow::Error> {
        if let Some(s) = self.sessions.get_mut(session_id) {
            s.compaction_summary = Some(summary.clone());
        }
        self.db.with_conn_mut(|conn| {
            conn.execute(
                "UPDATE sessions SET compaction_summary = ? WHERE id = ?",
                params![summary, session_id],
            )?;
            Ok(())
        })?;
        Ok(())
    }
    fn get_compaction_summary(&self, session_id: &str) -> Option<String> {
        self.sessions.get(session_id).and_then(|s| s.compaction_summary.clone())
    }
}

// ── Tests ─────────────────────────────────────────────────────────────


pub struct SwitchResult {
    pub session_id: String,
    pub name: String,
    pub message_count: usize,
    pub was_new: bool,
}

// ── Subagent session delegation ───────────────────────────────────────────

/// Metadata returned when spawning a child session for subagent delegation.
///
/// The parent's `active_session_id` is **not** changed. The child session
/// lives in the shared database and must be managed externally by the
/// delegate tool, which creates its own `SessionManager` (or uses `SessionDB`
/// directly) and passes the child session ID to the child agent.
pub struct SpawnedSession {
    /// The newly created child session ID.
    pub session_id: String,
    /// The parent session ID that the child is linked to.
    pub parent_session_id: String,
}

impl DBSessionManager {
    /// Create a new session for a subagent delegate run.
    ///
    /// Sets `parent_session_id` on the session in the database so lineage
    /// is recorded. The parent's `active_session_id` is **not** changed.
    ///
    /// Returns the child session ID which the delegate tool can use to
    /// pass to a child `SessionManager` or `SessionDB` directly.
    ///
    /// Returns an error if there is no active session.
    pub fn spawn_session_for_subagent(
        &mut self,
        name: impl Into<String>,
    ) -> Result<SpawnedSession> {
        let parent_id = self.active_session_id.clone().ok_or_else(|| {
            anyhow::anyhow!(
                "No active session to spawn child from \
                 — call init() or create_session() first"
            )
        })?;

        // create_session sets active_session_id to the new session.
        // We restore the parent after to keep the parent active.
        let child = self.create_session(&name.into());
        let child_id = child.id.clone();

        // Restore parent as active session.
        self.active_session_id = Some(parent_id.clone());

        // Set parent_session_id
        self.db.set_parent_session_id(&child_id, &parent_id)?;
        if let Some(s) = self.sessions.get_mut(&child_id) {
            s.metadata.parent_session_id = Some(parent_id.clone());
        }

        Ok(SpawnedSession {
            session_id: child_id,
            parent_session_id: parent_id,
        })
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use oben_models::Message;

    fn make_test_dir() -> std::path::PathBuf {
        tempfile::tempdir().unwrap().path().join("sessions")
    }

    #[test]
    fn test_manager_creates_session() {
        let path = make_test_dir();
        let mut mgr = DBSessionManager::new_with_path(path).unwrap();
        let session = mgr.new_session("test-session");
        assert_eq!(session.name, "test-session");
        assert!(!session.id.is_empty());
        assert_eq!(mgr.session_count(), 1);
    }

    #[test]
    fn test_manager_list_sessions() {
        let path = make_test_dir();
        let mut mgr = DBSessionManager::new_with_path(path).unwrap();
        mgr.new_session("s1");
        mgr.new_session("s2");
        assert_eq!(mgr.list_sessions_ref().len(), 2);
    }

    #[test]
    fn test_manager_get_or_create_reuses_existing() {
        let path = make_test_dir();
        let mut mgr = DBSessionManager::new_with_path(path).unwrap();
        let s1 = mgr.get_or_create_session("my-session");
        s1.add_message(Message::user("first"));
        let s2 = mgr.get_or_create_session("my-session");
        assert_eq!(s2.name, "my-session");
        assert_eq!(s2.message_count(), 1);
    }

    #[test]
    fn test_save_and_load_roundtrip() {
        let path = make_test_dir();
        let mut mgr = DBSessionManager::new_with_path(path.clone()).unwrap();
        let session = mgr.new_session("persist-test");
        let sid = session.id.clone();
        session.add_message(Message::user("hello"));
        session.add_message(Message::assistant("hi there"));
        let count = mgr.session_count();
        mgr.incremental_save(None).unwrap();
        // Test metadata-only load (no messages)
        let mut mgr3 = DBSessionManager::new_with_path(path.clone()).unwrap();
        mgr3.load(Some(&sid)).unwrap();
        let meta_loaded = mgr3.active_session().unwrap();
        assert_eq!(meta_loaded.metadata.message_count, 2);
        // Test full roundtrip with messages
        let mut mgr2 = DBSessionManager::new_with_path(path.clone()).unwrap();
        mgr2.load(Some(&sid)).unwrap();
        assert_eq!(mgr2.session_count(), count);
        let loaded = mgr2.active_session().unwrap();
        assert_eq!(loaded.name, "persist-test");
        assert_eq!(loaded.message_count(), 2);
    }

    #[test]
    fn test_switch_session() {
        let path = make_test_dir();
        let mut mgr = DBSessionManager::new_with_path(path).unwrap();
        let s1 = mgr.new_session("s1");
        s1.add_message(Message::user("msg in s1"));
        let s1_id = s1.id.clone();
        mgr.incremental_save(None).unwrap(); // persist s1 to DB
        let s2 = mgr.new_session("s2");
        s2.add_message(Message::user("msg in s2"));
        assert!(mgr.active_session().unwrap().name == "s2");
        let switched = mgr.switch_session(&s1_id).unwrap();
        assert_eq!(switched.name, "s1");
        assert_eq!(switched.message_count(), 1);
    }

    #[test]
    fn test_switch_to_nonexistent_session_fails() {
        let path = make_test_dir();
        let mut mgr = DBSessionManager::new_with_path(path).unwrap();
        let err = mgr.switch_session("nonexistent-id").unwrap_err();
        assert!(err.to_string().contains("not found"));
    }

    #[test]
    fn test_load_empty_directory() {
        let path = make_test_dir();
        let mut mgr = DBSessionManager::new_with_path(path.clone()).unwrap();
        mgr.incremental_save(None).unwrap();
        let mut mgr2 = DBSessionManager::new_with_path(path.clone()).unwrap();
        mgr2.load(None).unwrap();
        assert_eq!(mgr2.session_count(), 0);
    }

    #[test]
    fn test_save_load_messages_persisted() {
        // SQLite persists all messages to DB — verified by round-trip load
        let path = make_test_dir();
        let mut mgr = DBSessionManager::new_with_path(path.clone()).unwrap();
        let session = mgr.new_session("persist-test");
        let sid = session.id.clone();
        session.add_message(Message::user("msg1"));
        session.add_message(Message::assistant("msg2"));
        session.add_message(Message::user("msg3"));
        session.add_message(Message::assistant("msg4"));
        mgr.incremental_save(None).unwrap();
        let mut mgr2 = DBSessionManager::new_with_path(path.clone()).unwrap();
        // load(None) loads metadata only (no messages), use load(Some) for message roundtrip
        mgr2.load(Some(&sid)).unwrap();
        let loaded = mgr2.active_session().unwrap();
        assert_eq!(loaded.message_count(), 4);
    }

    #[test]
    fn test_manager_delete() {
        let mut mgr = DBSessionManager::new_with_path(make_test_dir()).unwrap();
        mgr.new_session("to-delete");
        assert_eq!(mgr.list().len(), 1);
        let id = mgr.list()[0].id.clone();
        mgr.remove_session(&id).unwrap();
        assert_eq!(mgr.list().len(), 0);
    }

    #[test]
    fn test_manager_delete_by_name() {
        // Regression: deletion was broken when called with session name
        // (not UUID), because remove_session passed the name directly
        // to db.delete_session() and HashMap::remove() which both
        // expect the UUID primary key.
        let mut mgr = DBSessionManager::new_with_path(make_test_dir()).unwrap();
        mgr.new_session("delete-by-name");
        assert_eq!(mgr.list().len(), 1);
        mgr.remove_session("delete-by-name").unwrap();
        assert_eq!(mgr.list().len(), 0);
    }

    fn make_db_dir() -> std::path::PathBuf {
        tempfile::tempdir().unwrap().path().join("db")
    }

    #[test]
    fn test_db_create_and_get_session() {
        let path = make_db_dir().join("sessions.db");
        let db = SessionDB::new(&path).unwrap();
        let session = db.get_or_create_session("test-session").unwrap();
        assert_eq!(session.name, "test-session");
        assert!(!session.id.is_empty());
        let session2 = db.get_or_create_session("test-session").unwrap();
        assert_eq!(session.id, session2.id);
    }

    #[test]
    fn test_db_save_and_load_messages() {
        let path = make_db_dir().join("state2.db");
        let db = SessionDB::new(&path).unwrap();
        let session = db.get_or_create_session("msg-test").unwrap();
        let session_id = session.id.clone();
        let mut msgs = vec![
            Message::user("hello"),
            Message::assistant("hi there"),
            Message::user("how are you"),
        ];
        db.save_messages(&session_id, &mut msgs).unwrap();
        let loaded = db.load_messages(&session_id).unwrap();
        assert_eq!(loaded.len(), 3);
        assert_eq!(loaded[0].content.to_text(), "hello");
        assert_eq!(loaded[1].content.to_text(), "hi there");
    }

    #[test]
    fn test_db_messages_around() {
        let path = make_db_dir().join("around.db");
        let db = SessionDB::new(&path).unwrap();
        let session = db.get_or_create_session("around-test").unwrap();
        let sid = session.id.clone();
        let mut msgs: Vec<Message> = (0..10)
            .map(|i| Message::user(format!("message {}", i)))
            .collect();
        db.save_messages(&sid, &mut msgs).unwrap();
        let loaded = db.load_messages(&sid).unwrap();
        let anchor_id: i64 = loaded[5].id.unwrap();
        let result = db.get_messages_around(&sid, anchor_id, 2).unwrap();
        assert!(result.window.len() >= 3);
    }

    #[test]
    fn test_db_delete_session() {
        let path = make_db_dir().join("delete.db");
        let db = SessionDB::new(&path).unwrap();
        let sid = db.get_or_create_session("del-me").unwrap().id;
        let sid2 = db.get_or_create_session("del-me").unwrap().id;
        assert_eq!(sid, sid2, "same session should be returned");
        db.delete_session(&sid).unwrap();
        let sid3 = db.get_or_create_session("del-me").unwrap().id;
        assert_ne!(sid, sid3, "should create new session after delete");
    }

    #[test]
    fn test_db_list_sessions() {
        let path = make_db_dir().join("list.db");
        let db = SessionDB::new(&path).unwrap();
        db.get_or_create_session("session-a").unwrap();
        db.get_or_create_session("session-b").unwrap();
        let sessions = db.list_sessions(None, &[], 10, 0, false).unwrap();
        assert!(sessions.len() >= 2);
    }

    // ── split_after_compression tests ─────────────────────────────────────

    /// Tests that `split_after_compression` ends the parent session.
    ///
    /// given: a session with messages and no end_reason
    /// when: split_after_compression(parent_id) is called
    /// then: parent session has end_reason="compression" and ended_at set
    #[test]
    fn test_split_after_compression_creates_child_session() {
        let mut mgr = DBSessionManager::new_with_path(make_test_dir()).unwrap();
        let parent_id = {
            let parent = mgr.new_session("test-chat");
            parent.add_message(Message::user("hello"));
            parent.add_message(Message::assistant("hi there"));
            parent.id.clone()
        };
        mgr.incremental_save(None).unwrap();

        let child = mgr.split_after_compression(&parent_id).unwrap();

        // New session was created
        assert!(!child.id.is_empty());
        assert_ne!(child.id, parent_id, "child must have different id");

        // Parent is marked ended
        let parent_db = mgr.db.get_session(&parent_id).unwrap().unwrap();
        assert_eq!(
            parent_db.metadata.end_reason,
            Some("compression".to_string()),
            "parent end_reason should be 'compression'"
        );
        assert!(
            parent_db.metadata.ended_at.is_some(),
            "parent ended_at should be set"
        );
    }

    /// Tests that `split_after_compression` sets parent_session_id on the child.
    ///
    /// given: a parent session
    /// when: split_after_compression(parent_id) is called
    /// then: child session has parent_session_id = parent_id
    #[test]
    fn test_split_after_compression_sets_parent_session_id() {
        let mut mgr = DBSessionManager::new_with_path(make_test_dir()).unwrap();
        let parent_id = {
            let parent = mgr.new_session("test-chat-2");
            parent.add_message(Message::user("hello"));
            parent.id.clone()
        };
        mgr.incremental_save(None).unwrap();

        let child = mgr.split_after_compression(&parent_id).unwrap();

        // Child references parent
        let child_db = mgr.db.get_session(&child.id).unwrap().unwrap();
        assert_eq!(
            child_db.metadata.parent_session_id,
            Some(parent_id),
            "child parent_session_id should match parent"
        );
    }

    /// Tests that `split_after_compression` auto-numbers the child title.
    ///
    /// given: a session with title "chat-12345"
    /// when: split_after_compression is called
    /// then: child title is "chat-12345 (2)"
    #[test]
    fn test_split_after_compression_auto_numbers_title() {
        let mut mgr = DBSessionManager::new_with_path(make_test_dir()).unwrap();
        let parent_id = {
            let parent = mgr.new_session("chat-12345");
            parent.id.clone()
        };
        mgr.incremental_save(None).unwrap();

        let child = mgr.split_after_compression(&parent_id).unwrap();

        // Title should be "chat-12345 (2)"
        let child_db = mgr.db.get_session(&child.id).unwrap().unwrap();
        assert_eq!(
            child_db.metadata.title.as_deref(),
            Some("chat-12345 (2)"),
            "title should be auto-numbered"
        );
    }

    /// Tests that `split_after_compression` returns the child session.
    ///
    /// given: a session with messages
    /// when: split_after_compression(parent_id) is called
    /// then: returned session is the child with correct title
    #[test]
    fn test_split_after_compression_returns_child() {
        let mut mgr = DBSessionManager::new_with_path(make_test_dir()).unwrap();
        let parent_id = {
            let parent = mgr.new_session("split-test");
            parent.add_message(Message::user("msg1"));
            parent.add_message(Message::assistant("msg2"));
            parent.id.clone()
        };
        mgr.incremental_save(None).unwrap();

        let child = mgr.split_after_compression(&parent_id).unwrap();

        // Returned session is the child with auto-numbered title, stable name
        assert_eq!(child.id, child.id);
        assert_eq!(
            child.name,
            "split-test".to_string(),
            "name should be stable"
        );
        assert_eq!(
            child.metadata.title.as_deref(),
            Some("split-test (2)"),
            "title should be auto-numbered"
        );
    }

    // ── Concurrent write tests ────────────────────────────────────────────────

    /// Regression test: concurrent writes to the same session must not fail
    /// with "database is locked" under the jittered retry protocol.
    ///
    /// Spawns 10 threads, each appending messages to the same session.
    /// Without jittered retry, this would fail with SQLite locking errors
    /// because multiple threads compete for the WAL write lock.
    #[test]
    fn test_concurrent_writes_no_lock_errors() {
        use std::sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        };

        let dir = make_db_dir();
        let db = Arc::new(SessionDB::new(dir.join("concurrent.db")).unwrap());

        // Create a session shared by all threads
        let session = db.get_or_create_session("concurrent-test").unwrap();
        let sid = session.id;

        let num_threads = 10;
        let msgs_per_thread = 20;
        let mut handles = Vec::with_capacity(num_threads);

        let success_count = Arc::new(AtomicUsize::new(0));
        let error_count = Arc::new(AtomicUsize::new(0));

        for i in 0..num_threads {
            let db_clone = Arc::clone(&db);
            let sid_clone = sid.clone();
            let success = Arc::clone(&success_count);
            let errors = Arc::clone(&error_count);

            let handle = std::thread::spawn(move || {
                let msgs: Vec<Message> = (0..msgs_per_thread)
                    .map(|j| Message::user(format!("t{}-m{}", i, j)))
                    .collect();

                if db_clone
                    .save_new_messages(&sid_clone, &mut msgs.clone().into_boxed_slice())
                    .is_ok()
                {
                    success.fetch_add(1, Ordering::Relaxed);
                } else {
                    errors.fetch_add(1, Ordering::Relaxed);
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().expect("thread panicked");
        }

        let successes = success_count.load(Ordering::Relaxed);
        let errors = error_count.load(Ordering::Relaxed);

        assert_eq!(
            errors, 0,
            "Expected 0 lock errors but got {} (successes={}/{})",
            errors, successes, num_threads
        );
        assert_eq!(successes, num_threads);

        // Verify all messages were persisted
        let loaded = db.load_messages(&sid).unwrap();
        assert_eq!(loaded.len(), num_threads * msgs_per_thread);
    }

    /// Rename session title test
    ///
    /// given: a session with title "old-title"
    /// when: set_title is called with a new title
    /// then: in-memory metadata.title and DB title are both updated
    #[test]
    fn test_set_title_updates_memory_and_db() {
        let mut mgr = DBSessionManager::new_with_path(make_test_dir()).unwrap();
        let session = mgr.create_session("old-session-name");
        session.metadata.title = Some("old-title".to_string());
        let sid = session.id.clone();
        mgr.incremental_save(None).unwrap();

        mgr.set_title("new-title").unwrap();

        // In-memory cache updated
        let mem_session = mgr.sessions.get(&sid).unwrap();
        assert_eq!(mem_session.metadata.title.as_deref(), Some("new-title"));

        // DB updated
        let db_session = mgr.db.get_session(&sid).unwrap().unwrap();
        assert_eq!(db_session.metadata.title.as_deref(), Some("new-title"));
    }

    /// Rename session title test — duplicate title returns error
    ///
    /// given: two sessions with different titles, first one is active
    /// when: rename first to the title of the second
    /// then: error is returned
    #[test]
    fn test_set_title_unique_constraint() {
        let mut mgr = DBSessionManager::new_with_path(make_test_dir()).unwrap();
        mgr.create_session("first-session-name");
        mgr.create_session("second-session-name");

        // First session is active after creation
        let sid1 = mgr.active_session().unwrap().id.clone();
        {
            let s1 = mgr.active_session_mut().unwrap();
            s1.metadata.title = Some("target-title".to_string());
            let _ = mgr.db.set_title(&sid1, "target-title");
        }

        // Second session shares DB
        let sid2 = if let Some(s) = mgr.sessions.values().find(|s| s.id != sid1) {
            s.id.clone()
        } else {
            panic!("expected two sessions");
        };
        {
            let s2 = mgr.sessions.get_mut(&sid2).unwrap();
            s2.metadata.title = Some("other-title".to_string());
            let _ = mgr.db.set_title(sid2.as_str(), "other-title");
        }

        // Now switch to second session and try to rename to first session's title
        mgr.switch_session(&sid2).unwrap();
        assert!(mgr.set_title("target-title").is_err());
    }

    /// Test `save_compacted()` persists correctly after compaction.
    ///
    /// This is the regression test for the compact persistence bug:
    /// before the fix, `save()` did `new_count = messages.len() - persisted_message_count`,
    /// which after compaction caused usize underflow (messages.len() < persisted_message_count).
    /// Even if it didn't panic, DB old messages were never deleted, so restart would see
    /// a mix of old (uncompressed) and new (compacted) messages.
    ///
    /// given: a session with 20 messages saved to DB (persisted_message_count = 20)
    /// when: save_compacted() is called with 5 compaction-reduced messages
    /// then: DB contains exactly those 5 messages (old 20 deleted), not 25 (20+5)
    #[test]
    fn test_save_compacted_replaces_db_messages() {
        let mut mgr = DBSessionManager::new_with_path(make_test_dir()).unwrap();
        let session = mgr.new_session("compaction-test");
        let sid = session.id.clone();

        // Write 20 messages
        let mut msgs: Vec<Message> = (0..20)
            .map(|i| Message::user(format!("message {} of the original 20", i)))
            .collect();
        mgr.db.save_messages(&sid, &mut msgs).unwrap();
        mgr.load(Some(&sid)).unwrap();
        let active = mgr.active_session_mut().unwrap();
        active.persisted_message_count = 20;

        // Verify: DB has 20 messages
        let loaded_before = mgr.db.load_messages(&sid).unwrap();
        assert_eq!(loaded_before.len(), 20, "should have 20 original messages");

        // Simulate compaction: messages reduced from 20 to 5
        let compacted_msgs: Vec<Message> = vec![
            Message::system("You are a helpful coding assistant."),
            Message::user("Question 1: What is Rust?".to_string()),
            Message::assistant("Answer 1: Rust is a systems programming language.".to_string()),
            Message::user("Question 2: What is borrowing?".to_string()),
            Message::assistant("Answer 2: Borrowing is Rust's ownership mechanism.".to_string()),
        ];

        // Call save_compacted — this should replace ALL messages
        mgr.save_compacted(&sid, &compacted_msgs).unwrap();

        // Verify: DB has exactly 5 messages (old 20 should be GONE)
        let loaded_after = mgr.db.load_messages(&sid).unwrap();
        assert_eq!(
            loaded_after.len(),
            5,
            "DB should have exactly 5 compacted messages, not 25 (20+5 mix)"
        );

        // Verify: old messages are gone from DB
        for msg in &loaded_after {
            assert!(
                !msg.content.to_text().contains("original 20"),
                "old message should not be in DB after compaction"
            );
        }

        // Verify: in-memory session also updated
        let mem_session = mgr.sessions.get(&sid).unwrap();
        assert_eq!(
            mem_session.messages.len(),
            5,
            "in-memory messages should be 5"
        );
        assert_eq!(
            mem_session.persisted_message_count, 5,
            "persisted_message_count should match"
        );

        // Verify: message_count in session metadata updated
        let db_session = mgr.db.get_session(&sid).unwrap().unwrap();
        assert_eq!(
            db_session.metadata.message_count, 5,
            "session metadata message_count should be 5"
        );
    }

    /// ChildSessionGuard creates a child session.
    ///
    /// Given: a manager with an active session
    /// When: spawn_child_session is called
    /// Then: the guard holds the child session and parent_id
    #[test]
    fn test_spawn_subagent_session_leaves_parent_active() {
        let mut mgr = DBSessionManager::new_with_path(make_test_dir()).unwrap();
        mgr.init().unwrap();
        let _parent = mgr.create_session("parent");
        let parent_id = mgr.active_session_id().unwrap();

        let spawned = mgr.spawn_session_for_subagent("child-task").unwrap();
        assert_eq!(spawned.session_id.len(), 36); // UUID
        assert_eq!(spawned.parent_session_id, parent_id);

        // Parent's active_session_id is NOT changed
        assert_eq!(
            mgr.active_session_id(),
            Some(parent_id),
            "parent should remain active"
        );
    }

    /// spawn_session_for_subagent persists parent_session_id to the DB.
    #[test]
    fn test_spawn_subagent_session_sets_parent_id_in_db() {
        let mut mgr = DBSessionManager::new_with_path(make_test_dir()).unwrap();
        mgr.init().unwrap();
        let _parent = mgr.create_session("parent");
        let parent_id = mgr.active_session_id().unwrap();

        let spawned = mgr.spawn_session_for_subagent("child-task").unwrap();

        // Check DB directly (child is not active, so we can't see it through manager)
        let db_child = mgr.db.get_session(&spawned.session_id).unwrap().unwrap();
        assert_eq!(
            db_child.metadata.parent_session_id,
            Some(parent_id.clone()),
            "child should have parent_session_id set"
        );
    }

    /// spawn_session_for_subagent fails when there is no active session.
    #[test]
    fn test_spawn_no_active_session_fails() {
        let mut mgr = DBSessionManager::new_with_path(make_test_dir()).unwrap();
        mgr.init().unwrap();
        match mgr.spawn_session_for_subagent("unexpected") {
            Err(ref e) => {
                assert!(
                    e.to_string().contains("No active session"),
                    "error: {}",
                    e.to_string()
                );
            }
            Ok(_) => panic!("expected Err"),
        }
    }

    /// Create a goal in the DB.
    ///
    /// Given: a session manager with an active session
    /// When: create_goal is called with a goal ID, text, and max turns
    /// Then: the goal record is persisted with correct fields
    #[test]
    fn test_create_goal_persists_in_db() {
        let mut mgr = DBSessionManager::new_with_path(make_test_dir()).unwrap();
        mgr.create_session("test-session");
        let sid = mgr.active_session_id().unwrap();

        let goal_id = "goal-create-test".to_string();
        mgr.db
            .create_goal(&goal_id, "Write a web scraper", 10, Some(&sid))
            .unwrap();

        let row = mgr.db.get_goal(&goal_id).unwrap().unwrap();
        assert_eq!(row.0, "Write a web scraper");
        assert_eq!(row.1, "pending");
        assert_eq!(row.2, 10);
        assert_eq!(row.3, 0);
    }

    /// Update a goal's status and turns.
    ///
    /// Given: a goal created in the DB
    /// When: update_goal is called with new status and turns
    /// Then: the goal record reflects the updates
    #[test]
    fn test_update_goal_changes_status_and_turns() {
        let mut mgr = DBSessionManager::new_with_path(make_test_dir()).unwrap();
        mgr.create_session("test-session");
        let sid = mgr.active_session_id().unwrap();

        let goal_id = "goal-update-test";
        mgr.db
            .create_goal(goal_id, "Build an API", 20, Some(&sid))
            .unwrap();

        mgr.db
            .update_goal(
                goal_id,
                "active",
                3,
                Some("continue"),
                Some("3 nodes remaining"),
            )
            .unwrap();

        let row = mgr.db.get_goal(goal_id).unwrap().unwrap();
        assert_eq!(row.1, "active");
        assert_eq!(row.3, 3);
    }

    /// Pause a goal and verify it's persisted.
    ///
    /// Given: an active goal
    /// When: pause_goal is called with a reason
    /// Then: the goal status becomes 'paused' and paused_reason is set
    #[test]
    fn test_pause_goal_persists_reason() {
        let mut mgr = DBSessionManager::new_with_path(make_test_dir()).unwrap();
        mgr.create_session("test-session");
        let sid = mgr.active_session_id().unwrap();

        let goal_id = "goal-pause-test";
        mgr.db
            .create_goal(goal_id, "Scrape product pages", 10, Some(&sid))
            .unwrap();
        mgr.db
            .update_goal(goal_id, "active", 5, None, None)
            .unwrap();

        mgr.db
            .pause_goal(goal_id, "budget low, will resume")
            .unwrap();

        let row = mgr.db.get_goal(goal_id).unwrap().unwrap();
        assert_eq!(row.1, "paused");
        assert_eq!(row.4, Some("budget low, will resume".to_string()));
    }

    /// Resume a goal, with budget reset.
    ///
    /// Given: a paused goal with turns_used = 8
    /// When: resume_goal is called with reset_budget = true
    /// Then: status becomes 'active' and turns_used resets to 0
    #[test]
    fn test_resume_goal_resets_budget() {
        let mut mgr = DBSessionManager::new_with_path(make_test_dir()).unwrap();
        mgr.create_session("test-session");
        let sid = mgr.active_session_id().unwrap();

        let goal_id = "goal-resume-test";
        mgr.db
            .create_goal(goal_id, "Analyze dataset", 20, Some(&sid))
            .unwrap();
        mgr.db
            .update_goal(goal_id, "active", 8, None, None)
            .unwrap();
        mgr.db.pause_goal(goal_id, "paused for later").unwrap();

        mgr.db.resume_goal(goal_id, true).unwrap();

        let row = mgr.db.get_goal(goal_id).unwrap().unwrap();
        assert_eq!(row.1, "active");
        assert_eq!(row.3, 0);
        assert!(row.4.is_none());
    }

    /// Delete a goal (cascades to goal_nodes).
    ///
    /// Given: a goal with nodes in goal_nodes table
    /// When: delete_goal is called
    /// Then: the goal and its nodes are removed
    #[test]
    fn test_delete_goal_removes_goal_and_nodes() {
        let mut mgr = DBSessionManager::new_with_path(make_test_dir()).unwrap();
        mgr.create_session("test-session");
        let sid = mgr.active_session_id().unwrap();

        let goal_id = "goal-delete-test";
        mgr.db
            .create_goal(goal_id, "Delete me", 10, Some(&sid))
            .unwrap();
        mgr.db.create_goal_node(goal_id, "First step").unwrap();
        mgr.db.create_goal_node(goal_id, "Second step").unwrap();

        assert_eq!(mgr.db.list_goal_nodes(goal_id).unwrap().len(), 2);

        mgr.db.delete_goal(goal_id).unwrap();

        assert!(mgr.db.get_goal(goal_id).unwrap().is_none());
        assert_eq!(mgr.db.list_goal_nodes(goal_id).unwrap().len(), 0);
    }

    /// List goals with optional status filter.
    ///
    /// Given: multiple goals with different statuses
    /// When: list_goals is called with a status filter
    /// Then: only matching goals are returned
    #[test]
    fn test_list_goals_filters_by_status() {
        let mut mgr = DBSessionManager::new_with_path(make_test_dir()).unwrap();
        mgr.create_session("test-session");
        let sid = mgr.active_session_id().unwrap();

        mgr.db
            .create_goal("goal-a", "Goal A", 10, Some(&sid))
            .unwrap();
        mgr.db
            .create_goal("goal-b", "Goal B", 20, Some(&sid))
            .unwrap();
        mgr.db
            .update_goal("goal-a", "active", 0, None, None)
            .unwrap();

        let all = mgr.db.list_goals(None).unwrap();
        assert_eq!(all.len(), 2);

        let active = mgr.db.list_goals(Some("active")).unwrap();
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].0, "goal-a");
    }

    /// Create goal and nodes, then list nodes.
    ///
    /// Given: a goal with multiple plan nodes
    /// When: list_goal_nodes is called
    /// Then: all nodes are returned in insertion order
    #[test]
    fn test_goal_node_lifecycle() {
        let mut mgr = DBSessionManager::new_with_path(make_test_dir()).unwrap();
        mgr.create_session("test-session");
        let sid = mgr.active_session_id().unwrap();

        let goal_id = "goal-nodes-test";
        mgr.db
            .create_goal(goal_id, "Multi-step plan", 10, Some(&sid))
            .unwrap();

        mgr.db.create_goal_node(goal_id, "Step 1").unwrap();
        mgr.db.create_goal_node(goal_id, "Step 2").unwrap();
        mgr.db.create_goal_node(goal_id, "Step 3").unwrap();

        let nodes = mgr.db.list_goal_nodes(goal_id).unwrap();
        assert_eq!(nodes.len(), 3);
        assert_eq!(nodes[0].0, "Step 1");
        assert_eq!(nodes[1].0, "Step 2");
        assert_eq!(nodes[2].1, "pending");
    }

    /// Update a goal node's status.
    ///
    /// Given: a goal with a pending node
    /// When: update_goal_node is called with new status
    /// Then: the node's status changes accordingly
    #[test]
    fn test_update_goal_node() {
        let mut mgr = DBSessionManager::new_with_path(make_test_dir()).unwrap();
        mgr.create_session("test-session");
        let sid = mgr.active_session_id().unwrap();

        let goal_id = "goal-node-update-test";
        mgr.db
            .create_goal(goal_id, "Update node test", 10, Some(&sid))
            .unwrap();

        // We need to know the row_id; create_node returns it but we can't get it back via list_goal_nodes
        // Instead test the update by creating a node and then modifying it
        let mgr_ref = &mgr.db;

        {
            mgr_ref.with_conn_mut(|conn| {
                conn.execute(
                    "INSERT INTO goal_nodes (goal_id, title, status, created_at) VALUES (?, ?, 'pending', ?)",
                    params![goal_id, "Node A", now_ts() as f64],
                )?;
                Ok(())
            }).unwrap();
        }

        mgr.db
            .update_goal_node(goal_id, "Node A", "done", "Completed successfully")
            .unwrap();

        let nodes = mgr.db.list_goal_nodes(goal_id).unwrap();
        assert_eq!(nodes[0].0, "Node A");
        assert_eq!(nodes[0].1, "done");
    }

    /// create_goal with no owner_session sets owner to NULL.
    ///
    /// Given: a session manager
    /// When: create_goal is called without owner_session
    /// Then: the goal is created with NULL owner_session
    #[test]
    fn test_create_goal_without_owner() {
        let mut mgr = DBSessionManager::new_with_path(make_test_dir()).unwrap();
        mgr.create_session("test-session");
        let _sid = mgr.active_session_id().unwrap();

        let goal_id = "goal-no-owner-test";
        mgr.db
            .create_goal(goal_id, "Unowned goal", 5, None)
            .unwrap();

        let row = mgr.db.get_goal(goal_id).unwrap().unwrap();
        assert_eq!(row.0, "Unowned goal");
        assert!(row.2 == 5, "max_turns should be 5");

        let all = mgr.db.list_goals(None).unwrap();
        assert!(all.iter().any(|g| g.0 == goal_id));
    }

    #[test]
    fn test_new_with_agent_creates_separate_directories() {
        let temp_dir = tempfile::tempdir().unwrap();
        let agents_dir = temp_dir.path().join("agents");
        
        // Test manager agent creates directory
        let manager_dir = agents_dir.join("manager");
        let _mgr1 = DBSessionManager::new_with_path(manager_dir.clone()).unwrap();
        
        // Test worker agent creates directory
        let worker_dir = agents_dir.join("worker");
        let _mgr2 = DBSessionManager::new_with_path(worker_dir.clone()).unwrap();
        
        // Verify directories were created
        assert!(agents_dir.exists());
        assert!(manager_dir.exists());
        assert!(worker_dir.exists());
    }

    #[test]
    fn test_new_with_agent_none_uses_default() {
        let agent_name: Option<&str> = None;
        let agent_dir = agent_name
            .and_then(|n| if n.is_empty() { None } else { Some(n) })
            .unwrap_or("default");
        
        assert_eq!(agent_dir, "default");
    }

    #[test]
    fn test_new_with_agent_worker_creates_worker_dir() {
        let temp_dir = tempfile::tempdir().unwrap();
        let agents_dir = temp_dir.path().join("agents");
        
        let _mgr = DBSessionManager::new_with_path(agents_dir.join("worker")).unwrap();
        
        let worker_dir = agents_dir.join("worker");
        assert!(worker_dir.exists());
    }

    #[test]
    fn test_new_with_agent_manager_creates_manager_dir() {
        let temp_dir = tempfile::tempdir().unwrap();
        let agents_dir = temp_dir.path().join("agents");
        
        let _mgr = DBSessionManager::new_with_path(agents_dir.join("manager")).unwrap();
        
        let manager_dir = agents_dir.join("manager");
        assert!(manager_dir.exists());
    }

    #[test]
    fn test_agent_isolation() {
        use oben_models::Message;
        
        let temp_dir = tempfile::tempdir().unwrap();
        
        // Create separate db paths for manager and worker
        let manager_dir = temp_dir.path().join("manager");
        let worker_dir = temp_dir.path().join("worker");
        
        // Create session for manager
        let mut mgr_manager = DBSessionManager::new_with_path(manager_dir.clone()).unwrap();
        mgr_manager.create_session("manager-session");
        mgr_manager.save_session(None).unwrap();
        
        // Create session for worker
        let mut mgr_worker = DBSessionManager::new_with_path(worker_dir.clone()).unwrap();
        mgr_worker.create_session("worker-session");
        mgr_worker.save_session(None).unwrap();
        
        // Manager should not see worker's session
        mgr_manager.init().unwrap();
        mgr_worker.init().unwrap();
        
        // Each manager has its own session
        assert_eq!(mgr_manager.list_sessions_ref().len(), 1);
        assert_eq!(mgr_worker.list_sessions_ref().len(), 1);
        
        // Verify sessions are different
        assert_ne!(
            mgr_manager.active_session_id(),
            mgr_worker.active_session_id()
        );
    }
}
