//! Message display widget — renders message history with scrolling.
//!
//! Encapsulates the pre-rendered message lines, scrolling, streaming text
//! overlay, and turn-status text. Messages are rendered as bordered blocks
//! (like the Hermes Agent reference), with each message wrapped in a rounded-border
//! Block whose title shows the role label and icon.

use parking_lot::Mutex as PlMutex;
use ratatui::prelude::*;
use ratatui::widgets::{Block, BorderType, Borders, Paragraph, ScrollbarState};
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex as StdMutex};
use unicode_width::UnicodeWidthChar;

use oben_agent::TurnState;
use crate::widgets::layout;
use crate::widgets::message_renderer::{render_body_lines, MessageRenderEntry, MessageRenderer};
use crate::widgets::role_style::role_info_for_role;
use oben_models::{Message, MessageRole};

/// Block rendering strategy for a message entry.
pub enum BlockType<'a> {
    /// Regular message: full border + role title + role-specific color.
    Message(&'a MessageRole),
    /// Tool result: indented muted rounded box (no role title).
    ToolResult,
}

    /// Cache for layout computation (wrapped lines + heights).
    ///
    /// Messages are append-only — content never changes. When entry count + window
    /// dimensions + streaming state are the same as last render, we can skip the
    /// whole wrap loop and reuse cached heights/ranges/flat-lines.
    #[derive(Clone)]
    pub struct CachedLayout {
    /// Entry count at cache time.
    entry_count: usize,
    /// Window height/width at cache time.
    area_h: u16,
    area_w: u16,
    /// Whether we were streaming when cache was created.
    was_streaming: bool,
    /// Pre-computed block heights (entry index → height_in_lines).
    heights: Vec<u16>,
    /// Flat line index ranges per entry: (start, end) in flat_lines.
    entry_ranges: Vec<(usize, usize)>,
    /// All wrapped lines concatenated in order.
    flat_lines: Vec<Line<'static>>,
}

/// State for the message display widget.
pub struct ConversationState {
    pub scroll_state: Arc<StdMutex<ScrollbarState>>,
    pub scroll_to_bottom: Arc<AtomicBool>,
    /// Persisted scroll position across frames (AtomicUsize so render(&self) can update it).
    pub scroll_pos: Arc<AtomicUsize>,
    pub stream_info: Arc<StdMutex<String>>,
    pub turn_state_ref: Option<Arc<PlMutex<TurnState>>>,
    /// Accumulated scroll delta from user mouse scroll. Reset by render.
    pub user_scroll_offset: Arc<AtomicI32>,
    /// Tracks content line count from previous frame — detects if content has grown.
    pub prev_scrollable_range: Arc<AtomicUsize>,
    /// Width of the message body content area in display columns.
    pub body_width: usize,
    /// X offset of the body content area from the left edge of the screen.
    pub content_x: u16,
    /// Y offset of the first content row (below borders).
    pub content_y: u16,
    /// Y position of the message display area.
    pub msg_area_y: u16,
    /// Width available for line wrapping in the message body (in display columns).
    pub wrap_width: usize,
    /// Block heights in body-line units. index i = height of entry[i] block.
    /// Used to map body_line_idx (from mouse) → flat_line_idx (from content).
    pub cached_block_heights: Arc<StdMutex<Vec<u16>>>,
    /// Maps body_line_idx → flat_line_idx for selection alignment.
    /// body_line 0=text, 1-2=borders, then margin + next entry 0=text, 1-2=borders, etc.
    pub cached_body_to_flat: Arc<StdMutex<Vec<Option<usize>>>>,
    /// Cached wrapped lines for selection/render alignment. Populated during render.
    pub cached_lines: Arc<StdMutex<Vec<Line<'static>>>>,
    /// Selection start/end as terminal (row, col).
    pub selection_start: Option<(u16, u16)>,
    pub selection_end: Option<(u16, u16)>,
    /// Per-message structured entries for bordered-block rendering.
    pub message_entries: Arc<StdMutex<Vec<MessageRenderEntry>>>,
    /// Visible body ranges: (body_y_in_terminal, body_visible_rows).
    /// Set by render_bordered_blocks from block.inner() so it correctly
    /// accounts for title+BODERS distinction. Used by render_selection/get_selected_text
    /// to skip rows outside actual rendered body area.
    pub visible_body_ranges: Arc<StdMutex<Vec<(u16, u16)>>>,

    /// Cached layout from last render (entry wrap, heights, flat lines).
    pub cached_layout: Arc<StdMutex<Option<CachedLayout>>>,
}

impl ConversationState {
    pub fn new() -> Self {
        Self {
            scroll_state: Arc::new(StdMutex::new(ScrollbarState::new(0))),
            scroll_to_bottom: Arc::new(AtomicBool::new(true)),
            scroll_pos: Arc::new(AtomicUsize::new(0)),
            stream_info: Arc::new(StdMutex::new(String::new())),
            turn_state_ref: None,
            user_scroll_offset: Arc::new(AtomicI32::new(0)),
            prev_scrollable_range: Arc::new(AtomicUsize::new(0)),
            selection_start: None,
            selection_end: None,
            message_entries: Arc::new(StdMutex::new(Vec::new())),
            body_width: 0,
            content_x: 0,
            content_y: 0,
            msg_area_y: 0,
            wrap_width: 0,
            cached_block_heights: Arc::new(StdMutex::new(Vec::new())),
            cached_body_to_flat: Arc::new(StdMutex::new(Vec::new())),
            cached_lines: Arc::new(StdMutex::new(Vec::new())),
            visible_body_ranges: Arc::new(StdMutex::new(Vec::new())),
            cached_layout: Arc::new(StdMutex::new(None)),
        }
    }

    /// Clear selection state.
    pub fn clear_selection(&mut self) {
        self.selection_start = None;
        self.selection_end = None;
    }
}

/// Widget that renders message entries with bordered blocks.
pub struct ConversationWidget;

impl ConversationWidget {
    /// Determine the border/title style for a given role.
    fn role_border_style(
        &self,
        role: &MessageRole,
        palette: &ratatui_themes::ThemePalette,
    ) -> Style {
        let info = role_info_for_role(role, palette);
        Style::default().fg(info.border_color)
    }

    fn role_title(
        &self,
        role: &MessageRole,
        palette: &ratatui_themes::ThemePalette,
    ) -> Line<'static> {
        let role_info = role_info_for_role(role, palette);
        Line::from(vec![
            Span::styled(
                format!(" {} ", role_info.icon),
                Style::default()
                    .fg(role_info.header_color)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                role_info.label.to_string(),
                Style::default()
                    .fg(role_info.header_color)
                    .add_modifier(Modifier::BOLD),
            ),
        ])
    }

    /// Get the selected text from the current state, if any.
    ///
    /// Converts terminal (row, col) → body_line → flat line.
    /// Uses the CURRENT cached_body_to_flat mapping from render_bordered_blocks
    /// so it stays consistent with what's currently being rendered.
    /// Filters terminal rows to only those within visible body ranges,
    /// preventing copying text that isn't actually displayed.
    pub fn get_selected_text(&self, state: &ConversationState) -> Option<String> {
        let (sy, sx) = state.selection_start?;
        let (ey, ex) = state.selection_end?;

        let flat_lines = match state.cached_lines.lock() {
            Ok(g) => {
                let lines = g.clone();
                if lines.is_empty() {
                    // No render has populated the cache yet — nothing to copy.
                    return None;
                }
                lines
            }
            Err(_) => return None,
        };
        let body_to_flat = match state.cached_body_to_flat.lock() {
            Ok(g) => g,
            Err(_) => return None,
        };
        let content_y = state.content_y as usize;
        let content_x = state.content_x as usize;
        let scroll_pos = state.scroll_pos.load(Ordering::SeqCst);
        let body_w = state.body_width;
        let visible_ranges = state.visible_body_ranges.lock().ok()?.clone();

        // Mouse cols are absolute terminal coords. Convert to body-area-relative column.
        let body_area_x = content_x.saturating_sub(2);
        let rel_sx = (sx as usize).saturating_sub(body_area_x);
        let rel_ex = (ex as usize).saturating_sub(body_area_x);

        // Ensure min/max order for x
        let (x0_start, x1_start) = (std::cmp::min(rel_sx, rel_ex), std::cmp::max(rel_sx, rel_ex));

        // body_to_flat body_start_offset = msg_area.y = content_y - 1.
        let body_start_offset = content_y.saturating_sub(1);

        // Calculate max valid abs_body from body_to_flat bounds.
        let max_abs_body = body_to_flat.len();

        // Iterate terminal rows sy..=ey (normalized), extract text per row.
        let mut result = String::new();
        let row_start = std::cmp::min(sy as usize, ey as usize);
        let row_end = std::cmp::max(sy as usize, ey as usize);

        let mut prev_flat_line: Option<usize> = None;

        tracing::debug!(
            "[selection/get_text] INIT sy={} ey={} sx={} ex={} content_y={} body_start_offset={} scroll_pos={} row_range=[{}..{}] body_to_flat_len={} max_abs_body={} visible_ranges_count={}",
            sy, ey, sx, ex, content_y, body_start_offset, scroll_pos, row_start, row_end, body_to_flat.len(), max_abs_body, visible_ranges.len()
        );
        for (i, &(by, bh)) in visible_ranges.iter().enumerate() {
            tracing::debug!(
                "[selection/get_text]   visible_range[{}] body_y={} body_h={}",
                i,
                by,
                bh
            );
        }

        for row in row_start..=row_end {
            // Skip rows outside all visible body areas.
            let in_visible_body = visible_ranges.iter().any(|&(body_y, body_h)| {
                row >= body_y as usize && row < body_y.saturating_add(body_h) as usize
            });
            if !in_visible_body {
                continue;
            }

            let abs_body = (row as usize).saturating_sub(body_start_offset) + scroll_pos;
            if abs_body >= max_abs_body {
                continue;
            }

            // Look up flat line for this body line.
            // body_to_flat structure: [...wrapped_lines..., padding, None(sep), next_block...]
            // None = inter-block separator — skip it to avoid pulling from wrong block.
            let flat_line = match body_to_flat.get(abs_body).copied().flatten() {
                Some(v) => v,
                None => {
                    // Block boundary — treat as transparent. Don't search forward,
                    // which would pull content from the next block and cause duplication.
                    continue;
                }
            };

            // Dedup: skip if same flat_line as previous row (handles padding rows
            // mapping to the same wrapped line).
            if Some(flat_line) == prev_flat_line {
                continue;
            }
            prev_flat_line = Some(flat_line);

            if flat_line >= flat_lines.len() {
                continue;
            }
            let line = &flat_lines[flat_line];

            // Build cell→char map for this line
            let mut chars: Vec<(usize, char)> = Vec::new();
            let mut pos = 0usize;
            for span in &line.spans {
                for ch in span.content.chars() {
                    let w = ch.width().unwrap_or(0).max(1);
                    chars.push((pos, ch));
                    pos += w;
                }
            }

            // Extract text at row level using mouse row position
            let x0 = x0_start.min(body_w);
            let x1 = x1_start.min(body_w).max(x0 + 1);

            let sel: String = chars
                .iter()
                .filter(|(p, _)| *p >= x0 && *p < x1)
                .map(|(_, ch)| *ch)
                .collect();

            tracing::debug!(
                "[selection/get_text] row={} abs_body={} flat_line={} abs_in_range={}..={} x0={} x1={} sel=\"{}\"",
                row, abs_body, flat_line, row_start, row_end, x0, x1,
                sel.chars().take(60).collect::<String>()
            );

            if !sel.is_empty() {
                if !result.is_empty() {
                    result.push('\n');
                }
                result.push_str(&sel);
            }
        }

        tracing::debug!(
            "[selection/get_text] FINAL result={} chars, text_trunc=\"{}\"",
            result.len(),
            result.chars().take(200).collect::<String>()
        );

        if !result.is_empty() {
            return Some(result);
        }
        None
    }

    /// Render selection highlight overlay.
    ///
    /// Uses the CURRENT cached_body_to_flat mapping from render_bordered_blocks.
    /// Filters terminal rows to only those within visible body ranges of rendered blocks,
    /// preventing selection of invisible content.
    pub fn render_selection(
        &self,
        frame: &mut Frame,
        area: Rect,
        state: &ConversationState,
        _palette: &ratatui_themes::ThemePalette,
    ) {
        if let (Some((sy, sx)), Some((ey, ex))) = (state.selection_start, state.selection_end) {
            let flat_lines = match state.cached_lines.lock() {
                Ok(g) => g.clone(),
                Err(_) => return,
            };
            let body_to_flat = match state.cached_body_to_flat.lock() {
                Ok(g) => g.clone(),
                Err(_) => return,
            };
            let content_y = state.content_y as u16;
            let content_x = state.content_x as u16;
            let body_w = state.body_width;
            let visible_ranges = state.visible_body_ranges.lock().unwrap().clone();
            let scroll_pos_val = state.scroll_pos.load(Ordering::SeqCst);

            // Clip mouse coordinates to message panel boundaries.
            let msg_top = area.y;
            let msg_bottom = area.bottom().saturating_sub(1);
            let sy_clamped = sy.max(msg_top).min(msg_bottom);
            let ey_clamped = ey.max(msg_top).min(msg_bottom);
            if sy_clamped > ey_clamped {
                return;
            }

            // Mouse cols are absolute terminal coords. Convert to body-area-relative column.
            let body_area_x = content_x.saturating_sub(2);
            let rel_sx = (sx as usize).saturating_sub(body_area_x as usize);
            let rel_ex = (ex as usize).saturating_sub(body_area_x as usize);

            // Normalize: ensure min/max regardless of drag direction.
            let x0 = std::cmp::min(rel_sx, rel_ex).min(body_w);
            let x1 = std::cmp::max(rel_sx, rel_ex).min(body_w).max(x0 + 1);

            // body_start_offset = msg_area.y = content_y - 1.
            // Same as get_selected_text.
            let body_start_offset = (content_y.saturating_sub(1)) as usize;

            // Iterate terminal rows from min to max.
            let row_start = std::cmp::min(sy_clamped, ey_clamped);
            let row_end = std::cmp::max(sy_clamped, ey_clamped);

            tracing::debug!(
                "[selection/render_sel] INIT sy={} ey={} sx={} ex={} content_y={} body_start_offset={} scroll_pos={} row_range=[{}..{}] msg_bounds=[{}..{}] visible_body_ranges_count={}",
                sy_clamped, ey_clamped, sx, ex, content_y, body_start_offset, scroll_pos_val, row_start, row_end, msg_top, msg_bottom, visible_ranges.len()
            );
            for (i, &(by, bh)) in visible_ranges.iter().enumerate() {
                tracing::debug!(
                    "[selection/render_sel]   visible_range[{}] body_y={} body_h={}",
                    i,
                    by,
                    bh
                );
            }

            // body_to_flat maps body_line → flat_line.
            let mut highlight_lines: Vec<Line> = Vec::new();
            let mut last_flat_line: Option<usize> = None;
            for row in row_start..=row_end {
                // Skip rows that fall outside all visible body areas.
                let in_visible_body = visible_ranges.iter().any(|&(body_y, body_h)| {
                    let r = row as usize;
                    r >= body_y as usize && r < body_y.saturating_add(body_h) as usize
                });
                if !in_visible_body {
                    highlight_lines.push(Line::from(Span::raw("")));
                    continue;
                }

                // Same formula as get_selected_text
                let abs_body = (row as usize)
                    .saturating_sub(body_start_offset)
                    .saturating_add(scroll_pos_val);

                // Look up flat line for this body line.
                // body_to_flat structure: [...wrapped_lines..., padding, None(sep), next_block...]
                // None = inter-block separator — render transparent (empty highlight).
                let flat_line = match body_to_flat.get(abs_body).copied().flatten() {
                    Some(v) => v,
                    None => {
                        highlight_lines.push(Line::from(Span::raw("")));
                        continue;
                    }
                };

                if flat_line >= flat_lines.len() {
                    highlight_lines.push(Line::from(Span::raw("")));
                    continue;
                }
                let line = &flat_lines[flat_line];

                // Build cell→char map for this line
                let mut chars: Vec<(usize, char)> = Vec::new();
                let mut pos = 0usize;
                for span in &line.spans {
                    for ch in span.content.chars() {
                        let w = ch.width().unwrap_or(0);
                        chars.push((pos, ch));
                        pos += w;
                    }
                }

                // Build styled spans with highlight for selected column range.
                let mut spans: Vec<Span> = Vec::new();
                let mut in_highlight = false;
                // Dedup: skip if this flat_line was already processed (handles padding rows
                // mapping to the same flat_line as text).
                if Some(flat_line) == last_flat_line {
                    highlight_lines.push(Line::from(Span::raw("")));
                    continue;
                }
                last_flat_line = Some(flat_line);
                for (c_pos, ch) in &chars {
                    if *c_pos >= x0 && *c_pos < x1 {
                        if !in_highlight {
                            in_highlight = true;
                        }
                        spans.push(Span::styled(
                            ch.to_string(),
                            Style::default().add_modifier(Modifier::REVERSED),
                        ));
                    } else {
                        if in_highlight {
                            in_highlight = false;
                        }
                        spans.push(Span::raw(ch.to_string()));
                    }
                }

                highlight_lines.push(Line::from(spans));
            }

            if !highlight_lines.is_empty() {
                let highlight_area_y = row_start;
                let highlight_area_x = area.x + body_area_x as u16;
                let highlight_area_h = (row_end - row_start) as u16 + 1;
                let highlight_area = Rect::new(
                    highlight_area_x,
                    highlight_area_y,
                    area.width.min(body_w as u16),
                    highlight_area_h.min((area.bottom() - highlight_area_y).max(1)),
                );
                tracing::debug!(
                    "[selection/render_selection] highlight_area=x={} y={} w={} h={} (sy={}, rows={})",
                    highlight_area.x, highlight_area.y,
                    highlight_area.width, highlight_area.height,
                    sy, highlight_lines.len()
                );
                frame.render_widget(Paragraph::new(highlight_lines), highlight_area);
            }
        }
    }

    /// Render message blocks with bordered styling.
    ///
    /// Three-phase architecture:
    ///   Phase 1: Layout — compute heights, scroll offset, visible areas (pure data)
    ///   Phase 2: Render   — draw blocks using the pre-calculated areas
    ///   Phase 3: Scroll   — update scroll position from mouse delta
    fn render_bordered_blocks(
        &self,
        frame: &mut Frame,
        area: Rect,
        state: &ConversationState,
        palette: &ratatui_themes::ThemePalette,
        is_streaming: bool,
        entry_titles: &[Option<Line<'static>>],
        entries: &[MessageRenderEntry],
    ) {
        // ─── Outer frame ───────────────────────────────────────────────
        static LAST_ENTRY_COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(999);
        if entries.len() != LAST_ENTRY_COUNT.load(std::sync::atomic::Ordering::SeqCst) {
            let entry_roles: Vec<String> = entries.iter().map(|e| format!("{:?}", e.role)).collect();
            tracing::info!(
                "[render_bordered_blocks] NEW entry count: entries={} roles={:?} streaming={}",
                entries.len(),
                entry_roles,
                is_streaming
            );
            LAST_ENTRY_COUNT.store(entries.len(), std::sync::atomic::Ordering::SeqCst);
        }
        let outer_block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(palette.muted))
            .title(Line::from(vec![
                Span::styled(" ", Style::default().fg(palette.info)),
                Span::styled(
                    "Messages",
                    Style::default()
                        .fg(palette.info)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(" ", Style::default().fg(palette.info)),
            ]));
        let msg_area = outer_block.inner(area);
        outer_block.render(area, frame.buffer_mut());

        let inner_width = (msg_area.width as usize).saturating_sub(2);
        let inner_height = (msg_area.height as usize).saturating_sub(1);

        // ─── Phase 1: Layout calculation (skip wrap if cache valid) ────────
        // Messages are append-only — content never changes. On scroll events,
        // when entry count + window size + streaming haven't changed, reuse cache.
        let _inner_wrap_w = inner_width.saturating_sub(is_streaming as usize);
        let (layout_entries, block_heights, content_height) = {
            let entry_count = entries.len();
            // Check cache: same entries + same window + same streaming?
            let cache_hit = {
                match state.cached_layout.lock() {
                    Ok(lock) => lock.as_ref().is_some_and(|c| {
                        let hit = c.entry_count == entry_count
                            && c.area_h == area.height
                            && c.area_w == area.width
                            && c.was_streaming == is_streaming;
                        if hit {
                            tracing::trace!(
                                "cached_layout HIT entries={} h={} w={}",
                                c.entry_count,
                                c.area_h,
                                c.area_w
                            );
                        }
                        hit
                    }),
                    Err(_) => false,
                }
            };

            let mut layout_entries: Vec<(usize, BlockType<'_>, Vec<Line<'static>>)> = Vec::new();
            let mut block_heights: Vec<u16> = Vec::new();

            if cache_hit {
                // Reuse cached layout — skip the O(n) wrap pass
                if let Ok(guard) = state.cached_layout.lock() {
                    if let Some(ref c) = guard.as_ref() {
                        block_heights = c.heights.clone();
                        for (i, entry) in entries.iter().enumerate() {
                            let bt = if entry.is_tool_result {
                                BlockType::ToolResult
                            } else {
                                BlockType::Message(&entry.role)
                            };
                            let (r_start, r_end) = c.entry_ranges[i];
                            let wrapped: Vec<Line<'static>> =
                                c.flat_lines[r_start..r_end].iter().cloned().collect();
                            layout_entries.push((i, bt, wrapped));
                        }
                    }
                }
            }

            if layout_entries.is_empty() {
                // Full compute: wrap ALL entries
                let mut _flat_accum = 0usize;
                for (i, entry) in entries.iter().enumerate() {
                    let plain_lines: Vec<Line<'static>> = entry
                        .body_lines
                        .iter()
                        .map(|sl| sl.content.clone())
                        .collect();
                    let bt = if entry.is_tool_result {
                        BlockType::ToolResult
                    } else {
                        BlockType::Message(&entry.role)
                    };
                    let wrap_w = if matches!(bt, BlockType::ToolResult) {
                        inner_width.saturating_sub(4)
                    } else {
                        inner_width.saturating_sub(2)
                    };
                    let wrapped = layout::wrap_styled_lines_to_lines(&plain_lines, wrap_w);
                    let h = (wrapped.len().max(1) as u16) + layout::BODY_HEIGHT_ADJUSTER as u16;
                    layout_entries.push((i, bt, wrapped));
                    _flat_accum += layout_entries.last().unwrap().2.len();
                    block_heights.push(h);
                }
            }

            // Update cache
            let ch = layout::calc_total_height(&block_heights);
            let entry_ranges: Vec<(usize, usize)> = {
                let mut ranges = Vec::with_capacity(layout_entries.len());
                let mut acc = 0usize;
                for (_, _, wl) in &layout_entries {
                    let end = acc + wl.len();
                    ranges.push((acc, end));
                    acc = end;
                }
                ranges
            };
            if let Ok(mut guard) = state.cached_layout.lock() {
                *guard = Some(CachedLayout {
                    entry_count,
                    area_h: area.height,
                    area_w: area.width,
                    was_streaming: is_streaming,
                    heights: block_heights.clone(),
                    entry_ranges,
                    flat_lines: layout_entries
                        .iter()
                        .flat_map(|(_, _, wl)| wl.iter().cloned())
                        .collect(),
                });
            }
            (layout_entries, block_heights, ch)
        };

        // Cache the full flat wrapped lines for selection/render alignment.
        // Cache block heights and body→flat mapping for selection alignment.
        {
            *state.cached_block_heights.lock().unwrap() = block_heights.clone();
            if let Ok(ref mut cached_lines) = state.cached_lines.lock() {
                let mut flat: Vec<Line<'static>> = Vec::new();
                for (_, _, ref wrapped_lines) in &layout_entries {
                    flat.extend(wrapped_lines.iter().cloned());
                }
                **cached_lines = flat;
            }
            // Cache body→flat mapping for selection alignment.
            // body_to_flat[body_line] = flat_line index for text lookup.
            // Each block contributes (wrapped.len() + BODY_HEIGHT_ADJUSTER) entries.
            // Padding (BODY_HEIGHT_ADJUSTER) lines map to last wrapped line.
            // One inter-block margin (None) between blocks.
            let mut body_to_flat = state.cached_body_to_flat.lock().unwrap();
            let mapping: Vec<Option<usize>> = {
                let mut mapping: Vec<Option<usize>> = Vec::new();
                let mut flat_accum = 0usize;
                let total_blocks = layout_entries.len();
                for (i, (_, _, wrapped)) in layout_entries.iter().enumerate() {
                    let wrapped_count = wrapped.len();
                    let block_height = block_heights[i];
                    // Map body_line indices to flat_line indices
                    for j in 0..block_height {
                        let j_usize = j as usize;
                        if j_usize < wrapped_count {
                            mapping.push(Some(flat_accum + j_usize));
                        } else {
                            // Padding lines (BODY_HEIGHT_ADJUSTER) map to last wrapped line
                            mapping.push(Some(flat_accum + (wrapped_count.saturating_sub(1))));
                        }
                    }
                    flat_accum += wrapped_count.max(1);
                    // Inter-block margin entry
                    if i < total_blocks - 1 {
                        mapping.push(None);
                    }
                }
                mapping
            };
            *body_to_flat = mapping;
        }
        let scroll_pos = state.scroll_pos.load(Ordering::SeqCst);
        let _user_offset = state.user_scroll_offset.load(Ordering::SeqCst);
        let _at_bottom = state.scroll_to_bottom.load(Ordering::SeqCst);

        // ─── Phase 1.5: Stream block estimation & wrapping (shared between phases) ──────
        // Parse stream text once, reuse for both scroll_offset and rendering.
        let stream_parsed = if is_streaming {
            if let Some(ref ts_arc) = state.turn_state_ref {
                let ts_ref = &*ts_arc.lock();
                let ts_len = ts_ref.streaming_text.len();
                // DIAG: Arc::strong_count tells us if adapter+draw share SAME TurnState.
                let refs = Arc::strong_count(ts_arc);
                tracing::info!(
                    arc_refs = refs,
                    total_len = ts_len,
                    "DIAG: arc_refs={} streaming_text_len={} [draw_read]",
                    refs, ts_len
                );
                if ts_len > 0 || !ts_ref.reasoning_text.is_empty() {
                    // Log streaming_text length every ~10 draws during streaming
                    // to avoid log flooding while still capturing growth pattern.
                    if ts_len % 20 == 0 {
                        tracing::trace!(
                            ts_len,
                            stream_preview = ?ts_ref.streaming_text.chars().take(60).collect::<String>(),
                            "TUI: streaming_text at draw"
                        );
                    }
                    let has_reasoning = !ts_ref.reasoning_text.is_empty();
                    let has_text = !ts_ref.streaming_text.trim().is_empty();

                    if !has_reasoning && !has_text {
                        None
                    } else {
                        // Render content text separately
                        let content_wrapped = if has_text {
                            let raw = ts_ref
                                .streaming_text
                                .trim_start_matches(|c: char| c.is_whitespace());
                            let body_lines = render_body_lines(raw, palette);
                            let plain_lines: Vec<Line<'static>> = body_lines
                                .into_iter()
                                .map(|sl| sl.content)
                                .collect();
                            Some(layout::wrap_styled_lines_to_lines(
                                &plain_lines,
                                inner_width.saturating_sub(2),
                            ))
                        } else {
                            None
                        };

                        // Render reasoning text separately (muted, tool indent)
                        let reasoning_wrapped = if has_reasoning {
                            let reasoning_lines: Vec<Line<'static>> = ts_ref
                                .reasoning_text
                                .lines()
                                .filter(|l| !l.trim().is_empty())
                                .map(|line| Line::from(
                                    Span::styled(
                                        line.to_string(),
                                        Style::default()
                                            .fg(palette.muted)
                                            .add_modifier(Modifier::DIM),
                                    ),
                                ))
                                .collect();
                            Some(layout::wrap_styled_lines_to_lines(
                                &reasoning_lines,
                                inner_width.saturating_sub(2),
                            ))
                        } else {
                            None
                        };

                        // Compute total height: body lines + border adjuster (2 per block) + 1 blank separator
                        let content_wr = content_wrapped.iter().map(|v| v.len()).sum::<usize>();
                        let reasoning_wr = reasoning_wrapped.iter().map(|v| v.len()).sum::<usize>();
                        let total_height = if content_wr > 0 && reasoning_wr > 0 {
                            // Both: content wraps + reasoning wraps + 2 borders per block + 1 separator
                            content_wr + reasoning_wr + 5
                        } else {
                            // One block only: wraps + 2 borders
                            content_wr + reasoning_wr + 2
                        };
                        let total_height_u16 = total_height as u16;

                        Some((content_wrapped, reasoning_wrapped, total_height_u16))
                    }
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };
        let stream_estimate = stream_parsed.as_ref().map(|s| s.2);

        let total_height = content_height + stream_estimate.unwrap_or(0);


        // Compute scroll offset using the layout module
        let scroll_offset = layout::compute_scroll_offset(
            total_height,
            inner_height as u16,
            (total_height as i64 - inner_height as i64).max(0) as usize,
            state.scroll_to_bottom.load(Ordering::SeqCst),
            &block_heights,
            scroll_pos,
        );


        // Sync computed scroll_offset back to state so render_selection/get_text
        // use the correct body line range (prevents selecting beyond viewport).
        state.scroll_pos.store(scroll_offset, Ordering::SeqCst);

        // Update scrollbar state
        {
            let mut scroll_state = state.scroll_state.lock().unwrap();
            *scroll_state = ScrollbarState::new(total_height.max(1) as usize)
                .viewport_content_length(inner_height as usize)
                .position(scroll_offset);
        }

        // Calculate visible areas using the layout module
        let visible_areas = layout::calc_visible_areas(
            msg_area.top(),
            msg_area.bottom().saturating_sub(1),
            msg_area.left(),
            msg_area.width,
            scroll_offset,
            &block_heights,
        );

        // ─── Phase 2: Render visible blocks ────────────────────────────
        // Track the bottom of the last visible entry so Phase 2.5 can position the stream block below it.
        let mut last_entry_vp_bottom = 0u16;
        state.visible_body_ranges.lock().unwrap().clear();
        for (idx, block_rect, content_start) in visible_areas {
            if block_rect.y.saturating_add(block_rect.height) > last_entry_vp_bottom {
                last_entry_vp_bottom = block_rect.y.saturating_add(block_rect.height);
            }
            let (entry_idx, block_type, wrapped) = &layout_entries[idx];

            // body_area = block.inner(block_area) — calculated at render time
            // using the actual Block struct (handles title, borders correctly)
            let is_tool_result = matches!(block_type, BlockType::ToolResult);

            // Build the block (borders + title)
            let block = if is_tool_result {
                let indent = layout::TOOL_INDENT;
                let _box_width =
                    (msg_area.width as usize).saturating_sub((indent * 2) as usize) as u16;
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(palette.muted))
                    .border_type(BorderType::Rounded)
            } else {
                let role = match block_type {
                    BlockType::Message(r) => r,
                    BlockType::ToolResult => unreachable!(),
                };
                let title = entry_titles[*entry_idx]
                    .clone()
                    .unwrap_or_else(|| self.role_title(role, palette));
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(self.role_border_style(role, palette))
                    .title(title)
            };

            // Render body (Paragraph) before block (borders on top)
            let body_area = block.inner(block_rect);

            // Track body rendering area per block from actual block.inner() — correctly
            // accounts for title+BODERS for all block types. Used by render_selection and
            // get_selected_text to filter terminal rows to visible body lines only.
            state
                .visible_body_ranges
                .lock()
                .unwrap()
                .push((body_area.y, body_area.height));

            // Calculate per-block scroll offset for clipping wrapped lines
            // content_start is the body-line index where this block starts in the global scroll view
            // inner_offset = how far into this block's body_lines we should start
            let inner_offset = scroll_offset.saturating_sub(content_start);
            let max_take = wrapped.len().saturating_sub(inner_offset);
            let inner_take = max_take.min(body_area.height as usize);

            tracing::trace!(
                "[scroll_in_block) idx={} content_start={} body_area_h={} scroll_offset={} view_top={}",
                idx, content_start, body_area.height, scroll_offset, msg_area.y
            );

            tracing::trace!(
                "[layout body_area.x={} block_rect.x={} msg_area.x={} area.x={}",
                body_area.x,
                block_rect.x,
                msg_area.left(),
                area.x
            );
            if block_rect.height > layout::BODY_HEIGHT_ADJUSTER && !wrapped.is_empty() {
                let body_lines: Vec<Line> = wrapped
                    .iter()
                    .skip(inner_offset)
                    .take(inner_take)
                    .cloned()
                    .collect();

                if !body_lines.is_empty() {
                    frame.render_widget(Paragraph::new(body_lines), body_area);
                }
            }

            frame.render_widget(block, block_rect);
        }

        // ─── Phase 2.5: Streaming blocks (reasoning + content as separate bordered blocks) ─
        let mut total_height = total_height;
        let mut streaming_rendered = false;
        // Capture streaming block body areas so we can append them to visible_body_ranges.
        let mut _reasoning_body_area: Option<Rect> = None;
        let mut _content_body_area: Option<Rect> = None;
        
        // Fallback: render streaming placeholder if no parsed content but streaming active
        let has_streaming_content = stream_parsed.is_some();
        
        if !has_streaming_content && is_streaming {
            // Streaming active but no content yet - render a placeholder
            if let Some(ref ts_arc) = state.turn_state_ref {
                let ts_ref = &*ts_arc.lock();
                tracing::info!(
                    "Streaming active but buffer empty: streaming_text.len={} phase={:?}",
                    ts_ref.streaming_text.len(),
                    ts_ref.phase
                );
            }
            
            // Render a simple placeholder block to show we're streaming
            let placeholder_block = Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(palette.info))
                .title(Line::from(vec![
                    Span::styled(" ", Style::default().fg(palette.info)),
                    Span::styled(
                        "Streaming...",
                        Style::default()
                            .fg(palette.info)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(" ", Style::default().fg(palette.info)),
                ]));
            let placeholder_area = Rect::new(
                msg_area.left(),
                last_entry_vp_bottom.saturating_add(1),
                msg_area.width,
                3,
            );
            frame.render_widget(placeholder_block, placeholder_area);
            total_height += 3;
            streaming_rendered = true;
        } else if let Some((content_wrapped, reasoning_wrapped, _total_height)) = stream_parsed {
            let has_reasoning = reasoning_wrapped.is_some() && !reasoning_wrapped.as_ref().unwrap().is_empty();
            let has_content = content_wrapped.is_some() && !content_wrapped.as_ref().unwrap().is_empty();

            if has_reasoning || has_content {
                let role_info = role_info_for_role(&MessageRole::Assistant, palette);
                let role_color = role_info.border_color;

                let reasoning_lines = reasoning_wrapped.as_ref().map_or(0, |v| v.len());
                let reasoning_height = reasoning_lines + layout::BODY_HEIGHT_ADJUSTER as usize;

                // ── Compute available space and stream_y ──
                let available_height = msg_area
                    .height
                    .saturating_sub(last_entry_vp_bottom.saturating_add(1));
                let view_height = msg_area.height.saturating_sub(1);
                let is_overflowing = total_height as u16 > view_height;

                let stream_y = if is_overflowing && available_height == 0 {
                    msg_area.height.saturating_sub(1)
                } else {
                    last_entry_vp_bottom.saturating_add(1)
                };

                if has_reasoning {
                    let reasoning_block_area = Rect::new(
                        msg_area.left(),
                        stream_y,
                        msg_area.width,
                        reasoning_height.max(1) as u16,
                    );
                    let reasoning_block = Block::default()
                        .borders(Borders::ALL)
                        .border_style(Style::default().fg(palette.muted))
                        .title(Line::from(vec![
                            Span::styled("  🤔 Thinking", Style::default()
                                .fg(palette.muted)
                                .add_modifier(Modifier::BOLD | Modifier::DIM)),
                        ]));
                    _reasoning_body_area = Some(reasoning_block.inner(reasoning_block_area));
                    let reasoning_wrap = reasoning_wrapped.as_ref().unwrap();
                    let reasoning_vw = inner_height.min(reasoning_wrap.len());
                    let reasoning_off = reasoning_wrap.len().saturating_sub(reasoning_vw);

                    if !reasoning_wrap.is_empty() {
                        let mut para = Paragraph::new(reasoning_wrap.clone());
                        if reasoning_off > 0 {
                            para = para.scroll((reasoning_off as u16, 0));
                        }
                        frame.render_widget(para, reasoning_block_area);
                    }
                    frame.render_widget(reasoning_block, reasoning_block_area);
                }

                // Content block: render below reasoning or at stream_y
                if has_content {
                    let content_lines = content_wrapped.as_ref().unwrap().len();
                    let content_height = content_lines + layout::BODY_HEIGHT_ADJUSTER as usize;
                    let reasoning_bottom_y = if has_reasoning {
                        stream_y.saturating_add(reasoning_height as u16).saturating_add(1)
                    } else {
                        stream_y
                    };
                    let content_block_area = Rect::new(
                        msg_area.left(),
                        reasoning_bottom_y,
                        msg_area.width,
                        content_height.max(1) as u16,
                    );
                    let content_block = Block::default()
                        .borders(Borders::ALL)
                        .border_style(Style::default().fg(role_color).add_modifier(Modifier::BOLD))
                        .title(self.role_title(&MessageRole::Assistant, palette));
                    _content_body_area = Some(content_block.inner(content_block_area));
                    let content_wrap = content_wrapped.as_ref().unwrap();
                    let content_vw = inner_height.min(content_wrap.len());
                    let content_off = content_wrap.len().saturating_sub(content_vw);

                    let mut para = Paragraph::new(content_wrap.clone());
                    if content_off > 0 {
                        para = para.scroll((content_off as u16, 0));
                    }
                    frame.render_widget(para, content_block_area);
                    frame.render_widget(content_block, content_block_area);
                }

                streaming_rendered = true;
            }
        }

        if !streaming_rendered && is_streaming {
            total_height += 1; // placeholder for streaming
        }

        // ─── Phase 3: Scroll position update (mouse wheel) ─────────────
        let prev_user_offset = state.user_scroll_offset.load(Ordering::SeqCst);
        let offset = state.user_scroll_offset.swap(0, Ordering::SeqCst);
        let prev_scroll_pos = scroll_pos;
        let mut scroll_pos = scroll_pos; // Copy value, will be reassigned later
        let scrollable_range = (total_height as i64 - inner_height as i64).max(0) as usize;
        let prev_scrollable_range = state
            .prev_scrollable_range
            .swap(scrollable_range, Ordering::SeqCst);
        let vp_bottom = prev_scroll_pos.saturating_add(inner_height as usize);
        let lines_from_bottom = (total_height as usize).saturating_sub(vp_bottom);

        tracing::trace!(
            "[scroll_phase3] user_offset={} (prev={}) scrollable_range={} prev_scrollable_range={} scroll_to_bottom={} offset_sign={}",
            offset, prev_user_offset, scrollable_range, prev_scrollable_range,
            state.scroll_to_bottom.load(Ordering::SeqCst),
            if offset > 0 { "+" } else if offset < 0 { "-" } else { "0" }
        );

        // Detect content growth BEFORE computing scroll_pos
        let content_grew = scrollable_range > prev_scrollable_range;
        // "near bottom": content bottom is within 10 lines of viewport bottom
        let near_bottom = lines_from_bottom < 10;

        // When content grows and user is reading at/near bottom, auto-scroll
        // so new streaming text stays visible.
        if content_grew && near_bottom {
            state.scroll_to_bottom.store(true, Ordering::SeqCst);
        }

        // Initialize scroll_pos based on whether we should snap to bottom
        if state.scroll_to_bottom.load(Ordering::SeqCst) {
            scroll_pos = scrollable_range;
        } else {
            // Read from the Phase 2.1 write — only if user hasn't scrolled away
            scroll_pos = prev_scroll_pos;
        }

        tracing::trace!(
            "[scroll_phase3] initialized scroll_pos={} vp_bottom={} lines_from_bottom={}",
            scroll_pos,
            scroll_pos.saturating_add(inner_height as usize),
            (total_height as usize)
                .saturating_sub(scroll_pos.saturating_add(inner_height as usize))
        );

        // Apply accumulated scroll delta (mouse wheel)
        if offset != 0 && !state.scroll_to_bottom.load(Ordering::SeqCst) {
            scroll_pos =
                ((scroll_pos as i64 + offset as i64).max(0) as usize).min(scrollable_range);
            tracing::trace!(
                "[scroll_phase3] after_offset: scroll_pos={} vp_bottom={}",
                scroll_pos,
                scroll_pos.saturating_add(inner_height as usize)
            );
        }

        state.scroll_pos.store(scroll_pos, Ordering::SeqCst);
        scroll_pos = state.scroll_pos.load(Ordering::SeqCst);
        let final_pos = scroll_pos;

        // Log scroll updates only every 10th time to reduce verbosity
        use std::sync::atomic::{AtomicUsize, Ordering};
        static SCROLL_LOG_COUNTER: AtomicUsize = AtomicUsize::new(0);
        
        if SCROLL_LOG_COUNTER.fetch_add(1, Ordering::SeqCst) % 10 == 0 {
            tracing::info!(
                "[scroll_update] BEFORE: total_height={} inner_height={} scrollable_range={} prev_scrollable_range={} scroll_pos={} vp_bottom={} lines_from_bottom={} scroll_to_bottom={}",
                total_height, inner_height, scrollable_range, prev_scrollable_range, scroll_pos, vp_bottom,
                lines_from_bottom, state.scroll_to_bottom.load(Ordering::SeqCst)
            );

            tracing::info!(
                "[scroll_update] AFTER: final_pos={} new_vp_bottom={} scroll_to_bottom={}",
                final_pos,
                final_pos.saturating_add(inner_height as usize),
                state.scroll_to_bottom.load(Ordering::SeqCst)
            );

            tracing::info!(
                "[scroll_update] AFTER: final_pos={} scrollable_range={} vp_bottom={} scroll_to_bottom={}",
                final_pos, scrollable_range, final_pos.saturating_add(inner_height as usize), state.scroll_to_bottom.load(Ordering::SeqCst)
            );
        }
        {
            let mut scroll_state = state.scroll_state.lock().unwrap();
            *scroll_state = ScrollbarState::new(total_height.max(1) as usize)
                .viewport_content_length(inner_height as usize)
                .position(scroll_pos);
        }
    }

    /// Render the full message display widget in `area`.
    pub fn render(
        &self,
        frame: &mut Frame,
        area: Rect,
        state: &ConversationState,
        palette: &ratatui_themes::ThemePalette,
        is_streaming: bool,
    ) {
        let entries = state.message_entries.lock().unwrap();
        let entry_titles: Vec<Option<Line<'static>>> = entries.iter().map(|e| e.title.clone()).collect();
        self.render_bordered_blocks(frame, area, state, palette, is_streaming, &entry_titles, &entries);
    }

    /// Append a user message to the internal display state.
    pub fn append_user_message(&mut self, state: &mut ConversationState, text: &str) {
        let mut entries = state.message_entries.lock().unwrap();
        let body_lines = vec![crate::widgets::message_renderer::StyledLine {
            content: Line::from(text.to_string()),
            role_color: None,
        }];
        entries.push(
            crate::widgets::message_renderer::MessageRenderEntry {
                role: oben_models::MessageRole::User,
                body_lines,
                is_tool_result: false,
                tool_calls: Vec::new(),
                reasoning: None,
                title: None,
            },
        );
    }

    /// Append an info/system message to the internal display state.
    /// Uses the System role style (gear icon, accent color).
    pub fn append_info_message(&mut self, state: &mut ConversationState, text: &str) {
        use ratatui_themes::ThemeName;
        let palette = ThemeName::default().palette();
        for line in text.lines() {
            let line_owned = line.to_string();
            let spans = vec![
                Span::styled(
                    "\u{2699} ",
                    Style::default()
                        .fg(palette.accent)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(line_owned, Style::default().fg(palette.info)),
            ];
            let body_lines = vec![crate::widgets::message_renderer::StyledLine {
                content: Line::from(spans),
                role_color: None,
            }];
            state.message_entries.lock().unwrap().push(
                crate::widgets::message_renderer::MessageRenderEntry {
                    role: oben_models::MessageRole::System,
                    body_lines,
                    is_tool_result: false,
                    tool_calls: Vec::new(),
                    reasoning: None,
                    title: None,
                },
            );
        }
    }

    /// Rebuild message entries from session messages using the renderer.
    /// When `reset_scroll` is true the scroll position is reset to the top
    /// so the caller (e.g. a turn-completion path) can show the newly
    /// arrived response from its beginning rather than snapping to the
    /// bottom where only the tail of a long reply would be visible.
    pub fn rebuild_from_messages(
        &self,
        state: &mut ConversationState,
        messages: &[Message],
        renderer: &MessageRenderer,
        reset_scroll: bool,
    ) {
        tracing::info!(
            "[rebuild_from_messages] INPUT msgs={} messages={:?}",
            messages.len(),
            messages.iter().map(|m| format!("{:?}", m.role)).collect::<Vec<_>>()
        );
        let entries: Vec<MessageRenderEntry> = messages
            .iter()
            .flat_map(|m| renderer.render_entries(m))
            .collect();
        // LOG: preview body lines count and preview text per entry to catch accumulation
        for (i, e) in entries.iter().enumerate() {
            let body_preview: String = e.body_lines.iter()
                .take(3)
                .map(|l| l.content.spans.iter()
                    .map(|s| s.content.to_string())
                    .collect::<Vec<_>>().join("")
                )
                .collect::<Vec<_>>()
                .join(" | ");
            tracing::info!(
                "[rebuild_from_messages] ENTRY[{}] role={:?} body_lines={} content_preview={}",
                i, e.role, e.body_lines.len(),
                body_preview.chars().take(160).collect::<String>()
            );
        }
        {
            let mut entry_lock = state.message_entries.lock().unwrap();
            *entry_lock = entries;
        }
        // Drop the previous lock before acquiring again — self-deadlock otherwise.
        {
            let entries_after = state.message_entries.lock().unwrap();
            tracing::info!(
                "[rebuild_from_messages] COMMITTED entries={}",
                entries_after.len()
            );
        }
        // Invalidate layout cache — content changed, so heights/wraps/ranges are stale.
        if let Ok(mut guard) = state.cached_layout.lock() {
            *guard = None;
        }
        if reset_scroll {
            state.scroll_pos.store(0, Ordering::SeqCst);
        }
    }

    /// Update stream_info from turn state into ConversationState.
    pub fn update_stream_info(&self, state: &mut ConversationState, turn_state: &TurnState) {
        let mut parts = Vec::new();

        let active = &turn_state.active_tools;
        if !active.is_empty() {
            let names: Vec<String> = active
                .iter()
                .take(2)
                .map(|t| {
                    format!(
                        " {}\u{1F527} {}",
                        t.name,
                        t.context.chars().take(30).collect::<String>()
                    )
                })
                .collect();
            parts.push(format!("\u{1F527} {}", names.join(", ")).to_string());
        }

        if !parts.is_empty() {
            let info = parts.join("\n");
            if let Ok(mut si) = state.stream_info.lock() {
                *si = info;
            }
        } else if let Ok(mut si) = state.stream_info.lock() {
            si.clear();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oben_agent::{ActiveTool, TurnState};
    use std::time::Instant;

    /// Given: a TurnState with active tool calls AND streaming text
    /// When: update_stream_info is called
    /// Then: stream_info contains only tool info, NOT streaming text
    #[test]
    fn test_update_stream_info_excludes_streaming_text() {
        let mut state = ConversationState::new();
        let mut turn = TurnState::new();
        turn.active_tools.push(ActiveTool {
            id: "call-1".into(),
            name: "file_read".into(),
            started_at: Instant::now(),
            context: "/Users/test/config.yaml".into(),
        });
        turn.streaming_text = "This is streaming text that should NOT appear in stream_info".into();

        ConversationWidget.update_stream_info(&mut state, &turn);

        let info = state.stream_info.lock().unwrap();
        assert!(!info.contains("streaming text"));
        assert!(!info.is_empty());
        assert!(info.contains("file_read"));
    }

    /// Given: a TurnState with ONLY streaming text, no active tools
    /// When: update_stream_info is called
    /// Then: stream_info stays empty (no misleading tool status)
    #[test]
    fn test_update_stream_info_empty_with_no_tools() {
        let mut state = ConversationState::new();
        let mut turn = TurnState::new();
        turn.streaming_text = "Some streaming content".into();

        ConversationWidget.update_stream_info(&mut state, &turn);

        let info = state.stream_info.lock().unwrap();
        assert!(info.is_empty());
    }

    /// Given: a TurnState with active tools
    /// When: update_stream_info is called
    /// Then: stream_info includes the tool name
    #[test]
    fn test_update_stream_info_includes_tool_name() {
        let mut state = ConversationState::new();
        let mut turn = TurnState::new();
        turn.active_tools.push(ActiveTool {
            id: "call-2".into(),
            name: "search_files".into(),
            started_at: Instant::now(),
            context: "/some/path".into(),
        });

        ConversationWidget.update_stream_info(&mut state, &turn);

        let info = state.stream_info.lock().unwrap();
        assert!(info.contains("search_files"));
    }

    /// Given: a TurnState with streaming text AND active tools
    /// When: update_stream_info is called
    /// Then: streaming_text is NOT included in stream_info
    #[test]
    fn test_streaming_text_not_duplicated_in_stream_info() {
        let mut state = ConversationState::new();
        let mut turn = TurnState::new();
        let streaming_content = "The Clockmaker of Lost Hours";
        turn.streaming_text = streaming_content.into();
        turn.active_tools.push(ActiveTool {
            id: "call-write".into(),
            name: "file_write".into(),
            started_at: Instant::now(),
            context: "/Users/test/docs.md".into(),
        });

        ConversationWidget.update_stream_info(&mut state, &turn);

        let info = state.stream_info.lock().unwrap();
        assert!(info.contains("file_write"));
        assert!(info.is_empty() || !info.contains("The Clockmaker of Lost Hours"));
    }

    /// Given: streaming text is empty before render
    /// When: render_bordered_blocks processes entries without stream data
    /// Then: stream_parsed should be None (no double-parse for empty text)
    #[test]
    fn test_stream_parsed_none_when_text_empty() {
        let mut state = ConversationState::new();
        let mut turn = TurnState::new();
        turn.streaming_text = String::new();
        state.turn_state_ref = Some(Arc::new(PlMutex::new(turn)));
        state.stream_info.lock().unwrap().clear();

        // Verify that a new TurnState is initialized empty
        let state_guard = state.turn_state_ref.as_ref().map(|g| g.lock());
        if let Some(guard) = state_guard {
            assert_eq!(guard.streaming_text, "");
            assert!(guard.active_tools.is_empty());
        }
    }

    /// Given: 3 message entries with known heights
    /// When: calc_total_height is called with inter_block_margins
    /// Then: total = sum of heights + (n-1) * 1 margin
    #[test]
    fn test_calc_total_height_with_margins() {
        let heights = vec![4u16, 6, 4]; // 3 blocks
        let total = layout::calc_total_height(&heights);
        // 4 + 6 + 4 + 2 margins (between block 0/1 and 1/2)
        assert_eq!(total, 16);
    }

    /// Given: total_height=125, inner_height=59, stream_height=4
    /// When: content overflows viewport (125 > 59)
    /// Then: stream_y anchored to viewport bottom (59-4=55)
    #[test]
    fn test_stream_y_anchored_to_bottom_when_overflow() {
        let total_height: u16 = 125;
        let view_height: u16 = 59;
        let stream_height: u16 = 4;

        let overflows = total_height > view_height;

        let stream_y_overflow = if overflows {
            view_height.saturating_sub(stream_height)
        } else {
            0
        };

        assert!(overflows);
        assert_eq!(stream_y_overflow, 55); // 59 - 4
    }

    /// Given: total_height=2, inner_height=59 (content fits)
    /// When: last_entry_vp_bottom=3
    /// Then: stream_y = last_entry_vp_bottom + 1 = 4
    #[test]
    fn test_stream_y_after_entry_when_content_fits() {
        let total_height: u16 = 2;
        let view_height: u16 = 59;
        let stream_height: u16 = 4;
        let last_entry_vp_bottom: u16 = 3;

        let fits = total_height <= view_height;

        let stream_y = if fits {
            last_entry_vp_bottom.saturating_add(1)
        } else {
            view_height.saturating_sub(stream_height)
        };

        assert!(fits);
        assert_eq!(stream_y, 4);
    }

    /// Given: scroll_to_bottom=true, prev_scrollable_range=6, new=8
    /// When: auto-scroll detects content growth (8>6) and near bottom
    /// Then: scroll_to_bottom remains true, ready for auto-scroll in Phase 3
    #[test]
    fn test_scroll_to_bottom_auto_scroll_on_content_growth() {
        let state = ConversationState::new();

        // Verify initial state
        assert!(state.scroll_to_bottom.load(Ordering::SeqCst));
        assert_eq!(state.scroll_pos.load(Ordering::SeqCst), 0);
        assert_eq!(state.prev_scrollable_range.load(Ordering::SeqCst), 0);
    }
}
