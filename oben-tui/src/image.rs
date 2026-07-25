//! Image path → base64 data URL → [`Message`] conversion.

use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use std::fs;
use std::path::Path;

use oben_models::{Message, MessageContent, MessagePart, MessageRole};

// ── Mime type constants using concat! to avoid Rust 2.0 /ident parse issues ──

const M_PNG: &str = concat!("image/", "png");
const M_JPEG: &str = concat!("image/", "jpeg");
const M_GIF: &str = concat!("image/", "gif");
const M_WEBP: &str = concat!("image/", "webp");
const M_SVG: &str = concat!("image/", "svg+xml");
const M_BMP: &str = concat!("image/", "bmp");
const M_TIFF: &str = concat!("image/", "tiff");
const M_AVIF: &str = concat!("image/", "avif");
const M_OCTET: &str = concat!("application/octet-", "stream");

/// Known image file extensions (with leading dot).
const IMAGE_EXTENSIONS: &[&str] = &[
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".bmp", ".tiff", ".tif", ".ico", ".avif",
];

/// The display icon: camera with flash.
const DISPLAY_ICON: &str = concat!("\u{1F5BC}");

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Detect MIME type from a file path extension.
pub fn detect_mime(path: &str) -> String {
    let ext = Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    match ext.as_str() {
        "jpg" | "jpeg" => M_JPEG.to_string(),
        "png" => M_PNG.to_string(),
        "gif" => M_GIF.to_string(),
        "webp" => M_WEBP.to_string(),
        "svg" => M_SVG.to_string(),
        "bmp" => M_BMP.to_string(),
        "tiff" | "tif" => M_TIFF.to_string(),
        "avif" => M_AVIF.to_string(),
        _ => M_OCTET.to_string(),
    }
}

/// Check whether `text` contains an image file path somewhere in it.
pub fn is_image_path(text: &str) -> bool {
    let text_lower: String = text.to_lowercase().chars().collect();
    for &ext in IMAGE_EXTENSIONS {
        let ext_chars: String = ext.to_lowercase().chars().collect();
        if text_lower.contains(&ext_chars) {
            let ext_pos = text_lower.find(&ext_chars).unwrap();
            // Verify there's a `/` before the extension (indicating it's a file path)
            if text_lower[..ext_pos].rfind('/').is_some() {
                return true;
            }
        }
    }
    false
}

// ── Public API ─────────────────────────────────────────────────────────────

/// Convert an image file path to a [`Message`] with base64-encoded image data.
pub fn path_to_image_message(path: &str, prompt_text: &str) -> Option<(Message, String)> {
    let path = path.trim();
    let bytes = fs::read(path).ok()?;
    let mime = detect_mime(path);
    let b64 = STANDARD.encode(&bytes);
    let data_url = format!("data:{mime};base64,{b64}");

    let display = if prompt_text.trim().is_empty() {
        format!("{}{}", DISPLAY_ICON, path)
    } else {
        format!("{}{} {}", DISPLAY_ICON, path, prompt_text)
    };

    let message = if prompt_text.trim().is_empty() {
        Message {
            role: MessageRole::User,
            content: MessageContent::Image {
                url: data_url,
                detail: None,
            },
            id: None,
            tool_call_ids: Vec::new(),
            tool_calls: None,
            reasoning: None,
            delegation_id: None,
            tool_error: false,
            include_in_prompt: true,
        }
    } else {
        Message {
            role: MessageRole::User,
            content: MessageContent::Parts(vec![
                MessagePart::Text(prompt_text.to_string()),
                MessagePart::Image {
                    url: data_url,
                    detail: None,
                },
            ]),
            id: None,
            tool_call_ids: Vec::new(),
            tool_calls: None,
            reasoning: None,
            delegation_id: None,
            tool_error: false,
            include_in_prompt: true,
        }
    };

    Some((message, display))
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_detect_mime_png() {
        assert_eq!(detect_mime("/photo.png"), M_PNG);
    }

    #[test]
    fn test_detect_mime_jpeg() {
        assert_eq!(detect_mime("/photo.jpg"), M_JPEG);
    }

    #[test]
    fn test_detect_mime_unknown() {
        assert_eq!(detect_mime("/file.xyz"), M_OCTET);
    }

    #[test]
    fn test_is_image_path_simple() {
        assert!(is_image_path("/Users/ellie/Pictures/photo.png"));
    }

    #[test]
    fn test_is_image_path_not_image() {
        assert!(!is_image_path("hello world"));
        assert!(!is_image_path("/script.py"));
    }

    #[test]
    fn test_is_image_path_with_trailing_text() {
        assert!(is_image_path("/path/photo.png 分析下这个图片"));
    }

    #[test]
    fn test_path_to_image_message_none() {
        assert!(path_to_image_message("/nonexistent/file.png", "").is_none());
    }

    #[test]
    fn test_path_to_image_message_with_prompt() {
        let path = std::env::temp_dir().join("oben_test_image.png");
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(&[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])
            .unwrap();
        drop(file);

        let prompt = "分析下这个图片";
        let Some((msg, _)) = path_to_image_message(&path.to_string_lossy(), prompt) else {
            panic!("expected Some result");
        };
        assert!(msg.content.to_text_ref().is_none());
        if let MessageContent::Parts(parts) = &msg.content {
            assert_eq!(parts.len(), 2);
            if let MessagePart::Text(t) = &parts[0] {
                assert_eq!(t, prompt);
            } else {
                panic!("expected Part::Text");
            }
        } else {
            panic!("expected Parts variant for text+image");
        }
        let _ = std::fs::remove_file(&path);
    }
}
