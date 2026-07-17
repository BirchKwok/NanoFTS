//! Text analyzer: NFKC normalization, multi-script tokenization, CJK n-grams.
//!
//! Ported from ApexFTS analyzer design for NanoFTS.

use ahash::AHashMap;
use std::borrow::Cow;
use unicode_normalization::UnicodeNormalization;
use unicode_segmentation::UnicodeSegmentation;

/// Analyzer configuration (subset used by tokenization).
#[derive(Clone, Copy, Debug)]
pub struct AnalyzerConfig {
    pub max_chinese_length: usize,
    pub min_term_length: usize,
}

impl Default for AnalyzerConfig {
    fn default() -> Self {
        Self {
            max_chinese_length: 3,
            min_term_length: 2,
        }
    }
}

/// Result of analyzing a document or query.
#[derive(Clone, Debug, Default)]
pub struct AnalyzedDocument {
    /// Sorted unique terms (stable order for indexing).
    pub terms: Vec<String>,
    /// One-based indexes into `terms`; zero is a field boundary.
    pub tokens: Vec<u32>,
}

pub fn normalize(text: &str) -> String {
    if text.is_ascii() {
        text.to_ascii_lowercase()
    } else {
        text.nfkc().flat_map(char::to_lowercase).collect()
    }
}

pub fn is_cjk(c: char) -> bool {
    matches!(
        c,
        '\u{3400}'..='\u{4dbf}'
            | '\u{4e00}'..='\u{9fff}'
            | '\u{f900}'..='\u{faff}'
            | '\u{3040}'..='\u{309f}'
            | '\u{30a0}'..='\u{30ff}'
            | '\u{31f0}'..='\u{31ff}'
            | '\u{ac00}'..='\u{d7af}'
    )
}

fn primary_tokens(normalized: &str, config: &AnalyzerConfig) -> Vec<String> {
    let mut tokens = Vec::new();
    for segment in normalized.split_word_bounds() {
        let mut non_cjk = String::new();
        let flush_non_cjk = |buffer: &mut String, output: &mut Vec<String>| {
            for word in buffer.unicode_words() {
                if word.chars().count() >= config.min_term_length {
                    output.push(word.to_string());
                }
            }
            buffer.clear();
        };
        for character in segment.chars() {
            if is_cjk(character) {
                flush_non_cjk(&mut non_cjk, &mut tokens);
                tokens.push(character.to_string());
            } else {
                non_cjk.push(character);
            }
        }
        flush_non_cjk(&mut non_cjk, &mut tokens);
    }
    tokens
}

struct AnalysisBuilder {
    terms: AHashMap<String, u32>,
    tokens: Vec<u32>,
    /// When false, skip recording positional tokens (boolean queries only need terms).
    record_positions: bool,
}

impl Default for AnalysisBuilder {
    fn default() -> Self {
        Self {
            terms: AHashMap::new(),
            tokens: Vec::new(),
            record_positions: true,
        }
    }
}

impl AnalysisBuilder {
    fn intern(&mut self, term: String) -> u32 {
        if let Some(&term_id) = self.terms.get(&term) {
            return term_id;
        }
        let term_id = self.terms.len() as u32 + 1;
        self.terms.insert(term, term_id);
        term_id
    }

    /// Like `intern`, but avoids allocating a `String` for terms that are already
    /// present in the dictionary (lookup works directly against the borrowed `&str`).
    fn intern_cow(&mut self, term: Cow<'_, str>) -> u32 {
        if let Some(&term_id) = self.terms.get(term.as_ref()) {
            return term_id;
        }
        let term_id = self.terms.len() as u32 + 1;
        self.terms.insert(term.into_owned(), term_id);
        term_id
    }

    /// ASCII-only tokenization that skips the whole-text lowercase allocation done by
    /// `normalize()`: words are located on the original byte slice (lowercasing doesn't
    /// change which bytes are alphanumeric/`_`, so split points are identical), and a
    /// lowercase copy of a word is only allocated when it actually contains uppercase
    /// bytes. Produces output identical to `add_text` for ASCII input.
    fn add_text_ascii_fast(&mut self, text: &str, config: &AnalyzerConfig) {
        debug_assert!(text.is_ascii());
        let bytes = text.as_bytes();
        let len = bytes.len();
        let mut start = 0usize;
        let mut idx = 0usize;
        while idx <= len {
            let is_word_byte =
                idx < len && (bytes[idx].is_ascii_alphanumeric() || bytes[idx] == b'_');
            if is_word_byte {
                idx += 1;
                continue;
            }
            if idx > start {
                let word = &text[start..idx];
                if word.len() >= config.min_term_length {
                    let lower: Cow<'_, str> = if word.bytes().any(|b| b.is_ascii_uppercase()) {
                        Cow::Owned(word.to_ascii_lowercase())
                    } else {
                        Cow::Borrowed(word)
                    };
                    let term_id = self.intern_cow(lower);
                    if self.record_positions {
                        self.tokens.push(term_id);
                    }
                }
            }
            start = idx + 1;
            idx += 1;
        }
    }

    fn add_text(&mut self, text: &str, config: &AnalyzerConfig) {
        let normalized = normalize(text);
        if normalized.is_ascii() {
            for word in normalized
                .split(|character: char| !character.is_ascii_alphanumeric() && character != '_')
            {
                if word.len() >= config.min_term_length {
                    let term_id = self.intern(word.to_string());
                    if self.record_positions {
                        self.tokens.push(term_id);
                    }
                }
            }
            return;
        }

        for token in primary_tokens(&normalized, config) {
            let term_id = self.intern(token);
            if self.record_positions {
                self.tokens.push(term_id);
            }
        }
        let chars: Vec<char> = normalized.chars().collect();
        let mut start = 0;
        while start < chars.len() {
            if !is_cjk(chars[start]) {
                start += 1;
                continue;
            }
            let mut end = start + 1;
            while end < chars.len() && is_cjk(chars[end]) {
                end += 1;
            }
            let run = &chars[start..end];
            let max_n = config.max_chinese_length.max(1).min(run.len());
            for n in 1..=max_n {
                for offset in 0..=run.len() - n {
                    self.intern(run[offset..offset + n].iter().collect());
                }
            }
            start = end;
        }
    }

    fn finish(self) -> AnalyzedDocument {
        let mut entries: Vec<(String, u32)> = self.terms.into_iter().collect();
        entries.sort_unstable_by(|left, right| left.0.cmp(&right.0));
        if !self.record_positions {
            return AnalyzedDocument {
                terms: entries.into_iter().map(|(term, _)| term).collect(),
                tokens: Vec::new(),
            };
        }
        let mut remap = vec![0; entries.len() + 1];
        let mut terms = Vec::with_capacity(entries.len());
        for (new_index, (term, old_id)) in entries.into_iter().enumerate() {
            remap[old_id as usize] = new_index as u32 + 1;
            terms.push(term);
        }
        let tokens = self
            .tokens
            .into_iter()
            .map(|term_id| {
                if term_id == 0 {
                    0
                } else {
                    remap[term_id as usize]
                }
            })
            .collect();
        AnalyzedDocument { terms, tokens }
    }
}

/// Analyze a single text value.
pub fn analyze_document(text: &str, config: &AnalyzerConfig) -> AnalyzedDocument {
    let mut builder = AnalysisBuilder::default();
    builder.add_text(text, config);
    builder.finish()
}

/// Analyze ASCII text without the Unicode normalization allocations that
/// `analyze_document` always performs. Produces output identical to
/// `analyze_document` for the same input; falls back to it entirely for
/// non-ASCII text (which still needs NFKC + CJK n-gram handling).
///
/// Intended for the document ingest hot path, where the vast majority of
/// input is plain ASCII (e.g. English text, identifiers, log lines).
pub fn analyze_document_fast(text: &str, config: &AnalyzerConfig) -> AnalyzedDocument {
    if !text.is_ascii() {
        return analyze_document(text, config);
    }
    let mut builder = AnalysisBuilder::default();
    builder.add_text_ascii_fast(text, config);
    builder.finish()
}

/// Analyze a search query.
///
/// When `need_positions` is false (plain boolean AND), skips building/remapping
/// the positional token stream — terms alone are enough for posting lookup.
/// Phrase queries must pass `need_positions = true`.
pub fn analyze_query(text: &str, config: &AnalyzerConfig, need_positions: bool) -> AnalyzedDocument {
    let mut builder = AnalysisBuilder {
        record_positions: need_positions,
        ..AnalysisBuilder::default()
    };
    if text.is_ascii() {
        builder.add_text_ascii_fast(text, config);
    } else {
        builder.add_text(text, config);
    }
    builder.finish()
}

/// Analyze multiple fields with field-boundary tokens (0).
pub fn analyze_values<'a>(
    values: impl IntoIterator<Item = &'a str>,
    config: &AnalyzerConfig,
) -> AnalyzedDocument {
    analyze_fields(values, config, true)
}

/// Analyze one or more fields for indexing / querying.
///
/// When `need_positions` is false, skips positional token recording (boolean
/// indexing only needs the unique term set). Field-boundary tokens are only
/// emitted when positions are recorded (required for phrase queries).
pub fn analyze_fields<'a>(
    values: impl IntoIterator<Item = &'a str>,
    config: &AnalyzerConfig,
    need_positions: bool,
) -> AnalyzedDocument {
    let mut builder = AnalysisBuilder {
        record_positions: need_positions,
        ..AnalysisBuilder::default()
    };
    if need_positions {
        for value in values {
            let prior_tokens = builder.tokens.len();
            if prior_tokens > 0 {
                builder.tokens.push(0);
            }
            let boundary_index = builder.tokens.len();
            if value.is_ascii() {
                builder.add_text_ascii_fast(value, config);
            } else {
                builder.add_text(value, config);
            }
            if builder.tokens.len() == boundary_index && prior_tokens > 0 {
                builder.tokens.pop();
            }
        }
    } else {
        for value in values {
            if value.is_ascii() {
                builder.add_text_ascii_fast(value, config);
            } else {
                builder.add_text(value, config);
            }
        }
    }
    builder.finish()
}

/// Return unique terms only (for indexing helpers).
pub fn analyze(text: &str, config: &AnalyzerConfig) -> Vec<String> {
    analyze_document(text, config).terms
}

/// Levenshtein distance with early exit when exceeding `max_distance`.
pub fn levenshtein(left: &str, right: &str, max_distance: usize) -> Option<usize> {
    let left: Vec<char> = left.chars().collect();
    let right: Vec<char> = right.chars().collect();
    if left.len().abs_diff(right.len()) > max_distance {
        return None;
    }
    let mut previous: Vec<usize> = (0..=right.len()).collect();
    let mut current = vec![0; right.len() + 1];
    for (i, &a) in left.iter().enumerate() {
        current[0] = i + 1;
        let mut row_min = current[0];
        for (j, &b) in right.iter().enumerate() {
            current[j + 1] = (previous[j + 1] + 1)
                .min(current[j] + 1)
                .min(previous[j] + usize::from(a != b));
            row_min = row_min.min(current[j + 1]);
        }
        if row_min > max_distance {
            return None;
        }
        std::mem::swap(&mut previous, &mut current);
    }
    (previous[right.len()] <= max_distance).then_some(previous[right.len()])
}

/// Detect `"phrase query"` syntax.
pub fn phrase_query(query: &str) -> (&str, bool) {
    let query = query.trim();
    if query.len() >= 2 && query.starts_with('"') && query.ends_with('"') {
        (&query[1..query.len() - 1], true)
    } else {
        (query, false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ascii_and_cjk_terms() {
        let cfg = AnalyzerConfig::default();
        let analyzed = analyze_document("Hello World 中文测试", &cfg);
        assert!(analyzed.terms.iter().any(|t| t == "hello"));
        assert!(analyzed.terms.iter().any(|t| t == "world"));
        assert!(analyzed.terms.iter().any(|t| t.contains('中')));
    }

    #[test]
    fn field_boundaries() {
        let cfg = AnalyzerConfig::default();
        let analyzed = analyze_values(["alpha beta", "gamma"], &cfg);
        assert!(analyzed.tokens.contains(&0));
    }

    #[test]
    fn levenshtein_early_exit() {
        assert_eq!(levenshtein("kitten", "sitting", 3), Some(3));
        assert_eq!(levenshtein("kitten", "sitting", 2), None);
    }

    #[test]
    fn analyze_document_fast_matches_slow_path_for_ascii() {
        let cfg = AnalyzerConfig::default();
        let samples = [
            "",
            "hello",
            "Hello World",
            "HELLO hello Hello hELLo",
            "The Quick Brown Fox jumps OVER the lazy_dog 42 times",
            "a ab abc ABC AbC   multiple   spaces___and_underscores",
            "UPPER_CASE_ONLY UPPER_CASE_ONLY repeated repeated",
            "punctuation, should! split? on-non-alnum; chars.",
        ];
        for sample in samples {
            let slow = analyze_document(sample, &cfg);
            let fast = analyze_document_fast(sample, &cfg);
            assert_eq!(fast.terms, slow.terms, "terms mismatch for {sample:?}");
            assert_eq!(fast.tokens, slow.tokens, "tokens mismatch for {sample:?}");
        }
    }

    #[test]
    fn analyze_document_fast_delegates_for_non_ascii() {
        let cfg = AnalyzerConfig::default();
        let sample = "Hello 中文测试 Café";
        let slow = analyze_document(sample, &cfg);
        let fast = analyze_document_fast(sample, &cfg);
        assert_eq!(fast.terms, slow.terms);
        assert_eq!(fast.tokens, slow.tokens);
    }
}
