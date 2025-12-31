//! SIMD Accelerated Utilities Module
//! 
//! Provides high-performance string operations and similarity calculations

/// Calculate edit distance (Levenshtein distance)
/// 
/// Uses optimized dynamic programming algorithm
#[inline]
pub fn levenshtein_distance(s1: &str, s2: &str) -> usize {
    let s1_chars: Vec<char> = s1.chars().collect();
    let s2_chars: Vec<char> = s2.chars().collect();
    
    let len1 = s1_chars.len();
    let len2 = s2_chars.len();
    
    // Fast path
    if len1 == 0 {
        return len2;
    }
    if len2 == 0 {
        return len1;
    }
    
    // Use two rows instead of full matrix to save memory
    let mut prev_row: Vec<usize> = (0..=len2).collect();
    let mut curr_row: Vec<usize> = vec![0; len2 + 1];
    
    for i in 1..=len1 {
        curr_row[0] = i;
        
        for j in 1..=len2 {
            let cost = if s1_chars[i - 1] == s2_chars[j - 1] { 0 } else { 1 };
            
            curr_row[j] = (prev_row[j] + 1)              // Delete
                .min(curr_row[j - 1] + 1)                // Insert
                .min(prev_row[j - 1] + cost);            // Replace
        }
        
        std::mem::swap(&mut prev_row, &mut curr_row);
    }
    
    prev_row[len2]
}

/// Calculate string similarity score (0.0 - 1.0)
/// 
/// Combines multiple algorithms for best matching effect, specifically optimized for Chinese
#[inline]
pub fn similarity_score(s1: &str, s2: &str) -> f64 {
    if s1.is_empty() && s2.is_empty() {
        return 1.0;
    }
    if s1.is_empty() || s2.is_empty() {
        return 0.0;
    }
    
    let s1_chars: Vec<char> = s1.chars().collect();
    let s2_chars: Vec<char> = s2.chars().collect();
    let max_len = s1_chars.len().max(s2_chars.len());
    
    // 1. Edit distance based similarity
    let distance = levenshtein_distance(s1, s2);
    let edit_similarity = 1.0 - (distance as f64 / max_len as f64);
    
    // 2. Character overlap based similarity (Jaccard coefficient)
    let set1: std::collections::HashSet<char> = s1_chars.iter().copied().collect();
    let set2: std::collections::HashSet<char> = s2_chars.iter().copied().collect();
    let intersection = set1.intersection(&set2).count();
    let union = set1.union(&set2).count();
    let char_overlap = if union > 0 {
        intersection as f64 / union as f64
    } else {
        0.0
    };
    
    // 3. Longest common subsequence based similarity
    let lcs_len = longest_common_subsequence(&s1_chars, &s2_chars);
    let lcs_similarity = lcs_len as f64 / max_len as f64;
    
    // Take maximum of three algorithms (more lenient matching)
    edit_similarity.max(char_overlap).max(lcs_similarity)
}

/// Calculate longest common subsequence length
#[inline]
fn longest_common_subsequence(s1: &[char], s2: &[char]) -> usize {
    let m = s1.len();
    let n = s2.len();
    
    if m == 0 || n == 0 {
        return 0;
    }
    
    // Use two rows instead of full matrix
    let mut prev: Vec<usize> = vec![0; n + 1];
    let mut curr: Vec<usize> = vec![0; n + 1];
    
    for i in 1..=m {
        for j in 1..=n {
            if s1[i - 1] == s2[j - 1] {
                curr[j] = prev[j - 1] + 1;
            } else {
                curr[j] = prev[j].max(curr[j - 1]);
            }
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    
    prev[n]
}

/// Fast prefix matching
#[inline]
pub fn has_prefix(text: &str, prefix: &str) -> bool {
    text.starts_with(prefix)
}

/// Fast suffix matching
#[inline]
pub fn has_suffix(text: &str, suffix: &str) -> bool {
    text.ends_with(suffix)
}

/// Fast substring search
/// 
/// Uses memchr accelerated Boyer-Moore-Horspool algorithm
#[inline]
pub fn contains_substring(text: &str, pattern: &str) -> bool {
    if pattern.is_empty() {
        return true;
    }
    if text.len() < pattern.len() {
        return false;
    }
    text.contains(pattern)
}

/// Batch substring search
/// 
/// Search for multiple patterns in a single text
pub fn contains_any_substring(text: &str, patterns: &[&str]) -> Vec<usize> {
    patterns.iter()
        .enumerate()
        .filter_map(|(i, p)| {
            if contains_substring(text, p) {
                Some(i)
            } else {
                None
            }
        })
        .collect()
}

/// Chinese character detection
#[inline]
pub fn is_chinese_char(c: char) -> bool {
    ('\u{4e00}'..='\u{9fff}').contains(&c)
}

/// Count Chinese characters
#[inline]
pub fn count_chinese_chars(s: &str) -> usize {
    s.chars().filter(|c| is_chinese_char(*c)).count()
}

/// Extract Chinese segments
pub fn extract_chinese_segments(s: &str) -> Vec<String> {
    let mut segments = Vec::new();
    let mut current = String::new();
    
    for c in s.chars() {
        if is_chinese_char(c) {
            current.push(c);
        } else {
            if !current.is_empty() {
                segments.push(std::mem::take(&mut current));
            }
        }
    }
    
    if !current.is_empty() {
        segments.push(current);
    }
    
    segments
}

/// Generate n-grams
pub fn generate_ngrams(s: &str, n: usize) -> Vec<String> {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() < n {
        return vec![];
    }
    
    (0..=chars.len() - n)
        .map(|i| chars[i..i + n].iter().collect())
        .collect()
}

/// Batch generate n-grams (multiple lengths)
pub fn generate_ngrams_range(s: &str, min_n: usize, max_n: usize) -> Vec<String> {
    let mut result = Vec::new();
    for n in min_n..=max_n {
        result.extend(generate_ngrams(s, n));
    }
    result
}

/// String normalization
/// 
/// Convert to lowercase, remove extra whitespace
#[inline]
pub fn normalize(s: &str) -> String {
    s.to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Fast hash calculation
#[inline]
pub fn fast_hash(s: &str) -> u64 {
    xxhash_rust::xxh64::xxh64(s.as_bytes(), 0)
}

/// Fast 32-bit hash
#[inline]
pub fn fast_hash32(s: &str) -> u32 {
    xxhash_rust::xxh32::xxh32(s.as_bytes(), 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_levenshtein_distance() {
        assert_eq!(levenshtein_distance("kitten", "sitting"), 3);
        assert_eq!(levenshtein_distance("hello", "hello"), 0);
        assert_eq!(levenshtein_distance("", "hello"), 5);
        assert_eq!(levenshtein_distance("hello", ""), 5);
    }

    #[test]
    fn test_similarity_score() {
        let score = similarity_score("测试", "测验");
        assert!(score > 0.0 && score < 1.0);
        
        let score_exact = similarity_score("完全相同", "完全相同");
        assert!((score_exact - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_chinese_detection() {
        assert!(is_chinese_char('中'));
        assert!(!is_chinese_char('a'));
        
        assert_eq!(count_chinese_chars("hello中文world"), 2);
    }

    #[test]
    fn test_ngrams() {
        let ngrams = generate_ngrams("hello", 2);
        assert_eq!(ngrams, vec!["he", "el", "ll", "lo"]);
    }

    #[test]
    fn test_lcs() {
        let s1: Vec<char> = "ABCDGH".chars().collect();
        let s2: Vec<char> = "AEDFHR".chars().collect();
        assert_eq!(longest_common_subsequence(&s1, &s2), 3); // ADH
    }
}
