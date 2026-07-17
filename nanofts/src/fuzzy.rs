//! Fuzzy-search candidate pruning.
//!
//! Before running Levenshtein, shrink the candidate set with:
//! 1. Length band: `|len(term) - len(query)| <= max_distance`
//! 2. Q-gram filter (Ukkonen): query and candidate must share at least one
//!    contiguous substring of length `floor(len(query) / (max_distance + 1))`.

use ahash::AHashSet;

/// Return references to terms that can possibly be within `max_distance` edits of `query`.
pub fn prune_fuzzy_candidates<'a>(
    terms: &'a [String],
    query: &str,
    max_distance: usize,
) -> Vec<&'a str> {
    let q_chars: Vec<char> = query.chars().collect();
    let qlen = q_chars.len();
    if qlen == 0 {
        return Vec::new();
    }

    let min_len = qlen.saturating_sub(max_distance);
    let max_len = qlen + max_distance;
    let gram_len = qlen / (max_distance + 1);

    let query_grams: AHashSet<String> = if gram_len > 0 {
        (0..=qlen - gram_len)
            .map(|i| q_chars[i..i + gram_len].iter().collect())
            .collect()
    } else {
        AHashSet::new()
    };

    let mut out = Vec::new();
    for term in terms {
        let tlen = term.chars().count();
        if tlen < min_len || tlen > max_len {
            continue;
        }
        if gram_len == 0 {
            out.push(term.as_str());
            continue;
        }
        if shares_query_gram(term, gram_len, &query_grams) {
            out.push(term.as_str());
        }
    }
    out
}

fn shares_query_gram(term: &str, gram_len: usize, query_grams: &AHashSet<String>) -> bool {
    let t_chars: Vec<char> = term.chars().collect();
    if t_chars.len() < gram_len {
        return false;
    }
    (0..=t_chars.len() - gram_len).any(|i| {
        let gram: String = t_chars[i..i + gram_len].iter().collect();
        query_grams.contains(&gram)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn length_band_excludes_far_terms() {
        let terms = vec![
            "cat".into(),
            "cats".into(),
            "category".into(),
            "dog".into(),
        ];
        let pruned = prune_fuzzy_candidates(&terms, "cat", 1);
        assert!(pruned.contains(&"cat"));
        assert!(pruned.contains(&"cats"));
        // "dog" is in the length band but shares no 1-gram with "cat"
        assert!(!pruned.contains(&"dog"));
        assert!(!pruned.contains(&"category"));
    }

    #[test]
    fn qgram_keeps_close_variants() {
        let terms = vec![
            "kitten".into(),
            "sitting".into(),
            "kitchen".into(),
            "zzzzzz".into(),
        ];
        // max_distance=2, gram_len = 6/3 = 2
        let pruned = prune_fuzzy_candidates(&terms, "kitten", 2);
        assert!(pruned.contains(&"kitten"));
        assert!(pruned.contains(&"kitchen"));
        // "zzzzzz" shares no 2-gram with "kitten"
        assert!(!pruned.contains(&"zzzzzz"));
    }
}
