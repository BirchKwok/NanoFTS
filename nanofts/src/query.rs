//! Query helpers: multi-term AND, phrase filtering, BM25 ranking.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::analyzer::AnalyzedDocument;
use crate::bitmap::FastBitmap;
use crate::doc_tokens::{DocTokenStore, RankedHit};

/// Intersect postings sorted by ascending length (early exit on empty).
///
/// Takes borrowed bitmaps and only clones the smallest one as the accumulator,
/// avoiding a full `Vec` clone of every posting list.
pub fn intersect_postings(bitmaps: &[FastBitmap]) -> FastBitmap {
    match bitmaps.len() {
        0 => FastBitmap::new(),
        1 => bitmaps[0].clone(),
        _ => {
            let mut order: Vec<usize> = (0..bitmaps.len()).collect();
            order.sort_unstable_by_key(|&i| bitmaps[i].len());
            let mut result = bitmaps[order[0]].clone();
            for &i in &order[1..] {
                result.and_inplace(&bitmaps[i]);
                if result.is_empty() {
                    return result;
                }
            }
            result
        }
    }
}

/// Filter candidates to those whose token stream contains the query token sequence.
pub fn filter_phrase(
    candidates: &FastBitmap,
    query_tokens: &[u32],
    store: &DocTokenStore,
) -> FastBitmap {
    if query_tokens.is_empty() {
        return FastBitmap::new();
    }
    let mut matches = FastBitmap::new();
    for doc_id in candidates.iter() {
        if store.get(doc_id).is_some_and(|tokens| {
            tokens
                .windows(query_tokens.len())
                .any(|window| window == query_tokens)
        }) {
            matches.add(doc_id);
        }
    }
    matches
}

/// Map local analyzed token ids (1-based into analyzed.terms) to global term ids via lookup.
pub fn map_query_tokens(
    analyzed: &AnalyzedDocument,
    lookup: impl Fn(&str) -> Option<u32>,
) -> Vec<u32> {
    analyzed
        .tokens
        .iter()
        .filter_map(|&local_id| {
            if local_id == 0 {
                None
            } else {
                analyzed
                    .terms
                    .get(local_id as usize - 1)
                    .and_then(|term| lookup(term))
            }
        })
        .collect()
}

/// BM25 top-K ranking over candidates using doc token store.
pub fn bm25_rank(
    candidates: &FastBitmap,
    query_term_ids: &[(u32, f32)], // (global_term_id, idf)
    store: &DocTokenStore,
    average_length: f32,
    limit: usize,
) -> Vec<RankedHit> {
    if limit == 0 || candidates.is_empty() || query_term_ids.is_empty() {
        return Vec::new();
    }
    const K1: f32 = 1.2;
    const B: f32 = 0.75;
    let heap_limit = limit.min(candidates.len() as usize);
    let mut top = BinaryHeap::with_capacity(heap_limit.saturating_add(1));

    for doc_id in candidates.iter() {
        let Some(tokens) = store.get(doc_id) else {
            continue;
        };
        let document_length = tokens.iter().filter(|&&token| token != 0).count() as f32;
        if document_length == 0.0 || average_length <= 0.0 {
            continue;
        }
        let normalization = K1 * (1.0 - B + B * document_length / average_length);
        let mut score = 0.0f32;
        for &(term_id, term_idf) in query_term_ids {
            let frequency = tokens.iter().filter(|&&token| token == term_id).count() as f32;
            if frequency > 0.0 {
                score += term_idf * frequency * (K1 + 1.0) / (frequency + normalization);
            }
        }
        let hit = RankedHit { doc_id, score };
        if top.len() < heap_limit {
            top.push(Reverse(hit));
        } else if top.peek().is_some_and(|worst| hit > worst.0) {
            top.pop();
            top.push(Reverse(hit));
        }
    }

    let mut hits: Vec<RankedHit> = top.into_iter().map(|Reverse(hit)| hit).collect();
    hits.sort_unstable_by(|left, right| right.cmp(left));
    hits
}

/// Okapi BM25 IDF.
pub fn bm25_idf(doc_count: f32, document_frequency: f32) -> f32 {
    ((doc_count - document_frequency + 0.5) / (document_frequency + 0.5) + 1.0).ln()
}
