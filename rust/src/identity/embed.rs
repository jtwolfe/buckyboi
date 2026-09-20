//! Embedding helpers: L2 normalize, cosine, fail-closed match.

use serde::{Deserialize, Serialize};

/// R50 working point. MBF should use 0.40 — see `face::face_cosine_threshold`.
pub const DEFAULT_FACE_THRESHOLD: f32 = 0.35;
/// Sherpa speaker manager search. Log-mel fallback is slightly higher (0.62).
pub const DEFAULT_VOICE_THRESHOLD: f32 = 0.60;
pub const DEFAULT_GESTURE_THRESHOLD: f32 = 0.78;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Embedding {
    pub kind: String,
    pub values: Vec<f32>,
}

impl Embedding {
    pub fn new(kind: impl Into<String>, values: Vec<f32>) -> Self {
        Self {
            kind: kind.into(),
            values: l2_normalize(&values),
        }
    }

    pub fn dim(&self) -> usize {
        self.values.len()
    }
}

pub fn l2_normalize(v: &[f32]) -> Vec<f32> {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n < 1e-9 {
        return v.to_vec();
    }
    v.iter().map(|x| x / n).collect()
}

/// Cosine similarity in [-1, 1]. Different kinds or empty → None (fail closed).
pub fn cosine(a: &Embedding, b: &Embedding) -> Option<f32> {
    if a.kind != b.kind || a.values.is_empty() || a.values.len() != b.values.len() {
        return None;
    }
    let dot = a
        .values
        .iter()
        .zip(b.values.iter())
        .map(|(x, y)| x * y)
        .sum::<f32>();
    Some(dot.clamp(-1.0, 1.0))
}

#[derive(Clone, Debug, PartialEq)]
pub struct MatchHit {
    pub person_id: String,
    pub name: String,
    pub score: f32,
}

/// Highest cosine among stored vectors, if it clears `threshold`. Ties: first person.
pub fn best_match(
    probe: &Embedding,
    gallery: &[(String, String, Vec<Embedding>)],
    threshold: f32,
) -> Option<MatchHit> {
    if !(0.0..=1.0).contains(&threshold) {
        return None;
    }
    let mut best: Option<MatchHit> = None;
    for (id, name, vecs) in gallery {
        for emb in vecs {
            let Some(score) = cosine(probe, emb) else {
                continue;
            };
            if score < threshold {
                continue;
            }
            let better = match &best {
                None => true,
                Some(h) => score > h.score + 1e-6,
            };
            if better {
                best = Some(MatchHit {
                    person_id: id.clone(),
                    name: name.clone(),
                    score,
                });
            }
        }
    }
    best
}

/// Ambiguous if the runner-up is within `margin` of the winner.
pub fn is_ambiguous(
    probe: &Embedding,
    gallery: &[(String, String, Vec<Embedding>)],
    winner_id: &str,
    winner_score: f32,
    margin: f32,
) -> bool {
    let mut runner = f32::NEG_INFINITY;
    for (id, _, vecs) in gallery {
        if id == winner_id {
            continue;
        }
        for emb in vecs {
            if let Some(score) = cosine(probe, emb) {
                runner = runner.max(score);
            }
        }
    }
    runner + margin >= winner_score
}

#[cfg(test)]
mod tests {
    use super::*;

    fn e(kind: &str, v: Vec<f32>) -> Embedding {
        Embedding::new(kind, v)
    }

    #[test]
    fn cosine_self_is_one() {
        let a = e("arcface", vec![1.0, 2.0, 3.0]);
        assert!((cosine(&a, &a).unwrap() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn cosine_rejects_kind_mismatch() {
        let a = e("arcface", vec![1.0, 0.0]);
        let b = e("logmel", vec![1.0, 0.0]);
        assert!(cosine(&a, &b).is_none());
    }

    #[test]
    fn match_fail_closed_below_threshold() {
        let probe = e("arcface", vec![1.0, 0.0, 0.0]);
        let other = e("arcface", vec![0.2, 0.98, 0.0]);
        let gal = vec![("p1".into(), "Ada".into(), vec![other])];
        assert!(best_match(&probe, &gal, 0.9).is_none());
    }

    #[test]
    fn match_picks_highest_over_threshold() {
        let probe = e("arcface", vec![1.0, 0.0]);
        let gal = vec![
            ("p1".into(), "Ada".into(), vec![e("arcface", vec![0.95, 0.05])]),
            ("p2".into(), "Bo".into(), vec![e("arcface", vec![0.2, 0.9])]),
        ];
        let hit = best_match(&probe, &gal, 0.5).unwrap();
        assert_eq!(hit.person_id, "p1");
        assert!(hit.score > 0.9);
    }

    #[test]
    fn ambiguous_near_neighbor() {
        let probe = e("k", vec![1.0, 0.0]);
        let gal = vec![
            ("a".into(), "A".into(), vec![e("k", vec![1.0, 0.02])]),
            ("b".into(), "B".into(), vec![e("k", vec![0.99, 0.05])]),
        ];
        let hit = best_match(&probe, &gal, 0.5).unwrap();
        assert!(is_ambiguous(&probe, &gal, &hit.person_id, hit.score, 0.05));
    }
}
