//! People, embeddings, and gesture maps under `~/.config/buckyboi/`.

use crate::identity::config_dir;
use crate::identity::embed::{
    best_match, Embedding, MatchHit, DEFAULT_FACE_THRESHOLD, DEFAULT_GESTURE_THRESHOLD,
    DEFAULT_VOICE_THRESHOLD,
};
use crate::identity::face::{FACE_KIND_ARCFACE, FACE_KIND_PROBE};
use crate::identity::gate::GateMode;
use crate::identity::hands::{
    gesture_centroid, GestureAction, GestureClass, GestureMap, DEFAULT_GESTURE_MAP,
};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Person {
    pub id: String,
    pub name: String,
    pub created_ms: u64,
    #[serde(default)]
    pub face: Vec<Embedding>,
    #[serde(default)]
    pub voice: Vec<Embedding>,
    #[serde(default)]
    pub gesture_samples: Vec<crate::identity::hands::GestureSample>,
    #[serde(default)]
    pub gesture_map: Option<Vec<(GestureClass, GestureAction)>>,
}

impl Person {
    pub fn new(name: impl Into<String>) -> Self {
        let name = name.into();
        let created_ms = now_ms();
        let slug: String = name
            .chars()
            .filter(|c| c.is_ascii_alphanumeric())
            .take(12)
            .collect::<String>()
            .to_ascii_lowercase();
        let slug = if slug.is_empty() {
            "person".into()
        } else {
            slug
        };
        Self {
            id: format!("{slug}-{created_ms:x}"),
            name,
            created_ms,
            face: Vec::new(),
            voice: Vec::new(),
            gesture_samples: Vec::new(),
            gesture_map: None,
        }
    }

    pub fn gesture_centroids(&self) -> Vec<(GestureClass, Vec<f32>)> {
        let mut out = Vec::new();
        for cls in GestureClass::all() {
            let xs: Vec<Vec<f32>> = self
                .gesture_samples
                .iter()
                .filter(|s| s.class == cls)
                .map(|s| s.landmarks.clone())
                .collect();
            if let Some(c) = gesture_centroid(&xs) {
                out.push((cls, c));
            }
        }
        out
    }

    pub fn map(&self) -> GestureMap {
        let mut map = DEFAULT_GESTURE_MAP;
        if let Some(custom) = &self.gesture_map {
            for (cls, act) in custom {
                if let Some(slot) = map.iter_mut().find(|(c, _)| c == cls) {
                    slot.1 = *act;
                }
            }
        }
        map
    }

    pub fn has_face(&self) -> bool {
        !self.face.is_empty()
    }

    pub fn has_voice(&self) -> bool {
        !self.voice.is_empty()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProfileFile {
    pub version: u32,
    #[serde(default)]
    pub people: Vec<Person>,
    #[serde(default)]
    pub face_threshold: f32,
    #[serde(default)]
    pub voice_threshold: f32,
    #[serde(default)]
    pub gesture_threshold: f32,
}

impl Default for ProfileFile {
    fn default() -> Self {
        Self {
            version: 1,
            people: Vec::new(),
            face_threshold: DEFAULT_FACE_THRESHOLD,
            voice_threshold: DEFAULT_VOICE_THRESHOLD,
            gesture_threshold: DEFAULT_GESTURE_THRESHOLD,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ProfileStore {
    pub file: ProfileFile,
    pub path: Option<PathBuf>,
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

impl ProfileStore {
    pub fn empty() -> Self {
        Self {
            file: ProfileFile::default(),
            path: None,
        }
    }

    pub fn profiles_path() -> Option<PathBuf> {
        Some(config_dir()?.join("profiles.json"))
    }

    pub fn load() -> Self {
        let path = Self::profiles_path();
        let mut store = Self {
            file: ProfileFile::default(),
            path: path.clone(),
        };
        if let Some(p) = &path {
            if let Ok(txt) = fs::read_to_string(p) {
                if let Ok(f) = serde_json::from_str::<ProfileFile>(&txt) {
                    store.file = f;
                }
            }
        }
        store.clamp();
        store
    }

    pub fn load_from_str(txt: &str) -> Self {
        let mut store = Self::empty();
        if let Ok(f) = serde_json::from_str::<ProfileFile>(txt) {
            store.file = f;
        }
        store.clamp();
        store
    }

    pub fn clamp(&mut self) {
        self.file.face_threshold = self.file.face_threshold.clamp(0.15, 0.95);
        self.file.voice_threshold = self.file.voice_threshold.clamp(0.20, 0.95);
        self.file.gesture_threshold = self.file.gesture_threshold.clamp(0.40, 0.99);
        if self.file.version == 0 {
            self.file.version = 1;
        }
    }

    pub fn save(&self) {
        let Some(dir) = config_dir() else {
            return;
        };
        let _ = fs::create_dir_all(&dir);
        let path = self.path.clone().or_else(Self::profiles_path);
        if let Some(p) = path {
            if let Ok(txt) = serde_json::to_string_pretty(&self.file) {
                let _ = fs::write(p, txt);
            }
        }
    }

    pub fn people(&self) -> &[Person] {
        &self.file.people
    }

    pub fn people_mut(&mut self) -> &mut Vec<Person> {
        &mut self.file.people
    }

    pub fn get(&self, id: &str) -> Option<&Person> {
        self.file.people.iter().find(|p| p.id == id)
    }

    pub fn get_mut(&mut self, id: &str) -> Option<&mut Person> {
        self.file.people.iter_mut().find(|p| p.id == id)
    }

    pub fn upsert_named(&mut self, name: &str) -> String {
        if let Some(p) = self
            .file
            .people
            .iter()
            .find(|p| p.name.eq_ignore_ascii_case(name))
        {
            return p.id.clone();
        }
        let p = Person::new(name);
        let id = p.id.clone();
        self.file.people.push(p);
        id
    }

    pub fn delete(&mut self, id: &str) -> bool {
        let before = self.file.people.len();
        self.file.people.retain(|p| p.id != id);
        before != self.file.people.len()
    }

    pub fn add_face(&mut self, id: &str, vecs: Vec<Embedding>) {
        if let Some(p) = self.get_mut(id) {
            p.face.extend(vecs);
            if p.face.len() > 16 {
                let extra = p.face.len() - 16;
                p.face.drain(0..extra);
            }
        }
    }

    pub fn add_voice(&mut self, id: &str, vecs: Vec<Embedding>) {
        if let Some(p) = self.get_mut(id) {
            p.voice.extend(vecs);
            if p.voice.len() > 12 {
                let extra = p.voice.len() - 12;
                p.voice.drain(0..extra);
            }
        }
    }

    pub fn add_gesture_samples(&mut self, id: &str, class: GestureClass, landmarks: Vec<Vec<f32>>) {
        if let Some(p) = self.get_mut(id) {
            p.gesture_samples.retain(|s| s.class != class);
            for lm in landmarks {
                p.gesture_samples
                    .push(crate::identity::hands::GestureSample {
                        class,
                        landmarks: lm,
                    });
            }
        }
    }

    pub fn replace_face(&mut self, id: &str, vecs: Vec<Embedding>) {
        if let Some(p) = self.get_mut(id) {
            p.face = vecs;
        }
    }

    pub fn replace_voice(&mut self, id: &str, vecs: Vec<Embedding>) {
        if let Some(p) = self.get_mut(id) {
            p.voice = vecs;
        }
    }

    pub fn match_face(&self, probe: &Embedding) -> Option<MatchHit> {
        let gal: Vec<_> = self
            .file
            .people
            .iter()
            .map(|p| (p.id.clone(), p.name.clone(), p.face.clone()))
            .collect();
        let model = crate::identity::face::loaded_rec_model_name();
        let thresh = crate::identity::face::effective_face_threshold(
            self.file.face_threshold,
            model.as_deref(),
        );
        best_match(probe, &gal, thresh)
    }

    pub fn match_voice(&self, probe: &Embedding) -> Option<MatchHit> {
        let gal: Vec<_> = self
            .file
            .people
            .iter()
            .map(|p| (p.id.clone(), p.name.clone(), p.voice.clone()))
            .collect();
        let model = crate::identity::voice::pick_speaker_model(
            crate::identity::models_dir()
                .unwrap_or_else(|| std::path::PathBuf::from("."))
                .as_path(),
        );
        let name = model
            .as_ref()
            .and_then(|p| p.file_name())
            .and_then(|s| s.to_str());
        let thresh = crate::identity::voice::voice_cosine_threshold(name);
        let thresh = if (self.file.voice_threshold - DEFAULT_VOICE_THRESHOLD).abs() < 1e-4 {
            thresh
        } else {
            self.file.voice_threshold
        };
        best_match(probe, &gal, thresh)
    }

    pub fn enrolled_count(&self) -> usize {
        self.file.people.len()
    }

    pub fn next_default_name(&self) -> String {
        format!("P{}", self.file.people.len() + 1)
    }
}

/// Identity knobs that live next to the overlay look sliders.
#[derive(Clone, Debug)]
pub struct IdentitySettings {
    pub gate: GateMode,
    pub gestures_need_face: bool,
    pub face_threshold: f32,
    pub voice_threshold: f32,
}

impl Default for IdentitySettings {
    fn default() -> Self {
        Self {
            gate: GateMode::Off,
            gestures_need_face: true,
            face_threshold: DEFAULT_FACE_THRESHOLD,
            voice_threshold: DEFAULT_VOICE_THRESHOLD,
        }
    }
}

pub fn parse_identity_ini(txt: &str, into: &mut IdentitySettings) {
    for line in txt.lines() {
        let Some((k, v)) = line.split_once('=') else {
            continue;
        };
        match k.trim() {
            "gate" => into.gate = GateMode::parse(v),
            "gestures_need_face" => {
                into.gestures_need_face = v.trim() == "1" || v.trim().eq_ignore_ascii_case("true")
            }
            "face_threshold" => {
                if let Ok(n) = v.trim().parse() {
                    into.face_threshold = n;
                }
            }
            "voice_threshold" => {
                if let Ok(n) = v.trim().parse() {
                    into.voice_threshold = n;
                }
            }
            _ => {}
        }
    }
}

pub fn identity_ini_lines(s: &IdentitySettings) -> String {
    format!(
        "gate={}\ngestures_need_face={}\nface_threshold={:.3}\nvoice_threshold={:.3}\n",
        s.gate.as_str(),
        if s.gestures_need_face { 1 } else { 0 },
        s.face_threshold,
        s.voice_threshold
    )
}

pub fn face_kinds_compatible(a: &str, b: &str) -> bool {
    a == b
        || (a == FACE_KIND_ARCFACE && b == FACE_KIND_ARCFACE)
        || (a == FACE_KIND_PROBE && b == FACE_KIND_PROBE)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::embed::Embedding;

    #[test]
    fn roundtrip_person_json() {
        let mut store = ProfileStore::empty();
        let id = store.upsert_named("Ada");
        store.add_face(&id, vec![Embedding::new("arcface", vec![1.0, 0.0, 0.0])]);
        let txt = serde_json::to_string(&store.file).unwrap();
        let loaded = ProfileStore::load_from_str(&txt);
        assert_eq!(loaded.people().len(), 1);
        assert_eq!(loaded.people()[0].name, "Ada");
        assert!(loaded.people()[0].has_face());
    }

    #[test]
    fn match_and_delete() {
        let mut store = ProfileStore::empty();
        let a = store.upsert_named("Ada");
        let b = store.upsert_named("Bo");
        store.add_face(&a, vec![Embedding::new("arcface", vec![1.0, 0.0])]);
        store.add_face(&b, vec![Embedding::new("arcface", vec![0.0, 1.0])]);
        let hit = store
            .match_face(&Embedding::new("arcface", vec![0.99, 0.02]))
            .unwrap();
        assert_eq!(hit.name, "Ada");
        assert!(store.delete(&a));
        assert_eq!(store.enrolled_count(), 1);
        assert!(store
            .match_face(&Embedding::new("arcface", vec![0.99, 0.02]))
            .is_none());
    }

    #[test]
    fn upsert_reuses_name() {
        let mut store = ProfileStore::empty();
        let a = store.upsert_named("Ada");
        let b = store.upsert_named("ada");
        assert_eq!(a, b);
        assert_eq!(store.enrolled_count(), 1);
    }

    #[test]
    fn identity_ini() {
        let mut s = IdentitySettings::default();
        parse_identity_ini(
            "gate=all\ngestures_need_face=0\nface_threshold=0.5\n",
            &mut s,
        );
        assert_eq!(s.gate, GateMode::All);
        assert!(!s.gestures_need_face);
        assert!((s.face_threshold - 0.5).abs() < 1e-5);
        let txt = identity_ini_lines(&s);
        assert!(txt.contains("gate=all"));
    }
}
