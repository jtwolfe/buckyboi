//! Voice print: log-mel filterbank embedding + optional sherpa-onnx.
//!
//! The extractor is pure Rust and always compiled. Live capture needs the
//! `voice` feature (`cpal`). A sherpa-onnx speaker model is used when the
//! `voice-sherpa` feature is on *and* the ONNX file is present.

use crate::identity::embed::Embedding;

pub const VOICE_ENROLL_NEED: usize = 3;
pub const VOICE_MIN_MS: u32 = 1_200;
pub const VOICE_KIND_LOGMEL: &str = "logmel";
pub const VOICE_KIND_SHERPA: &str = "sherpa";
pub const VOICE_SAMPLE_RATE: u32 = 16_000;
pub const MEL_BANDS: usize = 40;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VoiceQuality {
    pub duration_ms: u32,
    pub energy: f32,
    pub ok: bool,
}

impl VoiceQuality {
    pub fn assess(samples: &[f32], sample_rate: u32) -> Self {
        let duration_ms = if sample_rate == 0 {
            0
        } else {
            (samples.len() as u64 * 1000 / sample_rate as u64) as u32
        };
        let energy = speech_energy(samples);
        Self {
            duration_ms,
            energy,
            ok: duration_ms >= VOICE_MIN_MS && energy >= 0.012,
        }
    }
}

pub fn speech_energy(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let mean = samples.iter().sum::<f32>() / samples.len() as f32;
    let var = samples.iter().map(|s| (s - mean) * (s - mean)).sum::<f32>() / samples.len() as f32;
    var.sqrt()
}

fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
}

/// Mean + std of 40 log-mel bands (80-d), L2-normalized. Offline speaker print.
pub fn logmel_embed(samples: &[f32], sample_rate: u32) -> Embedding {
    let sr = sample_rate.max(1);
    let win = ((sr as f32 * 0.025).round() as usize).max(32);
    let hop = ((sr as f32 * 0.010).round() as usize).max(16);
    let nfft = win.next_power_of_two();
    let filters = mel_filters(nfft, sr, MEL_BANDS);
    let mut sums = vec![0.0f32; MEL_BANDS];
    let mut sumsq = vec![0.0f32; MEL_BANDS];
    let mut frames = 0u32;
    let mut i = 0usize;
    while i + win <= samples.len() {
        let frame = &samples[i..i + win];
        let spec = dft_power(frame, nfft);
        for b in 0..MEL_BANDS {
            let mut e = 0.0;
            for (k, w) in filters[b].iter().enumerate() {
                if *w == 0.0 {
                    continue;
                }
                e += spec[k] * *w;
            }
            let v = (e + 1e-6).ln();
            sums[b] += v;
            sumsq[b] += v * v;
        }
        frames += 1;
        i += hop;
    }
    let mut out = vec![0.0f32; MEL_BANDS * 2];
    if frames > 0 {
        let n = frames as f32;
        for b in 0..MEL_BANDS {
            let mean = sums[b] / n;
            let var = (sumsq[b] / n - mean * mean).max(0.0);
            out[b] = mean;
            out[MEL_BANDS + b] = var.sqrt();
        }
    }
    Embedding::new(VOICE_KIND_LOGMEL, out)
}

fn mel_filters(nfft: usize, sr: u32, bands: usize) -> Vec<Vec<f32>> {
    let n_bins = nfft / 2 + 1;
    let lo = hz_to_mel(0.0);
    let hi = hz_to_mel(sr as f32 * 0.5);
    let mut points = Vec::with_capacity(bands + 2);
    for i in 0..bands + 2 {
        let m = lo + (hi - lo) * i as f32 / (bands + 1) as f32;
        let hz = mel_to_hz(m);
        let bin = (hz * nfft as f32 / sr as f32).floor() as usize;
        points.push(bin.min(n_bins - 1));
    }
    let mut filters = vec![vec![0.0f32; n_bins]; bands];
    for b in 0..bands {
        let a = points[b];
        let c = points[b + 1];
        let d = points[b + 2];
        if c > a {
            for k in a..c {
                filters[b][k] = (k - a) as f32 / (c - a) as f32;
            }
        }
        if d > c {
            for k in c..d {
                filters[b][k] = (d - k) as f32 / (d - c) as f32;
            }
        }
    }
    filters
}

fn dft_power(frame: &[f32], nfft: usize) -> Vec<f32> {
    let n_bins = nfft / 2 + 1;
    let mut out = vec![0.0f32; n_bins];
    let n = frame.len() as f32;
    for k in 0..n_bins {
        let mut re = 0.0f32;
        let mut im = 0.0f32;
        for (i, s) in frame.iter().enumerate() {
            let w = 0.54 - 0.46 * (2.0 * std::f32::consts::PI * i as f32 / (n - 1.0)).cos();
            let ang = -2.0 * std::f32::consts::PI * k as f32 * i as f32 / nfft as f32;
            re += s * w * ang.cos();
            im += s * w * ang.sin();
        }
        out[k] = re * re + im * im;
    }
    out
}

pub fn extract_embedding(samples: &[f32], sample_rate: u32) -> (VoiceQuality, Option<Embedding>) {
    let q = VoiceQuality::assess(samples, sample_rate);
    if !q.ok {
        return (q, None);
    }
    #[cfg(feature = "voice-sherpa")]
    {
        if let Some(emb) = sherpa::embed(samples, sample_rate) {
            return (q, Some(emb));
        }
    }
    (q, Some(logmel_embed(samples, sample_rate)))
}

/// Downmix / resample interleaved f32 to mono 16 kHz (linear).
pub fn to_mono_16k(samples: &[f32], channels: u16, src_rate: u32) -> Vec<f32> {
    let ch = channels.max(1) as usize;
    let mut mono = Vec::with_capacity(samples.len() / ch + 1);
    let mut i = 0;
    while i + ch <= samples.len() {
        let s = samples[i..i + ch].iter().sum::<f32>() / ch as f32;
        mono.push(s);
        i += ch;
    }
    if src_rate == 0 || src_rate == VOICE_SAMPLE_RATE {
        return mono;
    }
    let ratio = src_rate as f32 / VOICE_SAMPLE_RATE as f32;
    let out_n = ((mono.len() as f32) / ratio).floor() as usize;
    let mut out = Vec::with_capacity(out_n);
    for j in 0..out_n {
        let x = j as f32 * ratio;
        let i0 = x.floor() as usize;
        let i1 = (i0 + 1).min(mono.len().saturating_sub(1));
        let t = x - i0 as f32;
        out.push(mono[i0] * (1.0 - t) + mono[i1] * t);
    }
    out
}

#[cfg(feature = "voice")]
pub mod capture {
    use super::*;
    use std::sync::{mpsc, Arc, Mutex};
    use std::thread;
    use std::time::Duration;

    pub struct MicBuffer {
        inner: Arc<Mutex<Vec<f32>>>,
        rate: u32,
    }

    impl MicBuffer {
        pub fn start() -> Option<Self> {
            use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
            let host = cpal::default_host();
            let dev = host.default_input_device()?;
            let cfg = dev.default_input_config().ok()?;
            let src_rate = cfg.sample_rate().0;
            let ch = cfg.channels();
            let buf = Arc::new(Mutex::new(Vec::new()));
            let buf2 = buf.clone();
            let (ready_tx, ready_rx) = mpsc::channel();
            thread::Builder::new()
                .name("buckyboi-mic".into())
                .spawn(move || {
                    let err_fn = |e| eprintln!("buckyboi: mic error {e}");
                    let stream = match cfg.sample_format() {
                        cpal::SampleFormat::F32 => dev.build_input_stream(
                            &cfg.config(),
                            move |data: &[f32], _| {
                                let mono = to_mono_16k(data, ch, src_rate);
                                if let Ok(mut g) = buf2.lock() {
                                    g.extend_from_slice(&mono);
                                    let cap = VOICE_SAMPLE_RATE as usize * 8;
                                    if g.len() > cap {
                                        let extra = g.len() - cap;
                                        g.drain(0..extra);
                                    }
                                }
                            },
                            err_fn,
                            None,
                        ),
                        cpal::SampleFormat::I16 => dev.build_input_stream(
                            &cfg.config(),
                            move |data: &[i16], _| {
                                let f: Vec<f32> = data.iter().map(|s| *s as f32 / 32768.0).collect();
                                let mono = to_mono_16k(&f, ch, src_rate);
                                if let Ok(mut g) = buf2.lock() {
                                    g.extend_from_slice(&mono);
                                    let cap = VOICE_SAMPLE_RATE as usize * 8;
                                    if g.len() > cap {
                                        let extra = g.len() - cap;
                                        g.drain(0..extra);
                                    }
                                }
                            },
                            err_fn,
                            None,
                        ),
                        _ => {
                            let _ = ready_tx.send(false);
                            return;
                        }
                    };
                    match stream {
                        Ok(s) => {
                            if s.play().is_err() {
                                let _ = ready_tx.send(false);
                                return;
                            }
                            let _ = ready_tx.send(true);
                            loop {
                                thread::sleep(Duration::from_millis(200));
                            }
                        }
                        Err(e) => {
                            eprintln!("buckyboi: mic open failed ({e})");
                            let _ = ready_tx.send(false);
                        }
                    }
                })
                .ok()?;
            match ready_rx.recv_timeout(Duration::from_secs(2)) {
                Ok(true) => Some(Self {
                    inner: buf,
                    rate: VOICE_SAMPLE_RATE,
                }),
                _ => None,
            }
        }

        pub fn take_last_ms(&self, ms: u32) -> Vec<f32> {
            let n = (self.rate as u64 * ms as u64 / 1000) as usize;
            let Ok(g) = self.inner.lock() else {
                return Vec::new();
            };
            if g.len() <= n {
                g.clone()
            } else {
                g[g.len() - n..].to_vec()
            }
        }
    }
}

#[cfg(feature = "voice-sherpa")]
mod sherpa {
    use super::*;
    use crate::identity::models_dir;

    pub fn embed(samples: &[f32], sample_rate: u32) -> Option<Embedding> {
        let dir = models_dir()?;
        let mut model = None;
        for name in [
            "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx",
            "wespeaker.onnx",
            "speaker.onnx",
        ] {
            let p = dir.join(name);
            if p.is_file() {
                model = Some(p.to_string_lossy().into_owned());
                break;
            }
        }
        let model = model?;
        let cfg = sherpa_onnx::SpeakerEmbeddingExtractorConfig {
            model: Some(model),
            num_threads: 1,
            debug: false,
            provider: Some("cpu".into()),
        };
        let extractor = sherpa_onnx::SpeakerEmbeddingExtractor::create(&cfg)?;
        let stream = extractor.create_stream()?;
        stream.accept_waveform(sample_rate as i32, samples);
        stream.input_finished();
        if !extractor.is_ready(&stream) {
            return None;
        }
        let v = extractor.compute(&stream)?;
        if v.is_empty() {
            return None;
        }
        Some(Embedding::new(VOICE_KIND_SHERPA, v))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(freq: f32, ms: u32, amp: f32) -> Vec<f32> {
        let n = VOICE_SAMPLE_RATE * ms / 1000;
        (0..n)
            .map(|i| {
                amp * (2.0 * std::f32::consts::PI * freq * i as f32 / VOICE_SAMPLE_RATE as f32).sin()
            })
            .collect()
    }

    #[test]
    fn quiet_is_rejected() {
        let s = vec![0.0001f32; 16_000];
        let q = VoiceQuality::assess(&s, 16_000);
        assert!(!q.ok);
    }

    #[test]
    fn tone_is_accepted_and_embedded() {
        let s = tone(220.0, 1600, 0.2);
        let (q, emb) = extract_embedding(&s, 16_000);
        assert!(q.ok, "{q:?}");
        let emb = emb.expect("embed");
        assert_eq!(emb.kind, VOICE_KIND_LOGMEL);
        assert_eq!(emb.values.len(), MEL_BANDS * 2);
    }

    #[test]
    fn different_tones_differ() {
        let a = logmel_embed(&tone(180.0, 1600, 0.25), 16_000);
        let b = logmel_embed(&tone(880.0, 1600, 0.25), 16_000);
        let sim = crate::identity::embed::cosine(&a, &b).unwrap();
        assert!(sim < 0.98, "expected distinguishable tones, got {sim}");
    }

    #[test]
    fn resample_mono() {
        let s = vec![1.0f32, -1.0, 1.0, -1.0];
        let out = to_mono_16k(&s, 2, 32_000);
        assert!(!out.is_empty());
    }
}
