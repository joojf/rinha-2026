use aligned_vec::{AVec, ConstAlign};
use flate2::read::GzDecoder;
use serde::Deserialize;
use std::io::Read;

pub struct Dataset {
    pub dims: [AVec<f32, ConstAlign<32>>; 14],
    pub labels: Vec<u8>,
    pub len: usize,
    pub padded_len: usize,
}

impl Dataset {
    pub fn load_embedded() -> Result<Self, Box<dyn std::error::Error>> {
        let compressed = include_bytes!("../spec/resources/references.json.gz");

        let mut gz = GzDecoder::new(&compressed[..]);
        let mut raw = Vec::with_capacity(10_200_000);
        gz.read_to_end(&mut raw)?;

        let entries: Vec<RefEntry> = sonic_rs::from_slice(&raw)?;
        drop(raw);

        let len = entries.len();
        let padded_len = len.next_multiple_of(8);

        let mut dims: [AVec<f32, ConstAlign<32>>; 14] =
            std::array::from_fn(|_| AVec::with_capacity(32, padded_len));
        let mut labels = Vec::with_capacity(padded_len);

        for entry in &entries {
            for (d, &v) in dims.iter_mut().zip(entry.vector.iter()) {
                d.push(v);
            }
            labels.push(if entry.label == "fraud" { 1u8 } else { 0u8 });
        }

        for d in &mut dims {
            while d.len() < padded_len {
                d.push(f32::INFINITY);
            }
        }
        while labels.len() < padded_len {
            labels.push(0);
        }

        Ok(Dataset { dims, labels, len, padded_len })
    }
}

#[derive(Deserialize)]
struct RefEntry {
    vector: [f32; 14],
    label: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn carrega_100k_entradas() {
        let ds = Dataset::load_embedded().unwrap();
        assert_eq!(ds.len, 100_000);
        assert_eq!(ds.padded_len % 8, 0);
        assert_eq!(ds.labels.len(), ds.padded_len);
        assert_eq!(ds.dims[0].len(), ds.padded_len);
        let fraud_count = ds.labels[..ds.len].iter().filter(|&&l| l == 1).count();
        assert_eq!(fraud_count, 33_327);
    }

    #[test]
    fn alinhamento_32_bytes() {
        let ds = Dataset::load_embedded().unwrap();
        for (i, col) in ds.dims.iter().enumerate() {
            let ptr = col.as_ptr() as usize;
            assert_eq!(ptr % 32, 0, "dim {i} não alinhada a 32 bytes");
        }
    }
}
