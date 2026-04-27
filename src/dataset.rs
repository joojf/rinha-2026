use aligned_vec::{AVec, ConstAlign};
use flate2::read::GzDecoder;
use serde::de::{Deserializer, SeqAccess, Visitor};
use serde::Deserialize;
use std::fmt;

pub struct Dataset {
    pub blocks: AVec<f32, ConstAlign<32>>,
    pub labels: Vec<u8>,
    pub len: usize,
    pub padded_len: usize,
}

impl Dataset {
    pub fn load_embedded() -> Result<Self, Box<dyn std::error::Error>> {
        let compressed = include_bytes!("../spec/resources/references.json.gz");
        let gz = GzDecoder::new(&compressed[..]);
        let mut de = serde_json::Deserializer::from_reader(gz);
        let ds = de.deserialize_seq(DatasetVisitor::new())?;
        Ok(ds)
    }
}

struct DatasetVisitor {
    blocks: AVec<f32, ConstAlign<32>>,
    labels: Vec<u8>,
    len: usize,
    row_buf: [[f32; 14]; 8],
    label_buf: [u8; 8],
    buf_idx: usize,
}

impl DatasetVisitor {
    fn new() -> Self {
        DatasetVisitor {
            blocks: AVec::with_capacity(32, 1_000_008 / 8 * 112),
            labels: Vec::with_capacity(1_000_008),
            len: 0,
            row_buf: [[0.0; 14]; 8],
            label_buf: [0; 8],
            buf_idx: 0,
        }
    }

    fn flush_block(&mut self) {
        for d in 0..14 {
            for k in 0..8 {
                self.blocks.push(self.row_buf[k][d]);
            }
        }
        for k in 0..8 {
            self.labels.push(self.label_buf[k]);
        }
        self.buf_idx = 0;
    }
}

impl<'de> Visitor<'de> for DatasetVisitor {
    type Value = Dataset;

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "array of reference entries")
    }

    fn visit_seq<A: SeqAccess<'de>>(mut self, mut seq: A) -> Result<Dataset, A::Error> {
        while let Some(entry) = seq.next_element::<RefEntry>()? {
            self.row_buf[self.buf_idx] = entry.vector;
            self.label_buf[self.buf_idx] = if entry.label == "fraud" { 1 } else { 0 };
            self.buf_idx += 1;
            self.len += 1;

            if self.buf_idx == 8 {
                self.flush_block();
            }
        }

        if self.buf_idx > 0 {
            for k in self.buf_idx..8 {
                self.row_buf[k] = [f32::INFINITY; 14];
                self.label_buf[k] = 0;
            }
            self.flush_block();
        }

        let len = self.len;
        let padded_len = len.next_multiple_of(8);

        Ok(Dataset { blocks: self.blocks, labels: self.labels, len, padded_len })
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
    fn carrega_entradas() {
        let ds = Dataset::load_embedded().unwrap();
        assert!(ds.len > 0);
        assert_eq!(ds.padded_len % 8, 0);
        assert_eq!(ds.labels.len(), ds.padded_len);
        assert_eq!(ds.blocks.len(), ds.padded_len / 8 * 112);
    }

    #[test]
    fn alinhamento_32_bytes() {
        let ds = Dataset::load_embedded().unwrap();
        let ptr = ds.blocks.as_ptr() as usize;
        assert_eq!(ptr % 32, 0, "blocks não alinhados a 32 bytes");
    }
}
