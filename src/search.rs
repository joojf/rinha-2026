use crate::dataset::Dataset;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub fn knn5_fraud_count_blocks(query: &[f32; 14], ds: &Dataset) -> u8 {
    #[cfg(target_arch = "x86_64")]
    return unsafe { knn5_blocks_avx2(query, ds) };

    #[cfg(not(target_arch = "x86_64"))]
    return knn5_blocks_scalar(query, ds);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn knn5_blocks_avx2(query: &[f32; 14], ds: &Dataset) -> u8 {
    let mut q_vecs = [_mm256_setzero_ps(); 14];
    for d in 0..14 {
        q_vecs[d] = _mm256_set1_ps(query[d]);
    }

    let mut top: [(f32, u8); 5] = [(f32::INFINITY, 0); 5];
    let mut worst_idx = 0usize;

    let n_blocks = ds.padded_len / 8;
    let blocks_ptr = ds.blocks.as_ptr();
    let labels_ptr = ds.labels.as_ptr();

    for block_i in 0..n_blocks {
        let prefetch_base = (block_i + 8) * 112;
        if prefetch_base + 112 <= n_blocks * 112 {
            _mm_prefetch(blocks_ptr.add(prefetch_base) as *const i8, _MM_HINT_T0);
            _mm_prefetch(blocks_ptr.add(prefetch_base + 56) as *const i8, _MM_HINT_T0);
        }
        let block_base = block_i * 112;
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        for d in (0..14).step_by(2) {
            let v0 = _mm256_load_ps(blocks_ptr.add(block_base + d * 8));
            let v1 = _mm256_load_ps(blocks_ptr.add(block_base + (d + 1) * 8));
            let diff0 = _mm256_sub_ps(v0, q_vecs[d]);
            let diff1 = _mm256_sub_ps(v1, q_vecs[d + 1]);
            acc0 = _mm256_fmadd_ps(diff0, diff0, acc0);
            acc1 = _mm256_fmadd_ps(diff1, diff1, acc1);
        }
        let acc = _mm256_add_ps(acc0, acc1);
        let mut dists = [0.0f32; 8];
        _mm256_storeu_ps(dists.as_mut_ptr(), acc);
        let label_base = block_i * 8;
        for k in 0..8 {
            let di = dists[k];
            if di < top[worst_idx].0 {
                top[worst_idx] = (di, *labels_ptr.add(label_base + k));
                let mut wi = 0;
                let mut wv = top[0].0;
                for j in 1..5 {
                    if top[j].0 > wv {
                        wv = top[j].0;
                        wi = j;
                    }
                }
                worst_idx = wi;
            }
        }
    }
    top.iter().filter(|(_, l)| *l == 1).count() as u8
}

fn knn5_blocks_scalar(query: &[f32; 14], ds: &Dataset) -> u8 {
    let mut top: [(f32, u8); 5] = [(f32::INFINITY, 0); 5];
    let mut worst_idx = 0usize;
    let n_blocks = ds.padded_len / 8;
    let blocks = ds.blocks.as_slice();
    let labels = ds.labels.as_slice();
    for block_i in 0..n_blocks {
        let block_base = block_i * 112;
        for k in 0..8 {
            let mut dist = 0.0f32;
            for d in 0..14 {
                let v = blocks[block_base + d * 8 + k];
                let diff = v - query[d];
                dist += diff * diff;
            }
            let label = labels[block_i * 8 + k];
            if dist < top[worst_idx].0 {
                top[worst_idx] = (dist, label);
                let mut wi = 0;
                let mut wv = top[0].0;
                for j in 1..5 {
                    if top[j].0 > wv { wv = top[j].0; wi = j; }
                }
                worst_idx = wi;
            }
        }
    }
    top.iter().filter(|(_, l)| *l == 1).count() as u8
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::Dataset;
    use aligned_vec::{AVec, ConstAlign};

    fn synth(rows: &[([f32; 14], u8)]) -> Dataset {
        let len = rows.len();
        let padded_len = len.next_multiple_of(8);
        let n_blocks = padded_len / 8;
        let mut blocks: AVec<f32, ConstAlign<32>> = AVec::with_capacity(32, n_blocks * 112);
        let mut labels = Vec::with_capacity(padded_len);

        let mut row_buf = [[f32::INFINITY; 14]; 8];
        let mut label_buf = [0u8; 8];
        let mut buf_idx = 0usize;

        for (v, l) in rows {
            row_buf[buf_idx] = *v;
            label_buf[buf_idx] = *l;
            buf_idx += 1;
            if buf_idx == 8 {
                for d in 0..14 { for k in 0..8 { blocks.push(row_buf[k][d]); } }
                for k in 0..8 { labels.push(label_buf[k]); }
                row_buf = [[f32::INFINITY; 14]; 8];
                label_buf = [0u8; 8];
                buf_idx = 0;
            }
        }
        if buf_idx > 0 {
            for d in 0..14 { for k in 0..8 { blocks.push(row_buf[k][d]); } }
            for k in 0..8 { labels.push(label_buf[k]); }
        }

        Dataset { blocks, labels, len, padded_len }
    }

    #[test]
    fn top5_todos_fraude() {
        let mut rows = vec![];
        for _ in 0..3 { rows.push(([0.0; 14], 1)); }
        for _ in 0..2 { rows.push(([0.01; 14], 1)); }
        for _ in 0..5 { rows.push(([0.9; 14], 0)); }
        let ds = synth(&rows);
        assert_eq!(knn5_fraud_count_blocks(&[0.0; 14], &ds), 5);
    }

    #[test]
    fn top5_todos_legit() {
        let mut rows = vec![];
        for _ in 0..5 { rows.push(([0.0; 14], 0)); }
        for _ in 0..5 { rows.push(([0.9; 14], 1)); }
        let ds = synth(&rows);
        assert_eq!(knn5_fraud_count_blocks(&[0.0; 14], &ds), 0);
    }

    #[test]
    fn empate_3_2() {
        let rows = vec![
            ([0.00; 14], 1), ([0.01; 14], 1), ([0.02; 14], 1),
            ([0.03; 14], 0), ([0.04; 14], 0),
            ([0.9; 14], 0), ([0.9; 14], 0),
        ];
        let ds = synth(&rows);
        assert_eq!(knn5_fraud_count_blocks(&[0.0; 14], &ds), 3);
    }

    #[test]
    fn padding_nao_entra_top5() {
        let rows = vec![([0.0; 14], 1), ([0.1; 14], 1), ([0.2; 14], 0)];
        let ds = synth(&rows);
        assert_eq!(knn5_fraud_count_blocks(&[0.0; 14], &ds), 2);
    }

    #[test]
    fn dataset_real_consulta() {
        let ds = Dataset::load_embedded().unwrap();
        let c = knn5_fraud_count_blocks(&[0.0; 14], &ds);
        assert!(c <= 5);
    }

    #[test]
    fn consistencia_resultados() {
        let ds = Dataset::load_embedded().unwrap();
        let q1 = [0.0; 14];
        let q2 = [0.5; 14];
        let a1 = knn5_fraud_count_blocks(&q1, &ds);
        let _ = knn5_fraud_count_blocks(&q2, &ds);
        let a3 = knn5_fraud_count_blocks(&q1, &ds);
        assert_eq!(a1, a3);
    }
}
