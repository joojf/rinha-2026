use crate::dataset::Dataset;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

const NPROBE: usize = 32;

pub fn knn5_fraud_count_ivf(query: &[f32; 14], ds: &Dataset) -> u8 {
    #[cfg(target_arch = "x86_64")]
    return unsafe { knn5_ivf_avx2(query, ds) };

    #[cfg(not(target_arch = "x86_64"))]
    knn5_ivf_scalar(query, ds)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn knn5_ivf_avx2(query: &[f32; 14], ds: &Dataset) -> u8 {
    let probes = top_nprobe_centroids_avx2(query, ds);

    let mut q_vecs = [unsafe { _mm256_setzero_ps() }; 14];
    for d in 0..14usize {
        unsafe {
            q_vecs[d] = _mm256_set1_ps(query[d]);
        }
    }

    let mut top: [(f32, u8); 5] = [(f32::INFINITY, 0); 5];
    let mut worst_idx = 0usize;

    let blocks_ptr = ds.blocks.as_ptr();
    let labels_ptr = ds.labels.as_ptr();

    for &ci in &probes {
        let start_block = unsafe { *ds.offsets.as_ptr().add(ci) } as usize;
        let end_block = unsafe { *ds.offsets.as_ptr().add(ci + 1) } as usize;
        unsafe {
            scan_blocks_avx2(
                &q_vecs,
                blocks_ptr,
                labels_ptr,
                start_block,
                end_block,
                &mut top,
                &mut worst_idx,
            );
        }
    }

    top.iter().filter(|(_, l)| *l == 1).count() as u8
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn top_nprobe_centroids_avx2(query: &[f32; 14], ds: &Dataset) -> [usize; NPROBE] {
    let k = ds.k;
    let centroids_ptr = ds.centroids.as_ptr();

    let mut dists = [0.0f32; 1024];

    for d in 0..14usize {
        let qd = unsafe { _mm256_set1_ps(query[d]) };
        let base = d * k;
        let mut ci = 0usize;
        while ci + 8 <= k {
            unsafe {
                let cv = _mm256_loadu_ps(centroids_ptr.add(base + ci));
                let acc = _mm256_loadu_ps(dists.as_ptr().add(ci));
                let diff = _mm256_sub_ps(cv, qd);
                let new_acc = _mm256_fmadd_ps(diff, diff, acc);
                _mm256_storeu_ps(dists.as_mut_ptr().add(ci), new_acc);
            }
            ci += 8;
        }
    }

    let mut result = [0usize; NPROBE];
    let mut result_dist = [f32::INFINITY; NPROBE];
    let mut worst_in_result = 0usize;

    for ci in 0..k {
        let d = unsafe { *dists.as_ptr().add(ci) };
        if d < result_dist[worst_in_result] {
            result[worst_in_result] = ci;
            result_dist[worst_in_result] = d;
            let mut wi = 0;
            let mut wv = result_dist[0];
            for j in 1..NPROBE {
                if result_dist[j] > wv {
                    wv = result_dist[j];
                    wi = j;
                }
            }
            worst_in_result = wi;
        }
    }

    result
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn scan_blocks_avx2(
    q_vecs: &[__m256; 14],
    blocks_ptr: *const f32,
    labels_ptr: *const u8,
    start_block: usize,
    end_block: usize,
    top: &mut [(f32, u8); 5],
    worst_idx: &mut usize,
) {
    for block_i in start_block..end_block {
        let prefetch_block = block_i + 8;
        if prefetch_block < end_block {
            unsafe {
                _mm_prefetch(
                    blocks_ptr.add(prefetch_block * 112) as *const i8,
                    _MM_HINT_T0,
                );
                _mm_prefetch(
                    blocks_ptr.add(prefetch_block * 112 + 56) as *const i8,
                    _MM_HINT_T0,
                );
            }
        }
        let block_base = block_i * 112;
        let mut acc0 = unsafe { _mm256_setzero_ps() };
        let mut acc1 = unsafe { _mm256_setzero_ps() };
        for d in (0..14usize).step_by(2) {
            unsafe {
                let v0 = _mm256_load_ps(blocks_ptr.add(block_base + d * 8));
                let v1 = _mm256_load_ps(blocks_ptr.add(block_base + (d + 1) * 8));
                let diff0 = _mm256_sub_ps(v0, q_vecs[d]);
                let diff1 = _mm256_sub_ps(v1, q_vecs[d + 1]);
                acc0 = _mm256_fmadd_ps(diff0, diff0, acc0);
                acc1 = _mm256_fmadd_ps(diff1, diff1, acc1);
            }
        }
        let acc = unsafe { _mm256_add_ps(acc0, acc1) };
        let mut dists = [0.0f32; 8];
        unsafe { _mm256_storeu_ps(dists.as_mut_ptr(), acc) };
        let label_base = block_i * 8;
        for slot in 0..8usize {
            let di = dists[slot];
            if di < top[*worst_idx].0 {
                top[*worst_idx] = (di, unsafe { *labels_ptr.add(label_base + slot) });
                let mut wi = 0;
                let mut wv = top[0].0;
                for j in 1..5 {
                    if top[j].0 > wv {
                        wv = top[j].0;
                        wi = j;
                    }
                }
                *worst_idx = wi;
            }
        }
    }
}

fn knn5_ivf_scalar(query: &[f32; 14], ds: &Dataset) -> u8 {
    let k = ds.k;
    let mut dists = vec![0.0f32; k];
    for d in 0..14usize {
        let qd = query[d];
        let base = d * k;
        for ci in 0..k {
            let diff = ds.centroids[base + ci] - qd;
            dists[ci] += diff * diff;
        }
    }

    let mut probes = [0usize; NPROBE];
    let mut probe_dists = [f32::INFINITY; NPROBE];
    let mut worst = 0usize;
    for ci in 0..k {
        if dists[ci] < probe_dists[worst] {
            probes[worst] = ci;
            probe_dists[worst] = dists[ci];
            let mut wi = 0;
            let mut wv = probe_dists[0];
            for j in 1..NPROBE {
                if probe_dists[j] > wv {
                    wv = probe_dists[j];
                    wi = j;
                }
            }
            worst = wi;
        }
    }

    let mut top: [(f32, u8); 5] = [(f32::INFINITY, 0); 5];
    let mut worst_idx = 0usize;
    let blocks = ds.blocks.as_slice();
    let labels = ds.labels.as_slice();

    for &ci in &probes {
        let start_block = ds.offsets[ci] as usize;
        let end_block = ds.offsets[ci + 1] as usize;
        for block_i in start_block..end_block {
            let block_base = block_i * 112;
            for slot in 0..8usize {
                let mut dist = 0.0f32;
                for d in 0..14usize {
                    let v = blocks[block_base + d * 8 + slot];
                    let diff = v - query[d];
                    dist += diff * diff;
                }
                let label = labels[block_i * 8 + slot];
                if dist < top[worst_idx].0 {
                    top[worst_idx] = (dist, label);
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
    }

    top.iter().filter(|(_, l)| *l == 1).count() as u8
}

pub fn knn5_fraud_count_blocks(query: &[f32; 14], ds: &Dataset) -> u8 {
    #[cfg(target_arch = "x86_64")]
    return unsafe { knn5_blocks_avx2(query, ds) };

    #[cfg(not(target_arch = "x86_64"))]
    return knn5_blocks_scalar(query, ds);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn knn5_blocks_avx2(query: &[f32; 14], ds: &Dataset) -> u8 {
    let mut q_vecs = [unsafe { _mm256_setzero_ps() }; 14];
    for d in 0..14usize {
        unsafe {
            q_vecs[d] = _mm256_set1_ps(query[d]);
        }
    }

    let mut top: [(f32, u8); 5] = [(f32::INFINITY, 0); 5];
    let mut worst_idx = 0usize;

    let n_blocks = ds.padded_n / 8;
    let blocks_ptr = ds.blocks.as_ptr();
    let labels_ptr = ds.labels.as_ptr();

    unsafe {
        scan_blocks_avx2(
            &q_vecs,
            blocks_ptr,
            labels_ptr,
            0,
            n_blocks,
            &mut top,
            &mut worst_idx,
        );
    }
    top.iter().filter(|(_, l)| *l == 1).count() as u8
}

fn knn5_blocks_scalar(query: &[f32; 14], ds: &Dataset) -> u8 {
    let mut top: [(f32, u8); 5] = [(f32::INFINITY, 0); 5];
    let mut worst_idx = 0usize;
    let n_blocks = ds.padded_n / 8;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::Dataset;
    use aligned_vec::{AVec, ConstAlign};

    fn synth(rows: &[([f32; 14], u8)]) -> Dataset {
        let len = rows.len();
        let padded_n = len.next_multiple_of(8);
        let n_blocks = padded_n / 8;
        let mut blocks: AVec<f32, ConstAlign<32>> = AVec::with_capacity(32, n_blocks * 112);
        let mut labels = Vec::with_capacity(padded_n);

        let mut row_buf = [[f32::INFINITY; 14]; 8];
        let mut label_buf = [0u8; 8];
        let mut buf_idx = 0usize;

        for (v, l) in rows {
            row_buf[buf_idx] = *v;
            label_buf[buf_idx] = *l;
            buf_idx += 1;
            if buf_idx == 8 {
                for d in 0..14 {
                    for k in 0..8 {
                        blocks.push(row_buf[k][d]);
                    }
                }
                for k in 0..8 {
                    labels.push(label_buf[k]);
                }
                row_buf = [[f32::INFINITY; 14]; 8];
                label_buf = [0u8; 8];
                buf_idx = 0;
            }
        }
        if buf_idx > 0 {
            for d in 0..14 {
                for k in 0..8 {
                    blocks.push(row_buf[k][d]);
                }
            }
            for k in 0..8 {
                labels.push(label_buf[k]);
            }
        }

        Dataset {
            blocks,
            labels,
            n: len,
            padded_n,
            centroids: AVec::new(32),
            offsets: vec![0, n_blocks as u32],
            k: 1,
        }
    }

    #[test]
    fn top5_todos_fraude() {
        let mut rows = vec![];
        for _ in 0..3 {
            rows.push(([0.0; 14], 1));
        }
        for _ in 0..2 {
            rows.push(([0.01; 14], 1));
        }
        for _ in 0..5 {
            rows.push(([0.9; 14], 0));
        }
        let ds = synth(&rows);
        assert_eq!(knn5_fraud_count_blocks(&[0.0; 14], &ds), 5);
    }

    #[test]
    fn top5_todos_legit() {
        let mut rows = vec![];
        for _ in 0..5 {
            rows.push(([0.0; 14], 0));
        }
        for _ in 0..5 {
            rows.push(([0.9; 14], 1));
        }
        let ds = synth(&rows);
        assert_eq!(knn5_fraud_count_blocks(&[0.0; 14], &ds), 0);
    }

    #[test]
    fn empate_3_2() {
        let rows = vec![
            ([0.00; 14], 1),
            ([0.01; 14], 1),
            ([0.02; 14], 1),
            ([0.03; 14], 0),
            ([0.04; 14], 0),
            ([0.9; 14], 0),
            ([0.9; 14], 0),
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
        let c = knn5_fraud_count_ivf(&[0.0; 14], &ds);
        assert!(c <= 5);
    }

    #[test]
    fn ivf_recall_vs_exato() {
        let ds = Dataset::load_embedded().unwrap();
        let mut state = 0x12345678u32;
        let mut mismatches = 0u32;
        let mut binary_mismatches = 0u32;
        let n_queries = 500;
        for _ in 0..n_queries {
            let mut q = [0.0f32; 14];
            for v in q.iter_mut() {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                *v = (state >> 8) as f32 / (1u32 << 24) as f32;
            }
            let exact = knn5_fraud_count_blocks(&q, &ds);
            let approx = knn5_fraud_count_ivf(&q, &ds);
            if exact != approx {
                mismatches += 1;
                if (exact >= 3) != (approx >= 3) {
                    binary_mismatches += 1;
                }
            }
        }
        let recall = 1.0 - mismatches as f64 / n_queries as f64;
        let binary_recall = 1.0 - binary_mismatches as f64 / n_queries as f64;
        eprintln!(
            "IVF recall: {:.1}% ({} mismatches / {}), binary: {:.2}% ({} threshold-crossing)",
            recall * 100.0, mismatches, n_queries,
            binary_recall * 100.0, binary_mismatches
        );
        assert!(
            binary_recall >= 0.995,
            "binary recall {:.2}% below 99.5%",
            binary_recall * 100.0
        );
    }

    #[test]
    fn consistencia_resultados() {
        let ds = Dataset::load_embedded().unwrap();
        let q1 = [0.0; 14];
        let q2 = [0.5; 14];
        let a1 = knn5_fraud_count_ivf(&q1, &ds);
        let _ = knn5_fraud_count_ivf(&q2, &ds);
        let a3 = knn5_fraud_count_ivf(&q1, &ds);
        assert_eq!(a1, a3);
    }
}
