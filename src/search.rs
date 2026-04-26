use crate::dataset::Dataset;
use aligned_vec::{AVec, ConstAlign};
use std::cell::RefCell;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

thread_local! {
    static DIST_BUF: RefCell<AVec<f32, ConstAlign<32>>> =
        RefCell::new(AVec::with_capacity(32, 100_032));
}

pub fn knn5_fraud_count(query: &[f32; 14], ds: &Dataset) -> u8 {
    DIST_BUF.with(|cell| {
        let mut dist = cell.borrow_mut();
        if dist.len() < ds.padded_len {
            dist.resize(ds.padded_len, 0.0);
        }
        compute_distances(query, ds, &mut dist);
        select_top5(&dist[..ds.padded_len], &ds.labels)
    })
}

pub fn warm_up_buffer(padded_len: usize) {
    DIST_BUF.with(|cell| {
        cell.borrow_mut().resize(padded_len, 0.0);
    });
}

#[cfg(target_arch = "x86_64")]
fn compute_distances(query: &[f32; 14], ds: &Dataset, dist: &mut [f32]) {
    unsafe { compute_distances_avx2(query, ds, dist) }
}

#[cfg(not(target_arch = "x86_64"))]
fn compute_distances(query: &[f32; 14], ds: &Dataset, dist: &mut [f32]) {
    compute_distances_scalar(query, ds, dist)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn compute_distances_avx2(query: &[f32; 14], ds: &Dataset, dist: &mut [f32]) {
    let n = ds.padded_len;
    let dist_ptr = dist.as_mut_ptr();

    unsafe {
        let q0 = _mm256_set1_ps(query[0]);
        let col0 = ds.dims[0].as_ptr();
        let mut i = 0;
        while i < n {
            let v = _mm256_load_ps(col0.add(i));
            let diff = _mm256_sub_ps(v, q0);
            _mm256_store_ps(dist_ptr.add(i), _mm256_mul_ps(diff, diff));
            i += 8;
        }

        for d in 1..14usize {
            let q = _mm256_set1_ps(query[d]);
            let col = ds.dims[d].as_ptr();
            let mut i = 0;
            while i < n {
                let v = _mm256_load_ps(col.add(i));
                let diff = _mm256_sub_ps(v, q);
                let acc = _mm256_load_ps(dist_ptr.add(i));
                _mm256_store_ps(dist_ptr.add(i), _mm256_fmadd_ps(diff, diff, acc));
                i += 8;
            }
        }
    }
}

#[cfg(any(not(target_arch = "x86_64"), test))]
fn compute_distances_scalar(query: &[f32; 14], ds: &Dataset, dist: &mut [f32]) {
    let n = ds.padded_len;
    let q0 = query[0];
    let col0 = ds.dims[0].as_slice();
    for i in 0..n {
        let diff = col0[i] - q0;
        dist[i] = diff * diff;
    }
    for d in 1..14usize {
        let q = query[d];
        let col = ds.dims[d].as_slice();
        for i in 0..n {
            let diff = col[i] - q;
            dist[i] += diff * diff;
        }
    }
}

#[cfg(target_arch = "x86_64")]
pub fn knn5_fraud_count_blocks(query: &[f32; 14], ds: &Dataset) -> u8 {
    unsafe { knn5_blocks_avx2(query, ds) }
}

#[cfg(not(target_arch = "x86_64"))]
pub fn knn5_fraud_count_blocks(query: &[f32; 14], ds: &Dataset) -> u8 {
    knn5_fraud_count(query, ds)
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

fn select_top5(dist: &[f32], labels: &[u8]) -> u8 {
    let mut top: [(f32, u8); 5] = [(f32::INFINITY, 0); 5];
    let mut worst_idx = 0usize;
    for (i, &di) in dist.iter().enumerate() {
        if di < top[worst_idx].0 {
            top[worst_idx] = (di, labels[i]);
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
        let mut dims: [AVec<f32, ConstAlign<32>>; 14] =
            std::array::from_fn(|_| AVec::with_capacity(32, padded_len));
        let mut labels = Vec::with_capacity(padded_len);
        for (v, l) in rows {
            for (d, &x) in dims.iter_mut().zip(v.iter()) {
                d.push(x);
            }
            labels.push(*l);
        }
        for d in &mut dims {
            while d.len() < padded_len {
                d.push(f32::INFINITY);
            }
        }
        while labels.len() < padded_len {
            labels.push(0);
        }
        let n_blocks = padded_len / 8;
        let mut blocks: AVec<f32, ConstAlign<32>> = AVec::with_capacity(32, n_blocks * 112);
        for block_i in 0..n_blocks {
            let base = block_i * 8;
            for d in 0..14 {
                for k in 0..8 {
                    blocks.push(dims[d][base + k]);
                }
            }
        }
        Dataset { dims, blocks, labels, len, padded_len }
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
        assert_eq!(knn5_fraud_count(&[0.0; 14], &ds), 5);
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
        assert_eq!(knn5_fraud_count(&[0.0; 14], &ds), 0);
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
        assert_eq!(knn5_fraud_count(&[0.0; 14], &ds), 3);
    }

    #[test]
    fn padding_nao_entra_top5() {
        let rows = vec![
            ([0.0; 14], 1),
            ([0.1; 14], 1),
            ([0.2; 14], 0),
        ];
        let ds = synth(&rows);
        assert_eq!(knn5_fraud_count(&[0.0; 14], &ds), 2);
    }

    #[test]
    fn dataset_real_consulta() {
        let ds = Dataset::load_embedded().unwrap();
        let c = knn5_fraud_count(&[0.0; 14], &ds);
        assert!(c <= 5);
    }

    #[test]
    fn buffer_reuso_consistencia() {
        let ds = Dataset::load_embedded().unwrap();
        let q1 = [0.0; 14];
        let q2 = [0.5; 14];
        let a1 = knn5_fraud_count(&q1, &ds);
        let _ = knn5_fraud_count(&q2, &ds);
        let a3 = knn5_fraud_count(&q1, &ds);
        assert_eq!(a1, a3);
    }

    #[test]
    fn blocks_iguala_soa_no_real() {
        let ds = Dataset::load_embedded().unwrap();
        let queries: [[f32; 14]; 3] = [
            [0.0; 14],
            [0.5; 14],
            [0.3, 0.5, 0.1, 0.7, 0.2, 0.4, 0.6, 0.05, 0.9, 1.0, 0.0, 0.0, 0.5, 0.1],
        ];
        for q in &queries {
            let soa = knn5_fraud_count(q, &ds);
            let blk = knn5_fraud_count_blocks(q, &ds);
            assert_eq!(soa, blk, "blocks != soa for query {:?}", q);
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn simd_iguala_escalar_no_real() {
        let ds = Dataset::load_embedded().unwrap();
        let q = [0.3, 0.5, 0.1, 0.7, 0.2, 0.4, 0.6, 0.05, 0.9, 1.0, 0.0, 0.0, 0.5, 0.1];
        let mut buf_simd = AVec::<f32, ConstAlign<32>>::with_capacity(32, ds.padded_len);
        buf_simd.resize(ds.padded_len, 0.0);
        let mut buf_scalar = vec![0.0f32; ds.padded_len];
        unsafe { compute_distances_avx2(&q, &ds, &mut buf_simd) };
        compute_distances_scalar(&q, &ds, &mut buf_scalar);
        for (a, b) in buf_simd.iter().zip(buf_scalar.iter()) {
            assert!((a - b).abs() < 1e-3, "simd {a} vs scalar {b}");
        }
    }
}
