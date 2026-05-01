use crate::dataset::Dataset;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

const FAST_NPROBE: usize = 12;
const FULL_NPROBE: usize = 24;
#[cfg(target_arch = "x86_64")]
const MAX_CENTROIDS: usize = 4096;
const VECTOR_SCALE: f32 = 0.0001;

pub fn knn5_fraud_count_ivf_i16(query: &[i16; 14], ds: &Dataset) -> u8 {
    let fast = {
        #[cfg(target_arch = "x86_64")]
        {
            unsafe { knn5_ivf_i16_avx2::<FAST_NPROBE>(query, ds) }
        }

        #[cfg(not(target_arch = "x86_64"))]
        {
            knn5_ivf_i16_scalar::<FAST_NPROBE>(query, ds)
        }
    };

    if fast != 2 && fast != 3 {
        return fast;
    }

    #[cfg(target_arch = "x86_64")]
    return unsafe { knn5_ivf_i16_avx2::<FULL_NPROBE>(query, ds) };

    #[cfg(not(target_arch = "x86_64"))]
    knn5_ivf_i16_scalar::<FULL_NPROBE>(query, ds)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn knn5_ivf_i16_avx2<const NPROBE: usize>(query: &[i16; 14], ds: &Dataset) -> u8 {
    let probes = unsafe { top_nprobe_centroids_i16_avx2::<NPROBE>(query, ds) };

    let mut q_vecs = [_mm256_setzero_si256(); 14];
    for d in 0..14usize {
        q_vecs[d] = _mm256_set1_epi32(query[d] as i32);
    }

    let mut top: [(i32, u8); 5] = [(i32::MAX, 0); 5];
    let mut worst_idx = 0usize;

    unsafe {
        scan_probes_i16_avx2(
            &probes,
            ds,
            &q_vecs,
            ds.blocks.as_ptr(),
            ds.labels.as_ptr(),
            &mut top,
            &mut worst_idx,
        );
    }

    top.iter().filter(|(_, l)| *l == 1).count() as u8
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn top_nprobe_centroids_i16_avx2<const NPROBE: usize>(
    query: &[i16; 14],
    ds: &Dataset,
) -> [usize; NPROBE] {
    let k = ds.k;
    let centroids_ptr = ds.centroids.as_ptr();

    assert!(k <= MAX_CENTROIDS);
    let mut dists = [0.0f32; MAX_CENTROIDS];

    for d in 0..14usize {
        let qd = _mm256_set1_ps(query[d] as f32 * VECTOR_SCALE);
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

    sort_probe_results(&mut result, &mut result_dist);
    result
}

fn sort_probe_results<const NPROBE: usize>(
    result: &mut [usize; NPROBE],
    result_dist: &mut [f32; NPROBE],
) {
    for i in 1..NPROBE {
        let mut j = i;
        while j > 0 && result_dist[j] < result_dist[j - 1] {
            result_dist.swap(j, j - 1);
            result.swap(j, j - 1);
            j -= 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn scan_probes_i16_avx2(
    probes: &[usize],
    ds: &Dataset,
    q_vecs: &[__m256i; 14],
    blocks_ptr: *const i16,
    labels_ptr: *const u8,
    top: &mut [(i32, u8); 5],
    worst_idx: &mut usize,
) {
    let mut last_ci = usize::MAX;
    for &ci in probes {
        if ci == last_ci {
            continue;
        }
        last_ci = ci;
        let start_block = unsafe { *ds.offsets.as_ptr().add(ci) } as usize;
        let end_block = unsafe { *ds.offsets.as_ptr().add(ci + 1) } as usize;
        unsafe {
            scan_blocks_i16_avx2(
                q_vecs,
                blocks_ptr,
                labels_ptr,
                start_block,
                end_block,
                top,
                worst_idx,
            );
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn scan_blocks_i16_avx2(
    q_vecs: &[__m256i; 14],
    blocks_ptr: *const i16,
    labels_ptr: *const u8,
    start_block: usize,
    end_block: usize,
    top: &mut [(i32, u8); 5],
    worst_idx: &mut usize,
) {
    let padding_i16 = _mm_set1_epi16(i16::MAX);
    let max_dist = _mm256_set1_epi32(i32::MAX);

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
        let first_dim = unsafe { _mm_loadu_si128(blocks_ptr.add(block_base) as *const __m128i) };
        let padding_mask = _mm256_cvtepi16_epi32(_mm_cmpeq_epi16(first_dim, padding_i16));

        let mut acc0 = _mm256_setzero_si256();
        let mut acc1 = _mm256_setzero_si256();
        for d in (0..14usize).step_by(2) {
            unsafe {
                let raw0 = _mm_loadu_si128(blocks_ptr.add(block_base + d * 8) as *const __m128i);
                let raw1 =
                    _mm_loadu_si128(blocks_ptr.add(block_base + (d + 1) * 8) as *const __m128i);
                let diff0 = _mm256_sub_epi32(_mm256_cvtepi16_epi32(raw0), q_vecs[d]);
                let diff1 = _mm256_sub_epi32(_mm256_cvtepi16_epi32(raw1), q_vecs[d + 1]);
                acc0 = _mm256_add_epi32(acc0, _mm256_mullo_epi32(diff0, diff0));
                acc1 = _mm256_add_epi32(acc1, _mm256_mullo_epi32(diff1, diff1));
            }
        }

        let acc = _mm256_blendv_epi8(_mm256_add_epi32(acc0, acc1), max_dist, padding_mask);
        let closer = _mm256_cmpgt_epi32(_mm256_set1_epi32(top[*worst_idx].0), acc);
        let mut mask = _mm256_movemask_ps(_mm256_castsi256_ps(closer)) as u32;
        if mask == 0 {
            continue;
        }

        let mut dists = [0i32; 8];
        unsafe { _mm256_storeu_si256(dists.as_mut_ptr() as *mut __m256i, acc) };
        let label_base = block_i * 8;
        while mask != 0 {
            let slot = mask.trailing_zeros() as usize;
            mask &= mask - 1;

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

#[cfg(not(target_arch = "x86_64"))]
fn knn5_ivf_i16_scalar<const NPROBE: usize>(query: &[i16; 14], ds: &Dataset) -> u8 {
    let probes = top_nprobe_centroids_i16_scalar::<NPROBE>(query, ds);
    let mut top: [(i32, u8); 5] = [(i32::MAX, 0); 5];
    let mut worst_idx = 0usize;

    scan_probes_i16_scalar(&probes, query, ds, &mut top, &mut worst_idx);
    top.iter().filter(|(_, l)| *l == 1).count() as u8
}

#[cfg(not(target_arch = "x86_64"))]
fn top_nprobe_centroids_i16_scalar<const NPROBE: usize>(
    query: &[i16; 14],
    ds: &Dataset,
) -> [usize; NPROBE] {
    let k = ds.k;
    let mut dists = vec![0.0f32; k];
    for d in 0..14usize {
        let qd = query[d] as f32 * VECTOR_SCALE;
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

    sort_probe_results(&mut probes, &mut probe_dists);
    probes
}

#[cfg(not(target_arch = "x86_64"))]
fn scan_probes_i16_scalar(
    probes: &[usize],
    query: &[i16; 14],
    ds: &Dataset,
    top: &mut [(i32, u8); 5],
    worst_idx: &mut usize,
) {
    let blocks = ds.blocks.as_slice();
    let labels = ds.labels.as_slice();

    let mut last_ci = usize::MAX;
    for &ci in probes {
        if ci == last_ci {
            continue;
        }
        last_ci = ci;
        let start_block = ds.offsets[ci] as usize;
        let end_block = ds.offsets[ci + 1] as usize;
        for block_i in start_block..end_block {
            let block_base = block_i * 112;
            for slot in 0..8usize {
                if blocks[block_base + slot] == i16::MAX {
                    continue;
                }
                let mut dist = 0i32;
                for d in 0..14usize {
                    let diff = blocks[block_base + d * 8 + slot] as i32 - query[d] as i32;
                    dist += diff * diff;
                }
                let label = labels[block_i * 8 + slot];
                if dist < top[*worst_idx].0 {
                    top[*worst_idx] = (dist, label);
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::Dataset;
    use aligned_vec::{AVec, ConstAlign};

    fn synth(rows: &[([i16; 14], u8)]) -> Dataset {
        let len = rows.len();
        let padded_n = len.next_multiple_of(8);
        let n_blocks = padded_n / 8;
        let mut blocks: AVec<i16, ConstAlign<32>> = AVec::with_capacity(32, n_blocks * 112);
        let mut labels = Vec::with_capacity(padded_n);

        let mut row_buf = [[i16::MAX; 14]; 8];
        let mut label_buf = [0u8; 8];
        let mut buf_idx = 0usize;

        for (v, l) in rows {
            row_buf[buf_idx] = *v;
            label_buf[buf_idx] = *l;
            buf_idx += 1;
            if buf_idx == 8 {
                push_block(&mut blocks, &mut labels, &row_buf, &label_buf);
                row_buf = [[i16::MAX; 14]; 8];
                label_buf = [0u8; 8];
                buf_idx = 0;
            }
        }
        if buf_idx > 0 {
            push_block(&mut blocks, &mut labels, &row_buf, &label_buf);
        }

        let mut centroids: AVec<f32, ConstAlign<32>> = AVec::with_capacity(32, 14);
        for _ in 0..14 {
            centroids.push(0.0);
        }

        Dataset {
            blocks,
            labels,
            n: len,
            padded_n,
            centroids,
            offsets: vec![0, n_blocks as u32],
            k: 1,
        }
    }

    fn push_block(
        blocks: &mut AVec<i16, ConstAlign<32>>,
        labels: &mut Vec<u8>,
        rows: &[[i16; 14]; 8],
        label_buf: &[u8; 8],
    ) {
        for d in 0..14 {
            for k in 0..8 {
                blocks.push(rows[k][d]);
            }
        }
        labels.extend_from_slice(label_buf);
    }

    fn vec14(v: i16) -> [i16; 14] {
        [v; 14]
    }

    #[test]
    fn top5_todos_fraude() {
        let mut rows = vec![];
        for _ in 0..3 {
            rows.push((vec14(0), 1));
        }
        for _ in 0..2 {
            rows.push((vec14(100), 1));
        }
        for _ in 0..5 {
            rows.push((vec14(9000), 0));
        }
        let ds = synth(&rows);
        assert_eq!(knn5_fraud_count_ivf_i16(&vec14(0), &ds), 5);
    }

    #[test]
    fn top5_todos_legit() {
        let mut rows = vec![];
        for _ in 0..5 {
            rows.push((vec14(0), 0));
        }
        for _ in 0..5 {
            rows.push((vec14(9000), 1));
        }
        let ds = synth(&rows);
        assert_eq!(knn5_fraud_count_ivf_i16(&vec14(0), &ds), 0);
    }

    #[test]
    fn empate_3_2() {
        let rows = vec![
            (vec14(0), 1),
            (vec14(100), 1),
            (vec14(200), 1),
            (vec14(300), 0),
            (vec14(400), 0),
            (vec14(9000), 0),
            (vec14(9000), 0),
        ];
        let ds = synth(&rows);
        assert_eq!(knn5_fraud_count_ivf_i16(&vec14(0), &ds), 3);
    }

    #[test]
    fn padding_nao_entra_top5() {
        let rows = vec![(vec14(0), 1), (vec14(1000), 1), (vec14(2000), 0)];
        let ds = synth(&rows);
        assert_eq!(knn5_fraud_count_ivf_i16(&vec14(0), &ds), 2);
    }

    #[test]
    fn dataset_real_consulta() {
        let ds = Dataset::load_embedded().unwrap();
        let c = knn5_fraud_count_ivf_i16(&[0; 14], &ds);
        assert!(c <= 5);
    }

    #[test]
    fn consistencia_resultados() {
        let ds = Dataset::load_embedded().unwrap();
        let q1 = [0; 14];
        let q2 = [5000; 14];
        let a1 = knn5_fraud_count_ivf_i16(&q1, &ds);
        let _ = knn5_fraud_count_ivf_i16(&q2, &ds);
        let a3 = knn5_fraud_count_ivf_i16(&q1, &ds);
        assert_eq!(a1, a3);
    }
}
