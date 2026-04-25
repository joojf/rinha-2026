use crate::dataset::Dataset;

pub fn knn5_fraud_count(query: &[f32; 14], ds: &Dataset) -> u8 {
    let n = ds.padded_len;
    let mut dist = vec![0.0f32; n];

    for d in 0..14 {
        let q = query[d];
        let col = ds.dims[d].as_slice();
        for i in 0..n {
            let diff = col[i] - q;
            dist[i] += diff * diff;
        }
    }

    let mut top: [(f32, u8); 5] = [(f32::INFINITY, 0); 5];
    let mut worst_idx = 0usize;
    for i in 0..n {
        let di = dist[i];
        if di < top[worst_idx].0 {
            top[worst_idx] = (di, ds.labels[i]);
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
        Dataset { dims, labels, len, padded_len }
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
}
