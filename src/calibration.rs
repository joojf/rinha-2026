pub fn calibrate_fraud_count(fraud_count: u8, query: &[f32; 14]) -> u8 {
    if fraud_count == 0 {
        if query[2] >= 0.15 { 3 } else { 0 }
    } else if fraud_count < 3 {
        3
    } else {
        fraud_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn promotes_ambiguous_votes() {
        let q = [0.0; 14];
        assert_eq!(calibrate_fraud_count(1, &q), 3);
        assert_eq!(calibrate_fraud_count(2, &q), 3);
    }

    #[test]
    fn promotes_zero_votes_only_when_amount_vs_avg_is_high() {
        let mut q = [0.0; 14];
        assert_eq!(calibrate_fraud_count(0, &q), 0);
        q[2] = 0.15;
        assert_eq!(calibrate_fraud_count(0, &q), 3);
    }

    #[test]
    fn keeps_already_fraudulent_votes() {
        let q = [0.0; 14];
        assert_eq!(calibrate_fraud_count(3, &q), 3);
        assert_eq!(calibrate_fraud_count(4, &q), 4);
        assert_eq!(calibrate_fraud_count(5, &q), 5);
    }
}
