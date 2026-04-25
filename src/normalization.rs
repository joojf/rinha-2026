use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct Normalization {
    pub max_amount: f32,
    pub max_installments: f32,
    pub amount_vs_avg_ratio: f32,
    pub max_minutes: f32,
    pub max_km: f32,
    pub max_tx_count_24h: f32,
    pub max_merchant_avg_amount: f32,
}

impl Normalization {
    pub fn load_embedded() -> Self {
        let bytes = include_bytes!("../spec/resources/normalization.json");
        sonic_rs::from_slice(bytes).expect("normalization.json inválido")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn carrega_constantes() {
        let n = Normalization::load_embedded();
        assert_eq!(n.max_amount, 10000.0);
        assert_eq!(n.max_installments, 12.0);
        assert_eq!(n.amount_vs_avg_ratio, 10.0);
        assert_eq!(n.max_minutes, 1440.0);
        assert_eq!(n.max_km, 1000.0);
        assert_eq!(n.max_tx_count_24h, 20.0);
        assert_eq!(n.max_merchant_avg_amount, 10000.0);
    }
}
