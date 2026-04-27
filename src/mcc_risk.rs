pub struct MccRisk;

impl MccRisk {
    pub fn load_embedded() -> Self {
        MccRisk
    }

    pub fn lookup(&self, mcc: &str) -> f32 {
        match mcc.as_bytes() {
            b"5411" => 0.15,
            b"5812" => 0.30,
            b"5912" => 0.20,
            b"5944" => 0.45,
            b"7801" => 0.80,
            b"7802" => 0.75,
            b"7995" => 0.85,
            b"4511" => 0.35,
            b"5311" => 0.25,
            b"5999" => 0.50,
            _ => 0.5,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookup_conhecido() {
        let m = MccRisk::load_embedded();
        assert!((m.lookup("5411") - 0.15).abs() < 1e-6);
        assert!((m.lookup("7995") - 0.85).abs() < 1e-6);
    }

    #[test]
    fn lookup_desconhecido_retorna_default() {
        let m = MccRisk::load_embedded();
        assert_eq!(m.lookup("0000"), 0.5);
    }

    #[test]
    fn lookup_mcc_tamanho_errado() {
        let m = MccRisk::load_embedded();
        assert_eq!(m.lookup("12"), 0.5);
        assert_eq!(m.lookup(""), 0.5);
    }
}
