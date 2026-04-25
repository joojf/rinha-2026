use std::collections::HashMap;

pub struct MccRisk(HashMap<[u8; 4], f32>);

impl MccRisk {
    pub fn load_embedded() -> Self {
        let bytes = include_bytes!("../spec/resources/mcc_risk.json");
        let raw: HashMap<String, f32> =
            sonic_rs::from_slice(bytes).expect("mcc_risk.json inválido");
        let map = raw
            .into_iter()
            .filter_map(|(k, v)| {
                let b = k.as_bytes();
                if b.len() == 4 {
                    Some(([b[0], b[1], b[2], b[3]], v))
                } else {
                    None
                }
            })
            .collect();
        MccRisk(map)
    }

    pub fn lookup(&self, mcc: &str) -> f32 {
        let b = mcc.as_bytes();
        if b.len() == 4 {
            *self.0.get(&[b[0], b[1], b[2], b[3]]).unwrap_or(&0.5)
        } else {
            0.5
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
