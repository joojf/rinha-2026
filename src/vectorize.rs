use crate::{
    mcc_risk::MccRisk,
    normalization::Normalization,
    payload::FraudRequest,
};

pub fn vectorize(req: &FraudRequest, norm: &Normalization, mcc: &MccRisk) -> [f32; 14] {
    let ts = req.transaction.requested_at.as_bytes();
    let hour = parse_u8_2(&ts[11..13]);
    let y = parse_u16_4(&ts[0..4]);
    let m = parse_u8_2(&ts[5..7]);
    let d = parse_u8_2(&ts[8..10]);

    let (dim5, dim6) = match &req.last_transaction {
        None => (-1.0_f32, -1.0_f32),
        Some(lt) => {
            let t1 = parse_iso_epoch_secs(&req.transaction.requested_at);
            let t0 = parse_iso_epoch_secs(&lt.timestamp);
            let minutes = (t1 - t0).max(0) as f32 / 60.0;
            (
                clamp01(minutes / norm.max_minutes),
                clamp01(lt.km_from_current as f32 / norm.max_km),
            )
        }
    };

    let unknown = !req
        .customer
        .known_merchants
        .iter()
        .any(|k| k == &req.merchant.id);

    [
        clamp01(req.transaction.amount as f32 / norm.max_amount),
        clamp01(req.transaction.installments as f32 / norm.max_installments),
        clamp01(
            (req.transaction.amount as f32 / req.customer.avg_amount as f32)
                / norm.amount_vs_avg_ratio,
        ),
        hour as f32 / 23.0,
        weekday_mon0(y as u32, m as u32, d as u32) as f32 / 6.0,
        dim5,
        dim6,
        clamp01(req.terminal.km_from_home as f32 / norm.max_km),
        clamp01(req.customer.tx_count_24h as f32 / norm.max_tx_count_24h),
        if req.terminal.is_online { 1.0 } else { 0.0 },
        if req.terminal.card_present { 1.0 } else { 0.0 },
        if unknown { 1.0 } else { 0.0 },
        mcc.lookup(&req.merchant.mcc),
        clamp01(req.merchant.avg_amount as f32 / norm.max_merchant_avg_amount),
    ]
}

#[inline]
fn clamp01(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

fn weekday_mon0(y: u32, m: u32, d: u32) -> u8 {
    const T: [u32; 12] = [0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4];
    let y = if m < 3 { y - 1 } else { y };
    let w = (y + y / 4 - y / 100 + y / 400 + T[(m - 1) as usize] + d) % 7;
    if w == 0 { 6 } else { (w - 1) as u8 }
}

fn parse_iso_epoch_secs(s: &str) -> i64 {
    let b = s.as_bytes();
    let y = parse_u16_4(&b[0..4]) as i64;
    let m = parse_u8_2(&b[5..7]) as i64;
    let d = parse_u8_2(&b[8..10]) as i64;
    let h = parse_u8_2(&b[11..13]) as i64;
    let min = parse_u8_2(&b[14..16]) as i64;
    let sec = parse_u8_2(&b[17..19]) as i64;
    let yy = if m <= 2 { y - 1 } else { y };
    let era = yy.div_euclid(400);
    let yoe = yy - era * 400;
    let doy = (153 * (m + if m > 2 { -3 } else { 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146097 + doe - 719468;
    days * 86400 + h * 3600 + min * 60 + sec
}

fn parse_u8_2(b: &[u8]) -> u8 {
    (b[0] - b'0') * 10 + (b[1] - b'0')
}

fn parse_u16_4(b: &[u8]) -> u16 {
    (b[0] - b'0') as u16 * 1000
        + (b[1] - b'0') as u16 * 100
        + (b[2] - b'0') as u16 * 10
        + (b[3] - b'0') as u16
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{mcc_risk::MccRisk, normalization::Normalization, payload::*};

    fn norm() -> Normalization {
        Normalization::load_embedded()
    }
    fn mcc() -> MccRisk {
        MccRisk::load_embedded()
    }

    fn req_legit() -> FraudRequest {
        FraudRequest {
            id: "tx-1329056812".into(),
            transaction: Transaction {
                amount: 41.12,
                installments: 2,
                requested_at: "2026-03-11T18:45:53Z".into(),
            },
            customer: Customer {
                avg_amount: 82.24,
                tx_count_24h: 3,
                known_merchants: vec!["MERC-003".into(), "MERC-016".into()],
            },
            merchant: Merchant {
                id: "MERC-016".into(),
                mcc: "5411".into(),
                avg_amount: 60.25,
            },
            terminal: Terminal {
                is_online: false,
                card_present: true,
                km_from_home: 29.23,
            },
            last_transaction: None,
        }
    }

    fn req_fraud() -> FraudRequest {
        FraudRequest {
            id: "tx-3330991687".into(),
            transaction: Transaction {
                amount: 9505.97,
                installments: 10,
                requested_at: "2026-03-14T05:15:12Z".into(),
            },
            customer: Customer {
                avg_amount: 81.28,
                tx_count_24h: 20,
                known_merchants: vec![
                    "MERC-008".into(),
                    "MERC-007".into(),
                    "MERC-005".into(),
                ],
            },
            merchant: Merchant {
                id: "MERC-068".into(),
                mcc: "7802".into(),
                avg_amount: 54.86,
            },
            terminal: Terminal {
                is_online: false,
                card_present: true,
                km_from_home: 952.27,
            },
            last_transaction: None,
        }
    }

    fn near(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-3
    }

    #[test]
    fn spec_example_legit() {
        let expected = [0.0041, 0.1667, 0.05, 0.7826, 0.3333, -1.0, -1.0,
                        0.0292, 0.15, 0.0, 1.0, 0.0, 0.15, 0.006];
        let got = vectorize(&req_legit(), &norm(), &mcc());
        for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(near(*g, *e), "dim {i}: got {g}, expected {e}");
        }
    }

    #[test]
    fn spec_example_fraud() {
        let expected = [0.9506, 0.8333, 1.0, 0.2174, 0.8333, -1.0, -1.0,
                        0.9523, 1.0, 0.0, 1.0, 1.0, 0.75, 0.0055];
        let got = vectorize(&req_fraud(), &norm(), &mcc());
        for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(near(*g, *e), "dim {i}: got {g}, expected {e}");
        }
    }

    #[test]
    fn dim0_amount_clamp() {
        let mut r = req_legit();
        r.transaction.amount = 0.0;
        assert_eq!(vectorize(&r, &norm(), &mcc())[0], 0.0);
        r.transaction.amount = 99999.0;
        assert_eq!(vectorize(&r, &norm(), &mcc())[0], 1.0);
    }

    #[test]
    fn dim1_installments() {
        let mut r = req_legit();
        r.transaction.installments = 12;
        assert!(near(vectorize(&r, &norm(), &mcc())[1], 1.0));
        r.transaction.installments = 6;
        assert!(near(vectorize(&r, &norm(), &mcc())[1], 0.5));
    }

    #[test]
    fn dim2_amount_vs_avg_clamp() {
        let mut r = req_legit();
        r.transaction.amount = 9999.0;
        r.customer.avg_amount = 1.0;
        assert_eq!(vectorize(&r, &norm(), &mcc())[2], 1.0);
    }

    #[test]
    fn dim3_hour() {
        let mut r = req_legit();
        r.transaction.requested_at = "2026-01-01T00:00:00Z".into();
        assert_eq!(vectorize(&r, &norm(), &mcc())[3], 0.0);
        r.transaction.requested_at = "2026-01-01T23:00:00Z".into();
        assert!(near(vectorize(&r, &norm(), &mcc())[3], 1.0));
    }

    #[test]
    fn dim4_weekday() {
        let mut r = req_legit();
        r.transaction.requested_at = "2026-03-09T10:00:00Z".into();
        assert_eq!(vectorize(&r, &norm(), &mcc())[4], 0.0);
        r.transaction.requested_at = "2026-03-15T10:00:00Z".into();
        assert!(near(vectorize(&r, &norm(), &mcc())[4], 1.0));
        r.transaction.requested_at = "2026-03-11T18:45:53Z".into();
        assert!(near(vectorize(&r, &norm(), &mcc())[4], 2.0 / 6.0));
        r.transaction.requested_at = "2026-03-14T05:15:12Z".into();
        assert!(near(vectorize(&r, &norm(), &mcc())[4], 5.0 / 6.0));
    }

    #[test]
    fn dim5_minutes_null_is_sentinel() {
        let r = req_legit();
        assert_eq!(vectorize(&r, &norm(), &mcc())[5], -1.0);
    }

    #[test]
    fn dim5_minutes_clamped() {
        let mut r = req_legit();
        r.transaction.requested_at = "2026-03-11T18:45:53Z".into();
        r.last_transaction = Some(crate::payload::LastTransaction {
            timestamp: "2026-03-11T18:45:53Z".into(),
            km_from_current: 0.0,
        });
        assert_eq!(vectorize(&r, &norm(), &mcc())[5], 0.0);
        r.last_transaction = Some(crate::payload::LastTransaction {
            timestamp: "2026-03-10T18:45:53Z".into(),
            km_from_current: 0.0,
        });
        assert!(near(vectorize(&r, &norm(), &mcc())[5], 1.0));
        r.last_transaction = Some(crate::payload::LastTransaction {
            timestamp: "2026-03-01T00:00:00Z".into(),
            km_from_current: 0.0,
        });
        assert_eq!(vectorize(&r, &norm(), &mcc())[5], 1.0);
    }

    #[test]
    fn dim6_km_last_null_is_sentinel() {
        let r = req_legit();
        assert_eq!(vectorize(&r, &norm(), &mcc())[6], -1.0);
    }

    #[test]
    fn dim6_km_last_clamp() {
        let mut r = req_legit();
        r.last_transaction = Some(crate::payload::LastTransaction {
            timestamp: "2026-03-11T10:00:00Z".into(),
            km_from_current: 500.0,
        });
        assert!(near(vectorize(&r, &norm(), &mcc())[6], 0.5));
        r.last_transaction = Some(crate::payload::LastTransaction {
            timestamp: "2026-03-11T10:00:00Z".into(),
            km_from_current: 9999.0,
        });
        assert_eq!(vectorize(&r, &norm(), &mcc())[6], 1.0);
    }

    #[test]
    fn dim7_km_from_home() {
        let mut r = req_legit();
        r.terminal.km_from_home = 0.0;
        assert_eq!(vectorize(&r, &norm(), &mcc())[7], 0.0);
        r.terminal.km_from_home = 2000.0;
        assert_eq!(vectorize(&r, &norm(), &mcc())[7], 1.0);
    }

    #[test]
    fn dim8_tx_count() {
        let mut r = req_legit();
        r.customer.tx_count_24h = 20;
        assert!(near(vectorize(&r, &norm(), &mcc())[8], 1.0));
        r.customer.tx_count_24h = 0;
        assert_eq!(vectorize(&r, &norm(), &mcc())[8], 0.0);
    }

    #[test]
    fn dim9_is_online() {
        let mut r = req_legit();
        r.terminal.is_online = true;
        assert_eq!(vectorize(&r, &norm(), &mcc())[9], 1.0);
        r.terminal.is_online = false;
        assert_eq!(vectorize(&r, &norm(), &mcc())[9], 0.0);
    }

    #[test]
    fn dim10_card_present() {
        let mut r = req_legit();
        r.terminal.card_present = false;
        assert_eq!(vectorize(&r, &norm(), &mcc())[10], 0.0);
        r.terminal.card_present = true;
        assert_eq!(vectorize(&r, &norm(), &mcc())[10], 1.0);
    }

    #[test]
    fn dim11_unknown_merchant() {
        let mut r = req_legit();
        assert_eq!(vectorize(&r, &norm(), &mcc())[11], 0.0);
        r.merchant.id = "MERC-999".into();
        assert_eq!(vectorize(&r, &norm(), &mcc())[11], 1.0);
    }

    #[test]
    fn dim12_mcc_risk() {
        let mut r = req_legit();
        r.merchant.mcc = "7995".into();
        assert!(near(vectorize(&r, &norm(), &mcc())[12], 0.85));
        r.merchant.mcc = "9999".into();
        assert!(near(vectorize(&r, &norm(), &mcc())[12], 0.5));
    }

    #[test]
    fn dim13_merchant_avg_amount() {
        let mut r = req_legit();
        r.merchant.avg_amount = 5000.0;
        assert!(near(vectorize(&r, &norm(), &mcc())[13], 0.5));
        r.merchant.avg_amount = 20000.0;
        assert_eq!(vectorize(&r, &norm(), &mcc())[13], 1.0);
    }
}
