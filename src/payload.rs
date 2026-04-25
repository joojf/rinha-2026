#![allow(dead_code)]

use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct Transaction {
    pub amount: f64,
    pub installments: u32,
    pub requested_at: String,
}

#[derive(Debug, Deserialize)]
pub struct Customer {
    pub avg_amount: f64,
    pub tx_count_24h: u32,
    pub known_merchants: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct Merchant {
    pub id: String,
    pub mcc: String,
    pub avg_amount: f64,
}

#[derive(Debug, Deserialize)]
pub struct Terminal {
    pub is_online: bool,
    pub card_present: bool,
    pub km_from_home: f64,
}

#[derive(Debug, Deserialize)]
pub struct LastTransaction {
    pub timestamp: String,
    pub km_from_current: f64,
}

#[derive(Debug, Deserialize)]
pub struct FraudRequest {
    pub id: String,
    pub transaction: Transaction,
    pub customer: Customer,
    pub merchant: Merchant,
    pub terminal: Terminal,
    pub last_transaction: Option<LastTransaction>,
}

pub fn parse(buf: &[u8]) -> Result<FraudRequest, sonic_rs::Error> {
    sonic_rs::from_slice(buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_payload_with_last_tx() {
        let raw = br#"{
            "id": "tx-3576980410",
            "transaction": { "amount": 384.88, "installments": 3, "requested_at": "2026-03-11T20:23:35Z" },
            "customer": { "avg_amount": 769.76, "tx_count_24h": 3, "known_merchants": ["MERC-009", "MERC-001"] },
            "merchant": { "id": "MERC-001", "mcc": "5912", "avg_amount": 298.95 },
            "terminal": { "is_online": false, "card_present": true, "km_from_home": 13.709 },
            "last_transaction": { "timestamp": "2026-03-11T14:58:35Z", "km_from_current": 18.862 }
        }"#;
        let req = parse(raw).unwrap();
        assert_eq!(req.id, "tx-3576980410");
        assert!((req.transaction.amount - 384.88).abs() < 1e-9);
        assert_eq!(req.transaction.installments, 3);
        assert_eq!(req.customer.tx_count_24h, 3);
        assert_eq!(req.merchant.mcc, "5912");
        assert!(!req.terminal.is_online);
        assert!(req.terminal.card_present);
        assert!(req.last_transaction.is_some());
    }

    #[test]
    fn parses_payload_null_last_tx() {
        let raw = br#"{
            "id": "tx-1329056812",
            "transaction": { "amount": 41.12, "installments": 2, "requested_at": "2026-03-11T18:45:53Z" },
            "customer": { "avg_amount": 82.24, "tx_count_24h": 3, "known_merchants": ["MERC-003"] },
            "merchant": { "id": "MERC-016", "mcc": "5411", "avg_amount": 60.25 },
            "terminal": { "is_online": false, "card_present": true, "km_from_home": 29.233 },
            "last_transaction": null
        }"#;
        let req = parse(raw).unwrap();
        assert_eq!(req.id, "tx-1329056812");
        assert!(req.last_transaction.is_none());
    }

    #[test]
    fn rejects_missing_field() {
        let raw = br#"{"id":"tx-1","transaction":{"amount":1.0,"installments":1,"requested_at":"2026-01-01T00:00:00Z"}}"#;
        assert!(parse(raw).is_err());
    }

    #[test]
    fn rejects_invalid_json() {
        assert!(parse(b"{not valid json}").is_err());
    }
}
