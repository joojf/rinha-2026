use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct Transaction<'a> {
    pub amount: f64,
    pub installments: u32,
    #[serde(borrow)]
    pub requested_at: &'a str,
}

#[derive(Debug, Deserialize)]
pub struct Customer<'a> {
    pub avg_amount: f64,
    pub tx_count_24h: u32,
    #[serde(borrow)]
    pub known_merchants: Vec<&'a str>,
}

#[derive(Debug, Deserialize)]
pub struct Merchant<'a> {
    #[serde(borrow)]
    pub id: &'a str,
    #[serde(borrow)]
    pub mcc: &'a str,
    pub avg_amount: f64,
}

#[derive(Debug, Deserialize)]
pub struct Terminal {
    pub is_online: bool,
    pub card_present: bool,
    pub km_from_home: f64,
}

#[derive(Debug, Deserialize)]
pub struct LastTransaction<'a> {
    #[serde(borrow)]
    pub timestamp: &'a str,
    pub km_from_current: f64,
}

#[derive(Debug, Deserialize)]
pub struct FraudRequest<'a> {
    #[serde(borrow)]
    pub transaction: Transaction<'a>,
    #[serde(borrow)]
    pub customer: Customer<'a>,
    #[serde(borrow)]
    pub merchant: Merchant<'a>,
    pub terminal: Terminal,
    #[serde(borrow)]
    pub last_transaction: Option<LastTransaction<'a>>,
}

#[derive(Debug)]
pub enum ParseError {
    Json(sonic_rs::Error),
    BadTimestamp,
}

impl From<sonic_rs::Error> for ParseError {
    fn from(e: sonic_rs::Error) -> Self {
        ParseError::Json(e)
    }
}

pub fn parse(buf: &[u8]) -> Result<FraudRequest<'_>, ParseError> {
    let req: FraudRequest<'_> = sonic_rs::from_slice(buf)?;
    if !valid_iso_ts(&req.transaction.requested_at) {
        return Err(ParseError::BadTimestamp);
    }
    if let Some(lt) = &req.last_transaction {
        if !valid_iso_ts(&lt.timestamp) {
            return Err(ParseError::BadTimestamp);
        }
    }
    Ok(req)
}

fn valid_iso_ts(s: &str) -> bool {
    let b = s.as_bytes();
    if b.len() < 19 {
        return false;
    }
    b[4] == b'-'
        && b[7] == b'-'
        && (b[10] == b'T' || b[10] == b' ')
        && b[13] == b':'
        && b[16] == b':'
        && b[0..4].iter().all(|&c| c.is_ascii_digit())
        && b[5..7].iter().all(|&c| c.is_ascii_digit())
        && b[8..10].iter().all(|&c| c.is_ascii_digit())
        && b[11..13].iter().all(|&c| c.is_ascii_digit())
        && b[14..16].iter().all(|&c| c.is_ascii_digit())
        && b[17..19].iter().all(|&c| c.is_ascii_digit())
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

    #[test]
    fn rejects_short_requested_at() {
        let raw = br#"{
            "id": "tx-1",
            "transaction": { "amount": 1.0, "installments": 1, "requested_at": "2026-01-01" },
            "customer": { "avg_amount": 1.0, "tx_count_24h": 1, "known_merchants": [] },
            "merchant": { "id": "x", "mcc": "5411", "avg_amount": 1.0 },
            "terminal": { "is_online": false, "card_present": true, "km_from_home": 0.0 },
            "last_transaction": null
        }"#;
        assert!(parse(raw).is_err());
    }

    #[test]
    fn rejects_non_digit_requested_at() {
        let raw = br#"{
            "id": "tx-1",
            "transaction": { "amount": 1.0, "installments": 1, "requested_at": "abcd-01-01T00:00:00Z" },
            "customer": { "avg_amount": 1.0, "tx_count_24h": 1, "known_merchants": [] },
            "merchant": { "id": "x", "mcc": "5411", "avg_amount": 1.0 },
            "terminal": { "is_online": false, "card_present": true, "km_from_home": 0.0 },
            "last_transaction": null
        }"#;
        assert!(parse(raw).is_err());
    }

    #[test]
    fn rejects_short_last_tx_timestamp() {
        let raw = br#"{
            "id": "tx-1",
            "transaction": { "amount": 1.0, "installments": 1, "requested_at": "2026-01-01T00:00:00Z" },
            "customer": { "avg_amount": 1.0, "tx_count_24h": 1, "known_merchants": [] },
            "merchant": { "id": "x", "mcc": "5411", "avg_amount": 1.0 },
            "terminal": { "is_online": false, "card_present": true, "km_from_home": 0.0 },
            "last_transaction": { "timestamp": "bad", "km_from_current": 0.0 }
        }"#;
        assert!(parse(raw).is_err());
    }

    #[test]
    fn accepts_space_separator_timestamp() {
        let raw = br#"{
            "id": "tx-1",
            "transaction": { "amount": 1.0, "installments": 1, "requested_at": "2026-01-01 00:00:00Z" },
            "customer": { "avg_amount": 1.0, "tx_count_24h": 1, "known_merchants": [] },
            "merchant": { "id": "x", "mcc": "5411", "avg_amount": 1.0 },
            "terminal": { "is_online": false, "card_present": true, "km_from_home": 0.0 },
            "last_transaction": null
        }"#;
        assert!(parse(raw).is_ok());
    }
}
