use crate::{dataset::Dataset, search};

#[derive(Debug)]
pub enum ScoreError {
    MissingField,
    InvalidJson,
    BadTimestamp,
}

pub fn score_body(buf: &[u8], ds: &Dataset) -> Result<u8, ScoreError> {
    let q = vectorize_body_i16(buf)?;
    Ok(search::knn5_fraud_count_ivf_i16(&q, ds))
}

pub fn vectorize_body_i16(buf: &[u8]) -> Result<[i16; 14], ScoreError> {
    let transaction = value_start(buf, 0, b"\"transaction\"")?;
    let amount = number_value(buf, transaction, b"\"amount\"")?;
    let installments = number_value(buf, transaction, b"\"installments\"")?;
    let requested_at = string_value(buf, transaction, b"\"requested_at\"")?;

    let customer = value_start(buf, 0, b"\"customer\"")?;
    let customer_avg_amount = number_value(buf, customer, b"\"avg_amount\"")?;
    let tx_count_24h = number_value(buf, customer, b"\"tx_count_24h\"")?;
    let known_merchants = array_value(buf, customer, b"\"known_merchants\"")?;

    let merchant = value_start(buf, 0, b"\"merchant\"")?;
    let merchant_id = string_value(buf, merchant, b"\"id\"")?;
    let mcc = string_value(buf, merchant, b"\"mcc\"")?;
    let merchant_avg_amount = number_value(buf, merchant, b"\"avg_amount\"")?;

    let terminal = value_start(buf, 0, b"\"terminal\"")?;
    let is_online = bool_value(buf, terminal, b"\"is_online\"")?;
    let card_present = bool_value(buf, terminal, b"\"card_present\"")?;
    let km_from_home = number_value(buf, terminal, b"\"km_from_home\"")?;

    let last_transaction = value_start(buf, 0, b"\"last_transaction\"")?;
    let (dim5, dim6) = if matches!(buf.get(last_transaction), Some(b'n')) {
        (-10000, -10000)
    } else {
        let timestamp = string_value(buf, last_transaction, b"\"timestamp\"")?;
        let km_from_current = number_value(buf, last_transaction, b"\"km_from_current\"")?;
        let minutes =
            (parse_iso_epoch_secs(requested_at)? - parse_iso_epoch_secs(timestamp)?).max(0) as f32
                / 60.0;
        (q_clamp(minutes / 1440.0), q_clamp(km_from_current / 1000.0))
    };

    let hour = parse_u8_2(&requested_at[11..13])?;
    let y = parse_u16_4(&requested_at[0..4])?;
    let m = parse_u8_2(&requested_at[5..7])?;
    let d = parse_u8_2(&requested_at[8..10])?;
    let unknown = !array_contains_string(known_merchants, merchant_id);

    Ok([
        q_clamp(amount / 10000.0),
        q_clamp(installments / 12.0),
        q_clamp((amount / customer_avg_amount) / 10.0),
        q_round(hour as f32 / 23.0),
        q_round(weekday_mon0(y as u32, m as u32, d as u32) as f32 / 6.0),
        dim5,
        dim6,
        q_clamp(km_from_home / 1000.0),
        q_clamp(tx_count_24h / 20.0),
        if is_online { 10000 } else { 0 },
        if card_present { 10000 } else { 0 },
        if unknown { 10000 } else { 0 },
        q_mcc(mcc),
        q_clamp(merchant_avg_amount / 10000.0),
    ])
}

#[inline]
fn q_clamp(x: f32) -> i16 {
    q_round(x.clamp(0.0, 1.0))
}

#[inline]
fn q_round(x: f32) -> i16 {
    (x * 10000.0).round() as i16
}

fn q_mcc(mcc: &[u8]) -> i16 {
    match mcc {
        b"5411" => 1500,
        b"5812" => 3000,
        b"5912" => 2000,
        b"5944" => 4500,
        b"7801" => 8000,
        b"7802" => 7500,
        b"7995" => 8500,
        b"4511" => 3500,
        b"5311" => 2500,
        b"5999" => 5000,
        _ => 5000,
    }
}

fn value_start(buf: &[u8], start: usize, key: &[u8]) -> Result<usize, ScoreError> {
    let mut i = find_key(buf, start, key)? + key.len();
    i = skip_ws(buf, i);
    if buf.get(i) != Some(&b':') {
        return Err(ScoreError::InvalidJson);
    }
    Ok(skip_ws(buf, i + 1))
}

fn string_value<'a>(buf: &'a [u8], start: usize, key: &[u8]) -> Result<&'a [u8], ScoreError> {
    let i = value_start(buf, start, key)?;
    if buf.get(i) != Some(&b'"') {
        return Err(ScoreError::InvalidJson);
    }
    let s = i + 1;
    let mut e = s;
    while e < buf.len() && buf[e] != b'"' {
        e += 1;
    }
    if e >= buf.len() {
        return Err(ScoreError::InvalidJson);
    }
    Ok(&buf[s..e])
}

fn array_value<'a>(buf: &'a [u8], start: usize, key: &[u8]) -> Result<&'a [u8], ScoreError> {
    let i = value_start(buf, start, key)?;
    if buf.get(i) != Some(&b'[') {
        return Err(ScoreError::InvalidJson);
    }
    let mut e = i + 1;
    while e < buf.len() && buf[e] != b']' {
        e += 1;
    }
    if e >= buf.len() {
        return Err(ScoreError::InvalidJson);
    }
    Ok(&buf[i + 1..e])
}

fn bool_value(buf: &[u8], start: usize, key: &[u8]) -> Result<bool, ScoreError> {
    let i = value_start(buf, start, key)?;
    if buf.get(i..i + 4) == Some(b"true") {
        Ok(true)
    } else if buf.get(i..i + 5) == Some(b"false") {
        Ok(false)
    } else {
        Err(ScoreError::InvalidJson)
    }
}

fn number_value(buf: &[u8], start: usize, key: &[u8]) -> Result<f32, ScoreError> {
    let i = value_start(buf, start, key)?;
    parse_number(buf, i)
}

fn parse_number(buf: &[u8], mut i: usize) -> Result<f32, ScoreError> {
    let mut sign = 1.0f32;
    if buf.get(i) == Some(&b'-') {
        sign = -1.0;
        i += 1;
    }

    let mut int = 0.0f32;
    let mut seen = false;
    while let Some(c @ b'0'..=b'9') = buf.get(i).copied() {
        seen = true;
        int = int * 10.0 + (c - b'0') as f32;
        i += 1;
    }

    let mut frac = 0.0f32;
    if buf.get(i) == Some(&b'.') {
        i += 1;
        let mut scale = 0.1f32;
        while let Some(c @ b'0'..=b'9') = buf.get(i).copied() {
            seen = true;
            frac += (c - b'0') as f32 * scale;
            scale *= 0.1;
            i += 1;
        }
    }

    if !seen {
        return Err(ScoreError::InvalidJson);
    }
    Ok(sign * (int + frac))
}

fn find_key(buf: &[u8], start: usize, key: &[u8]) -> Result<usize, ScoreError> {
    let first = key[0];
    let last = buf.len().saturating_sub(key.len());
    let mut i = start;
    while i <= last {
        if buf[i] == first && &buf[i..i + key.len()] == key {
            return Ok(i);
        }
        i += 1;
    }
    Err(ScoreError::MissingField)
}

fn skip_ws(buf: &[u8], mut i: usize) -> usize {
    while matches!(buf.get(i), Some(b' ' | b'\n' | b'\r' | b'\t')) {
        i += 1;
    }
    i
}

fn array_contains_string(array: &[u8], needle: &[u8]) -> bool {
    let mut i = 0usize;
    while i < array.len() {
        if array[i] == b'"' {
            let s = i + 1;
            let mut e = s;
            while e < array.len() && array[e] != b'"' {
                e += 1;
            }
            if e <= array.len() && &array[s..e] == needle {
                return true;
            }
            i = e.saturating_add(1);
        } else {
            i += 1;
        }
    }
    false
}

fn parse_iso_epoch_secs(s: &[u8]) -> Result<i64, ScoreError> {
    if s.len() < 19 {
        return Err(ScoreError::BadTimestamp);
    }
    let y = parse_u16_4(&s[0..4])? as i64;
    let m = parse_u8_2(&s[5..7])? as i64;
    let d = parse_u8_2(&s[8..10])? as i64;
    let h = parse_u8_2(&s[11..13])? as i64;
    let min = parse_u8_2(&s[14..16])? as i64;
    let sec = parse_u8_2(&s[17..19])? as i64;
    let yy = if m <= 2 { y - 1 } else { y };
    let era = yy.div_euclid(400);
    let yoe = yy - era * 400;
    let doy = (153 * (m + if m > 2 { -3 } else { 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146097 + doe - 719468;
    Ok(days * 86400 + h * 3600 + min * 60 + sec)
}

fn parse_u8_2(b: &[u8]) -> Result<u8, ScoreError> {
    if !b[0].is_ascii_digit() || !b[1].is_ascii_digit() {
        return Err(ScoreError::BadTimestamp);
    }
    Ok((b[0] - b'0') * 10 + (b[1] - b'0'))
}

fn parse_u16_4(b: &[u8]) -> Result<u16, ScoreError> {
    if !b.iter().all(|c| c.is_ascii_digit()) {
        return Err(ScoreError::BadTimestamp);
    }
    Ok((b[0] - b'0') as u16 * 1000
        + (b[1] - b'0') as u16 * 100
        + (b[2] - b'0') as u16 * 10
        + (b[3] - b'0') as u16)
}

fn weekday_mon0(y: u32, m: u32, d: u32) -> u8 {
    const T: [u32; 12] = [0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4];
    let y = if m < 3 { y - 1 } else { y };
    let w = (y + y / 4 - y / 100 + y / 400 + T[(m - 1) as usize] + d) % 7;
    if w == 0 { 6 } else { (w - 1) as u8 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vectoriza_legit_reference() {
        let raw = br#"{
            "id":"tx-1329056812",
            "transaction":{"amount":41.12,"installments":2,"requested_at":"2026-03-11T18:45:53Z"},
            "customer":{"avg_amount":82.24,"tx_count_24h":3,"known_merchants":["MERC-003","MERC-016"]},
            "merchant":{"id":"MERC-016","mcc":"5411","avg_amount":60.25},
            "terminal":{"is_online":false,"card_present":true,"km_from_home":29.23},
            "last_transaction":null
        }"#;
        assert_eq!(
            vectorize_body_i16(raw).unwrap(),
            [41, 1667, 500, 7826, 3333, -10000, -10000, 292, 1500, 0, 10000, 0, 1500, 60]
        );
    }

    #[test]
    fn vectoriza_fraud_reference() {
        let raw = br#"{
            "id":"tx-3330991687",
            "transaction":{"amount":9505.97,"installments":10,"requested_at":"2026-03-14T05:15:12Z"},
            "customer":{"avg_amount":81.28,"tx_count_24h":20,"known_merchants":["MERC-008","MERC-007","MERC-005"]},
            "merchant":{"id":"MERC-068","mcc":"7802","avg_amount":54.86},
            "terminal":{"is_online":false,"card_present":true,"km_from_home":952.2745933273},
            "last_transaction":null
        }"#;
        assert_eq!(
            vectorize_body_i16(raw).unwrap(),
            [9506, 8333, 10000, 2174, 8333, -10000, -10000, 9523, 10000, 0, 10000, 10000, 7500, 55]
        );
    }

    #[test]
    fn vectoriza_last_transaction() {
        let raw = br#"{
            "id":"tx-edge",
            "transaction":{"amount":1187.67,"installments":6,"requested_at":"2026-03-18T13:53:03Z"},
            "customer":{"avg_amount":360.36,"tx_count_24h":5,"known_merchants":["MERC-003","MERC-002"]},
            "merchant":{"id":"MERC-030","mcc":"5944","avg_amount":269.09},
            "terminal":{"is_online":true,"card_present":false,"km_from_home":390.2943426395},
            "last_transaction":{"timestamp":"2026-03-18T12:18:03Z","km_from_current":158.1653734525}
        }"#;
        assert_eq!(
            vectorize_body_i16(raw).unwrap(),
            [1188, 5000, 3296, 5652, 3333, 660, 1582, 3903, 2500, 10000, 0, 10000, 4500, 269]
        );
    }
}
