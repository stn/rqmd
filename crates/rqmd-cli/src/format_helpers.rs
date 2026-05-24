//! Display formatters: bytes, durations, RFC3339 timestamps.
//!
//! Maps to qmd helpers `formatBytes`, `formatTimeAgo`, `formatLsTime` in
//! `src/cli/qmd.ts` (lines 370–391, 1501–1518).

use std::time::{SystemTime, UNIX_EPOCH};

pub fn format_bytes(bytes: u64) -> String {
    const KB: f64 = 1024.0;
    const MB: f64 = KB * 1024.0;
    const GB: f64 = MB * 1024.0;
    let b = bytes as f64;
    if b < KB {
        format!("{bytes} B")
    } else if b < MB {
        format!("{:.1} KB", b / KB)
    } else if b < GB {
        format!("{:.1} MB", b / MB)
    } else {
        format!("{:.1} GB", b / GB)
    }
}

/// Human-readable elapsed/ETA. Port of qmd `formatETA` (`src/cli/qmd.ts:311`):
/// `<60s` → `Ns`; `<3600s` → `Nm Ns`; else `Nh Nm`. Rounding matches JS
/// `Math.round`/`Math.floor` for the non-negative inputs used here.
pub fn format_eta(seconds: f64) -> String {
    if seconds < 60.0 {
        format!("{}s", seconds.round() as i64)
    } else if seconds < 3600.0 {
        format!(
            "{}m {}s",
            (seconds / 60.0).floor() as i64,
            (seconds % 60.0).round() as i64
        )
    } else {
        format!(
            "{}h {}m",
            (seconds / 3600.0).floor() as i64,
            ((seconds % 3600.0) / 60.0).floor() as i64
        )
    }
}

/// Group an integer with `,` thousands separators. Port of qmd `formatCount`
/// (`src/cli/qmd.ts:3321`), which uses `n.toLocaleString("en-US")`.
pub fn format_count(n: usize) -> String {
    let digits = n.to_string();
    let bytes = digits.as_bytes();
    let len = bytes.len();
    let mut out = String::with_capacity(len + len / 3);
    for (i, b) in bytes.iter().enumerate() {
        if i > 0 && (len - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(*b as char);
    }
    out
}

/// Shorten a model URI for display. Port of qmd `shortModelName`
/// (`src/cli/qmd.ts:3325`): `hf:` URIs collapse to the last `/` segment;
/// otherwise truncate to 56 chars (`53 + "..."`). Char-based truncation is
/// boundary-safe and identical to qmd for the ASCII URIs in practice.
pub fn short_model_name(model: &str) -> String {
    if model.starts_with("hf:") {
        let last = model.rsplit('/').next().unwrap_or("");
        return if last.is_empty() {
            model.to_string()
        } else {
            last.to_string()
        };
    }
    if model.chars().count() > 56 {
        let prefix: String = model.chars().take(53).collect();
        format!("{prefix}...")
    } else {
        model.to_string()
    }
}

/// Parse an RFC3339-ish timestamp (`YYYY-MM-DDTHH:MM:SS[.sss]Z` or with offset)
/// to seconds-since-epoch. Returns `None` on parse failure.
///
/// We hand-roll the parser to avoid pulling in `chrono`/`time`; rqmd-core uses
/// the same approach.
pub fn parse_rfc3339_to_epoch(s: &str) -> Option<i64> {
    let bytes = s.as_bytes();
    if bytes.len() < 19 {
        return None;
    }
    let year: i32 = s.get(0..4)?.parse().ok()?;
    let month: u32 = s.get(5..7)?.parse().ok()?;
    let day: u32 = s.get(8..10)?.parse().ok()?;
    let hour: i64 = s.get(11..13)?.parse().ok()?;
    let minute: i64 = s.get(14..16)?.parse().ok()?;
    let second: i64 = s.get(17..19)?.parse().ok()?;

    let days = days_from_civil(year as i64, month, day);
    Some(days * 86_400 + hour * 3_600 + minute * 60 + second)
}

pub fn now_epoch() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// Human "time ago" for an RFC3339 string. Returns `"never"` on parse failure.
pub fn format_time_ago(rfc3339: &str) -> String {
    let Some(then) = parse_rfc3339_to_epoch(rfc3339) else {
        return "never".to_string();
    };
    let diff = now_epoch().saturating_sub(then);
    if diff < 60 {
        format!("{diff}s ago")
    } else if diff < 3_600 {
        format!("{}m ago", diff / 60)
    } else if diff < 86_400 {
        format!("{}h ago", diff / 3_600)
    } else {
        format!("{}d ago", diff / 86_400)
    }
}

/// `ls -l`-style date string: `Mon DD HH:MM` if within 6 months, else `Mon DD  YYYY`.
pub fn format_ls_time(rfc3339: &str) -> String {
    let Some(epoch) = parse_rfc3339_to_epoch(rfc3339) else {
        return "?".to_string();
    };
    let (year, month, day, hour, minute) = civil_from_epoch(epoch);
    let months = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    let mon = months[(month - 1) as usize];
    let now = now_epoch();
    let six_months = 6 * 30 * 86_400;
    if now - epoch > six_months {
        format!("{mon} {day:>2}  {year}")
    } else {
        format!("{mon} {day:>2} {hour:02}:{minute:02}")
    }
}

// ----- Howard Hinnant's days_from_civil (public domain) -----

fn days_from_civil(y: i64, m: u32, d: u32) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let m = m as i64;
    let d = d as i64;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

fn civil_from_epoch(epoch: i64) -> (i64, u32, u32, u32, u32) {
    let days = epoch.div_euclid(86_400);
    let tod = epoch.rem_euclid(86_400);
    let hour = (tod / 3_600) as u32;
    let minute = ((tod % 3_600) / 60) as u32;

    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d, hour, minute)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_eta_matches_qmd_buckets() {
        assert_eq!(format_eta(0.0), "0s");
        assert_eq!(format_eta(5.4), "5s");
        assert_eq!(format_eta(59.0), "59s");
        assert_eq!(format_eta(95.0), "1m 35s"); // 1m 35s
        assert_eq!(format_eta(3599.0), "59m 59s");
        assert_eq!(format_eta(3600.0), "1h 0m");
        assert_eq!(format_eta(7384.0), "2h 3m");
    }

    #[test]
    fn format_count_groups_thousands() {
        assert_eq!(format_count(0), "0");
        assert_eq!(format_count(42), "42");
        assert_eq!(format_count(999), "999");
        assert_eq!(format_count(1_000), "1,000");
        assert_eq!(format_count(12_345), "12,345");
        assert_eq!(format_count(1_234_567), "1,234,567");
    }

    #[test]
    fn short_model_name_collapses_hf_and_truncates() {
        assert_eq!(short_model_name("hf:user/repo/model.gguf"), "model.gguf");
        assert_eq!(short_model_name("hf:bare"), "hf:bare"); // no '/', pop = whole
        assert_eq!(short_model_name("hf:ends/with/"), "hf:ends/with/"); // empty pop → full
        assert_eq!(short_model_name("plain-model"), "plain-model");
        let long = "x".repeat(57);
        let short = short_model_name(&long);
        assert_eq!(short.chars().count(), 56);
        assert!(short.ends_with("..."));
        let exactly_56 = "y".repeat(56);
        assert_eq!(short_model_name(&exactly_56), exactly_56);
    }
}
