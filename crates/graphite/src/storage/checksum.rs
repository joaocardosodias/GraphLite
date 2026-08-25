use crate::error::{GraphiteError, Result};
use crate::storage::header::HEADER_SIZE;

/// Standard IEEE 802.3 CRC-32 polynomial (used in zlib, PNG, and Ethernet).
const CRC32_POLYNOMIAL: u32 = 0xEDB88320;

/// Compile-time precomputed 256-entry CRC-32 lookup table.
const CRC32_TABLE: [u32; 256] = {
    let mut table = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut crc = i as u32;
        let mut j = 0;
        while j < 8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ CRC32_POLYNOMIAL;
            } else {
                crc >>= 1;
            }
            j += 1;
        }
        table[i] = crc;
        i += 1;
    }
    table
};

/// Computes the standard IEEE 802.3 CRC-32 checksum of a byte slice.
pub fn crc32(data: &[u8]) -> u32 {
    crc32_update(0, data)
}

/// Updates an existing CRC-32 checksum with an additional chunk of data.
pub fn crc32_update(crc: u32, data: &[u8]) -> u32 {
    let mut current = !crc;
    for &b in data {
        let table_idx = ((current ^ (b as u32)) & 0xFF) as usize;
        current = CRC32_TABLE[table_idx] ^ (current >> 8);
    }
    !current
}

/// Computes the payload integrity checksum for a `.graph` file buffer.
///
/// Skips the 4-byte checksum field itself located at bytes 56..60 in the 64-byte header.
pub fn compute_file_checksum(file_bytes: &[u8]) -> u32 {
    if file_bytes.len() < HEADER_SIZE {
        return crc32(file_bytes);
    }

    // Hash Header bytes 0..56
    let crc = crc32(&file_bytes[0..56]);
    // Hash Header bytes 60..64 (reserved padding)
    let crc = crc32_update(crc, &file_bytes[60..HEADER_SIZE]);
    // Hash entire file payload after the header
    crc32_update(crc, &file_bytes[HEADER_SIZE..])
}

/// Validates that the file buffer matches the expected CRC32 checksum.
pub fn verify_file_integrity(file_bytes: &[u8], expected_checksum: u32) -> Result<()> {
    let calculated = compute_file_checksum(file_bytes);
    if calculated != expected_checksum {
        return Err(GraphiteError::ChecksumMismatch {
            expected: expected_checksum,
            calculated,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crc32_standard_test_vectors() {
        assert_eq!(crc32(b""), 0x00000000);
        assert_eq!(crc32(b"123456789"), 0xCBF43926);
        assert_eq!(
            crc32(b"The quick brown fox jumps over the lazy dog"),
            0x414FA339
        );
    }

    #[test]
    fn test_crc32_chunked_update_parity() {
        let full_data = b"Hello, Graphite binary storage with zero-copy!";
        let full_crc = crc32(full_data);

        // Compute in two separate chunks
        let part1 = &full_data[0..15];
        let part2 = &full_data[15..];

        let crc1 = crc32(part1);
        let chunked_crc = crc32_update(crc1, part2);

        assert_eq!(full_crc, chunked_crc);
    }

    #[test]
    fn test_file_checksum_skips_checksum_field() {
        let mut mock_file = vec![0u8; 256];
        mock_file[0..4].copy_from_slice(b"GRPH");

        // Calculate initial file checksum
        let initial_checksum = compute_file_checksum(&mock_file);

        // Modifying bytes 56..60 (the checksum field itself) must NOT change the result
        mock_file[56..60].copy_from_slice(&0xDEADBEEFu32.to_le_bytes());
        let updated_checksum = compute_file_checksum(&mock_file);

        assert_eq!(initial_checksum, updated_checksum);

        // But modifying ANY other byte (e.g. at index 100) MUST change the checksum!
        mock_file[100] = 0xFF;
        let corrupted_checksum = compute_file_checksum(&mock_file);
        assert_ne!(initial_checksum, corrupted_checksum);
    }

    #[test]
    fn test_verify_file_integrity_mismatch_error() {
        let mock_file = vec![0xABu8; 128];
        let real_checksum = compute_file_checksum(&mock_file);

        // Verification with correct checksum succeeds
        assert!(verify_file_integrity(&mock_file, real_checksum).is_ok());

        // Verification with wrong checksum returns ChecksumMismatch
        let err = verify_file_integrity(&mock_file, 0x12345678).unwrap_err();
        match err {
            GraphiteError::ChecksumMismatch {
                expected,
                calculated,
            } => {
                assert_eq!(expected, 0x12345678);
                assert_eq!(calculated, real_checksum);
            }
            _ => panic!("Expected ChecksumMismatch error"),
        }
    }
}
