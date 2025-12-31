//! VByte Variable-Length Encoding Module
//!
//! VByte (Variable Byte) is an efficient integer compression encoding:
//! - Small numbers (<128) need only 1 byte
//! - Each byte's highest bit is a continuation flag (1=continue, 0=end)
//! - Lower 7 bits store data
//!
//! Used to compress small posting lists, more space-efficient than Roaring Bitmap

/// Encode single u32 to VByte format
/// 
/// Returns number of bytes encoded
#[inline]
pub fn encode_u32(value: u32, output: &mut Vec<u8>) -> usize {
    let mut v = value;
    let start_len = output.len();
    
    // Process high bytes (with continuation flag)
    while v >= 0x80 {
        output.push((v as u8) | 0x80);
        v >>= 7;
    }
    // Last byte (no continuation flag)
    output.push(v as u8);
    
    output.len() - start_len
}

/// Decode VByte to u32
/// 
/// Returns (decoded value, bytes consumed)
#[inline]
pub fn decode_u32(input: &[u8]) -> Option<(u32, usize)> {
    let mut result: u32 = 0;
    let mut shift = 0;
    
    for (i, &byte) in input.iter().enumerate() {
        result |= ((byte & 0x7F) as u32) << shift;
        
        if byte & 0x80 == 0 {
            return Some((result, i + 1));
        }
        
        shift += 7;
        if shift > 28 {
            // Prevent overflow (u32 needs at most 5 bytes)
            return None;
        }
    }
    
    None // Incomplete input
}

/// Encode u64 to VByte format
#[inline]
pub fn encode_u64(value: u64, output: &mut Vec<u8>) -> usize {
    let mut v = value;
    let start_len = output.len();
    
    while v >= 0x80 {
        output.push((v as u8) | 0x80);
        v >>= 7;
    }
    output.push(v as u8);
    
    output.len() - start_len
}

/// Decode VByte to u64
#[inline]
pub fn decode_u64(input: &[u8]) -> Option<(u64, usize)> {
    let mut result: u64 = 0;
    let mut shift = 0;
    
    for (i, &byte) in input.iter().enumerate() {
        result |= ((byte & 0x7F) as u64) << shift;
        
        if byte & 0x80 == 0 {
            return Some((result, i + 1));
        }
        
        shift += 7;
        if shift > 63 {
            return None;
        }
    }
    
    None
}

/// Batch encode u32 array (delta encoding + VByte)
/// 
/// Uses delta encoding for further compression: stores differences between adjacent elements
/// Requires input array to be sorted
#[inline]
pub fn encode_sorted_u32_array(values: &[u32], output: &mut Vec<u8>) {
    if values.is_empty() {
        return;
    }
    
    // Store length first
    encode_u32(values.len() as u32, output);
    
    // First value stored completely
    encode_u32(values[0], output);
    
    // Subsequent values delta encoded
    for i in 1..values.len() {
        let delta = values[i] - values[i - 1];
        encode_u32(delta, output);
    }
}

/// Batch decode delta-encoded u32 array
#[inline]
pub fn decode_sorted_u32_array(input: &[u8]) -> Option<(Vec<u32>, usize)> {
    let mut offset = 0;
    
    // Read length
    let (len, bytes) = decode_u32(&input[offset..])?;
    offset += bytes;
    
    if len == 0 {
        return Some((Vec::new(), offset));
    }
    
    let mut values = Vec::with_capacity(len as usize);
    
    // Read first value
    let (first, bytes) = decode_u32(&input[offset..])?;
    offset += bytes;
    values.push(first);
    
    // Read delta values and restore
    let mut prev = first;
    for _ in 1..len {
        let (delta, bytes) = decode_u32(&input[offset..])?;
        offset += bytes;
        prev += delta;
        values.push(prev);
    }
    
    Some((values, offset))
}

/// Batch encode u64 array (delta encoding + VByte)
#[inline]
pub fn encode_sorted_u64_array(values: &[u64], output: &mut Vec<u8>) {
    if values.is_empty() {
        return;
    }
    
    encode_u64(values.len() as u64, output);
    encode_u64(values[0], output);
    
    for i in 1..values.len() {
        let delta = values[i] - values[i - 1];
        encode_u64(delta, output);
    }
}

/// Batch decode delta-encoded u64 array
#[inline]
pub fn decode_sorted_u64_array(input: &[u8]) -> Option<(Vec<u64>, usize)> {
    let mut offset = 0;
    
    let (len, bytes) = decode_u64(&input[offset..])?;
    offset += bytes;
    
    if len == 0 {
        return Some((Vec::new(), offset));
    }
    
    let mut values = Vec::with_capacity(len as usize);
    
    let (first, bytes) = decode_u64(&input[offset..])?;
    offset += bytes;
    values.push(first);
    
    let mut prev = first;
    for _ in 1..len {
        let (delta, bytes) = decode_u64(&input[offset..])?;
        offset += bytes;
        prev += delta;
        values.push(prev);
    }
    
    Some((values, offset))
}

/// Calculate encoded byte count (without actually encoding)
#[inline]
pub fn encoded_size_u32(value: u32) -> usize {
    match value {
        0..=0x7F => 1,
        0x80..=0x3FFF => 2,
        0x4000..=0x1F_FFFF => 3,
        0x20_0000..=0xFFF_FFFF => 4,
        _ => 5,
    }
}

/// Estimate byte count after delta encoding
#[inline]
pub fn estimate_sorted_array_size(values: &[u32]) -> usize {
    if values.is_empty() {
        return 1; // Only length
    }
    
    let mut size = encoded_size_u32(values.len() as u32);
    size += encoded_size_u32(values[0]);
    
    for i in 1..values.len() {
        let delta = values[i] - values[i - 1];
        size += encoded_size_u32(delta);
    }
    
    size
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_encode_decode_u32() {
        let test_values = [0, 1, 127, 128, 255, 16383, 16384, u32::MAX];
        
        for &value in &test_values {
            let mut buf = Vec::new();
            encode_u32(value, &mut buf);
            
            let (decoded, len) = decode_u32(&buf).unwrap();
            assert_eq!(decoded, value, "Failed for value {}", value);
            assert_eq!(len, buf.len());
        }
    }
    
    #[test]
    fn test_encode_decode_u64() {
        let test_values = [0u64, 1, 127, 128, u32::MAX as u64, u64::MAX];
        
        for &value in &test_values {
            let mut buf = Vec::new();
            encode_u64(value, &mut buf);
            
            let (decoded, len) = decode_u64(&buf).unwrap();
            assert_eq!(decoded, value);
            assert_eq!(len, buf.len());
        }
    }
    
    #[test]
    fn test_sorted_array() {
        let values = vec![1, 5, 10, 100, 1000, 10000];
        
        let mut buf = Vec::new();
        encode_sorted_u32_array(&values, &mut buf);
        
        let (decoded, _) = decode_sorted_u32_array(&buf).unwrap();
        assert_eq!(decoded, values);
    }
    
    #[test]
    fn test_compression_ratio() {
        // Test compression effectiveness
        let values: Vec<u32> = (0..1000).map(|i| i * 10).collect();
        
        let mut buf = Vec::new();
        encode_sorted_u32_array(&values, &mut buf);
        
        let raw_size = values.len() * 4;
        let compressed_size = buf.len();
        
        println!("Raw: {} bytes, Compressed: {} bytes, Ratio: {:.2}x",
            raw_size, compressed_size, raw_size as f64 / compressed_size as f64);
        
        // After delta encoding, each delta is 10, needs only 1 byte
        // Expected compression ratio > 2x
        assert!(compressed_size < raw_size / 2);
    }
    
    #[test]
    fn test_empty_array() {
        let values: Vec<u32> = vec![];
        
        let mut buf = Vec::new();
        encode_sorted_u32_array(&values, &mut buf);
        
        let (decoded, _) = decode_sorted_u32_array(&buf).unwrap();
        assert!(decoded.is_empty());
    }
}
