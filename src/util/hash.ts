/**
 * Deterministic hashing helpers for the divergence localizer and the pins
 * integrity check. FNV-1a over a canonical byte encoding — small, dependency
 * free, and identical on every engine.
 */

const FNV_OFFSET = 0x811c9dc5;
const FNV_PRIME = 0x01000193;

export function fnv1a(bytes: Uint8Array): number {
  let h = FNV_OFFSET;
  for (let i = 0; i < bytes.length; i++) {
    h ^= bytes[i]!;
    h = Math.imul(h, FNV_PRIME);
  }
  return h >>> 0;
}

export function hashHex(bytes: Uint8Array): string {
  return fnv1a(bytes).toString(16).padStart(8, "0");
}

/** Hash an array of integers (canonical little-endian int32) — used for logits/softmax fingerprints. */
export function hashInts(values: ArrayLike<number>): string {
  const buf = new Uint8Array(values.length * 4);
  const view = new DataView(buf.buffer);
  for (let i = 0; i < values.length; i++) {
    // Truncate toward zero into int32 domain; callers pass already-integral values.
    view.setInt32(i * 4, values[i]! | 0, true);
  }
  return hashHex(buf);
}
