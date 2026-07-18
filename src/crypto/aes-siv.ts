/**
 * AES-SIV (RFC 5297) — nonce-misuse-resistant authenticated encryption, matching
 * the source project's crypto posture. Clean-room implementation from the RFC on
 * top of WebCrypto primitives; verified against the RFC 4493 (CMAC) and RFC 5297
 * (SIV) known-answer vectors in the boot self-test (G3).
 *
 * WebCrypto has no raw single-block ECB, so we synthesize AES(K, block) with
 * AES-CTR: CTR keystream block 0 for counter=block over a zero plaintext IS
 * AES(K, block). CMAC and S2V are built on that; the CTR encryption step uses
 * AES-CTR directly.
 */
const subtle = globalThis.crypto.subtle;

const R = 0x87; // GF(2^128) reduction constant

function buf(u: Uint8Array): ArrayBuffer {
  return u.buffer.slice(u.byteOffset, u.byteOffset + u.byteLength) as ArrayBuffer;
}
function xor(a: Uint8Array, b: Uint8Array, n = Math.min(a.length, b.length)): Uint8Array {
  const out = new Uint8Array(n);
  for (let i = 0; i < n; i++) out[i] = a[i]! ^ b[i]!;
  return out;
}
/** Left-shift a 128-bit block by one bit, reducing with R on carry (RFC 4493 dbl). */
function dbl(x: Uint8Array): Uint8Array {
  const out = new Uint8Array(16);
  let carry = 0;
  for (let i = 15; i >= 0; i--) {
    const v = (x[i]! << 1) | carry;
    out[i] = v & 0xff;
    carry = (x[i]! & 0x80) ? 1 : 0;
  }
  if (x[0]! & 0x80) out[15] = out[15]! ^ R;
  return out;
}

async function importCtr(key: Uint8Array): Promise<CryptoKey> {
  return subtle.importKey("raw", buf(key), "AES-CTR", false, ["encrypt"]);
}

/** Single-block AES(K, block) via one AES-CTR keystream block. */
async function aesBlock(ctrKey: CryptoKey, block: Uint8Array): Promise<Uint8Array> {
  const zero = new Uint8Array(16);
  const out = await subtle.encrypt({ name: "AES-CTR", counter: buf(block), length: 128 }, ctrKey, buf(zero));
  return new Uint8Array(out).subarray(0, 16);
}

/** AES-CTR encrypt/decrypt (symmetric) with a full-block initial counter. */
async function aesCtr(key: Uint8Array, iv: Uint8Array, data: Uint8Array): Promise<Uint8Array> {
  if (data.length === 0) return new Uint8Array(0);
  const k = await importCtr(key);
  const out = await subtle.encrypt({ name: "AES-CTR", counter: buf(iv), length: 128 }, k, buf(data));
  return new Uint8Array(out);
}

/** CMAC (RFC 4493) over `msg` with `key`. */
export async function cmac(key: Uint8Array, msg: Uint8Array): Promise<Uint8Array> {
  const ck = await importCtr(key);
  const L = await aesBlock(ck, new Uint8Array(16));
  const K1 = dbl(L);
  const K2 = dbl(K1);

  const n = Math.ceil(msg.length / 16);
  const complete = msg.length > 0 && msg.length % 16 === 0;
  let last: Uint8Array;
  if (n === 0) {
    // empty message: pad(0^0) = 0x80 00..., xor K2
    const padded = new Uint8Array(16);
    padded[0] = 0x80;
    last = xor(padded, K2, 16);
  } else if (complete) {
    last = xor(msg.subarray((n - 1) * 16, n * 16), K1, 16);
  } else {
    const rem = msg.subarray((n - 1) * 16);
    const padded = new Uint8Array(16);
    padded.set(rem);
    padded[rem.length] = 0x80;
    last = xor(padded, K2, 16);
  }

  let x: Uint8Array = new Uint8Array(16);
  const blocks = Math.max(n, 1);
  for (let i = 0; i < blocks - 1; i++) {
    x = await aesBlock(ck, xor(x, msg.subarray(i * 16, i * 16 + 16), 16));
  }
  return aesBlock(ck, xor(x, last, 16));
}

/** S2V (RFC 5297 §2.4) — associated-data vector + plaintext → synthetic IV. */
async function s2v(key: Uint8Array, ads: Uint8Array[], plaintext: Uint8Array): Promise<Uint8Array> {
  const zero = new Uint8Array(16);
  if (ads.length === 0 && plaintext.length === 0) {
    const one = new Uint8Array(16);
    one[15] = 1;
    return cmac(key, one);
  }
  let d = await cmac(key, zero);
  for (const ad of ads) {
    d = xor(dbl(d), await cmac(key, ad), 16);
  }
  let t: Uint8Array;
  if (plaintext.length >= 16) {
    // xorend: xor D into the final 16 bytes of the plaintext
    t = new Uint8Array(plaintext);
    const off = t.length - 16;
    for (let i = 0; i < 16; i++) t[off + i] = t[off + i]! ^ d[i]!;
  } else {
    const padded = new Uint8Array(16);
    padded.set(plaintext);
    padded[plaintext.length] = 0x80;
    t = xor(dbl(d), padded, 16);
  }
  return cmac(key, t);
}

/** Clear the 31st and 63rd bits (from the right) of the SIV to form the CTR IV. */
function sivToCtrIv(v: Uint8Array): Uint8Array {
  const q = new Uint8Array(v);
  q[8] = q[8]! & 0x7f;
  q[12] = q[12]! & 0x7f;
  return q;
}

/**
 * AES-SIV encrypt. `key` is 2N bytes (K1 = first half for S2V/CMAC, K2 = second
 * half for CTR); 64 bytes → AES-256-SIV. Returns SIV(16) || ciphertext.
 */
export async function sivEncrypt(key: Uint8Array, ads: Uint8Array[], plaintext: Uint8Array): Promise<Uint8Array> {
  const half = key.length / 2;
  const k1 = key.subarray(0, half);
  const k2 = key.subarray(half);
  const v = await s2v(k1, ads, plaintext);
  const c = await aesCtr(k2, sivToCtrIv(v), plaintext);
  const out = new Uint8Array(16 + c.length);
  out.set(v, 0);
  out.set(c, 16);
  return out;
}

/** AES-SIV decrypt. Throws on authentication failure. */
export async function sivDecrypt(key: Uint8Array, ads: Uint8Array[], input: Uint8Array): Promise<Uint8Array> {
  if (input.length < 16) throw new Error("AES-SIV: input too short");
  const half = key.length / 2;
  const k1 = key.subarray(0, half);
  const k2 = key.subarray(half);
  const v = input.subarray(0, 16);
  const c = input.subarray(16);
  const p = await aesCtr(k2, sivToCtrIv(v), c);
  const t = await s2v(k1, ads, p);
  // constant-time-ish compare
  let diff = 0;
  for (let i = 0; i < 16; i++) diff |= t[i]! ^ v[i]!;
  if (diff !== 0) throw new Error("AES-SIV: authentication failed");
  return p;
}
