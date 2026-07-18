/**
 * AEAD layer — PBKDF2(≥600k, SHA-256) → AES-256-GCM, via WebCrypto (native in
 * Node and the browser). Fails closed: a wrong password or a tampered blob makes
 * the GCM tag verification throw, so garbage never reaches the codec.
 *
 * Blob layout:  salt(16) || nonce(12) || ciphertext(+16-byte tag)
 */
// Use the global Web Crypto API — present in Node >=20 (globalThis.crypto) and
// in every browser. Keeping this dependency-free is what lets the exact same
// crypto module ship to the Chunk B cross-arch tester unchanged.
const webcrypto: Crypto = globalThis.crypto;
const subtle = webcrypto.subtle;

export interface CryptoParams {
  kdfIterations: number;
  saltBytes: number;
  nonceBytes: number;
  keyBits: number;
}

export const DEFAULT_PARAMS: CryptoParams = {
  kdfIterations: 600000,
  saltBytes: 16,
  nonceBytes: 12,
  keyBits: 256,
};

async function deriveKey(password: string, salt: Uint8Array, params: CryptoParams): Promise<CryptoKey> {
  const baseKey = await subtle.importKey("raw", new TextEncoder().encode(password), "PBKDF2", false, [
    "deriveKey",
  ]);
  return subtle.deriveKey(
    { name: "PBKDF2", salt: bufferSource(salt), iterations: params.kdfIterations, hash: "SHA-256" },
    baseKey,
    { name: "AES-GCM", length: params.keyBits },
    false,
    ["encrypt", "decrypt"],
  );
}

/** WebCrypto wants an ArrayBuffer-backed view; normalize to satisfy the types. */
function bufferSource(u: Uint8Array): ArrayBuffer {
  return u.buffer.slice(u.byteOffset, u.byteOffset + u.byteLength) as ArrayBuffer;
}

export async function seal(
  password: string,
  plaintext: Uint8Array,
  aad: Uint8Array,
  params: CryptoParams = DEFAULT_PARAMS,
): Promise<Uint8Array> {
  const salt = webcrypto.getRandomValues(new Uint8Array(params.saltBytes));
  const nonce = webcrypto.getRandomValues(new Uint8Array(params.nonceBytes));
  const key = await deriveKey(password, salt, params);
  const ct = new Uint8Array(
    await subtle.encrypt(
      { name: "AES-GCM", iv: bufferSource(nonce), additionalData: bufferSource(aad) },
      key,
      bufferSource(plaintext),
    ),
  );
  const out = new Uint8Array(salt.length + nonce.length + ct.length);
  out.set(salt, 0);
  out.set(nonce, salt.length);
  out.set(ct, salt.length + nonce.length);
  return out;
}

export async function open(
  password: string,
  blob: Uint8Array,
  aad: Uint8Array,
  params: CryptoParams = DEFAULT_PARAMS,
): Promise<Uint8Array> {
  const salt = blob.slice(0, params.saltBytes);
  const nonce = blob.slice(params.saltBytes, params.saltBytes + params.nonceBytes);
  const ct = blob.slice(params.saltBytes + params.nonceBytes);
  const key = await deriveKey(password, salt, params);
  const pt = new Uint8Array(
    await subtle.decrypt(
      { name: "AES-GCM", iv: bufferSource(nonce), additionalData: bufferSource(aad) },
      key,
      bufferSource(ct),
    ),
  );
  return pt;
}

// --- Low-level primitives exposed for the boot known-answer tests (G3). ---

export async function pbkdf2Sha256(password: Uint8Array, salt: Uint8Array, iterations: number, dkBytes: number): Promise<Uint8Array> {
  const baseKey = await subtle.importKey("raw", bufferSource(password), "PBKDF2", false, ["deriveBits"]);
  const bits = await subtle.deriveBits(
    { name: "PBKDF2", salt: bufferSource(salt), iterations, hash: "SHA-256" },
    baseKey,
    dkBytes * 8,
  );
  return new Uint8Array(bits);
}

export async function aesGcmRaw(
  keyBytes: Uint8Array,
  iv: Uint8Array,
  plaintext: Uint8Array,
  aad: Uint8Array,
): Promise<Uint8Array> {
  const key = await subtle.importKey("raw", bufferSource(keyBytes), "AES-GCM", false, ["encrypt"]);
  const out = await subtle.encrypt(
    { name: "AES-GCM", iv: bufferSource(iv), additionalData: bufferSource(aad) },
    key,
    bufferSource(plaintext),
  );
  return new Uint8Array(out);
}
