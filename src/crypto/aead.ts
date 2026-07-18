/**
 * AEAD layer — PBKDF2(≥600k, SHA-256) → authenticated encryption, via WebCrypto.
 * Fails closed: a wrong password or a tampered blob makes authentication throw,
 * so garbage never reaches the codec.
 *
 *   - Messages: AES-256-SIV (RFC 5297, nonce-misuse resistant) — `sealSiv/openSiv`.
 *   - At-rest state: AES-256-GCM — `seal/open`.
 *
 * SIV blob layout:  salt(16) || nonce(16) || SIV(16) || ciphertext
 * GCM blob layout:  salt(16) || nonce(12) || ciphertext(+16-byte tag)
 */
import { sivEncrypt, sivDecrypt } from "./aes-siv.js";
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

/** Derive a 64-byte key for AES-256-SIV (two 256-bit halves) from the password. */
async function deriveSivKey(password: string, salt: Uint8Array, params: CryptoParams): Promise<Uint8Array> {
  return pbkdf2Sha256(new TextEncoder().encode(password), salt, params.kdfIterations, 64);
}

/**
 * Seal a message with AES-256-SIV. A random nonce is passed as an S2V associated
 * component (RFC 5297 §3) so identical plaintexts still produce distinct output,
 * while retaining nonce-misuse resistance.
 * Blob: salt || nonce(16) || SIV(16) || ciphertext.
 */
export async function sealSiv(
  password: string,
  plaintext: Uint8Array,
  aad: Uint8Array,
  params: CryptoParams = DEFAULT_PARAMS,
): Promise<Uint8Array> {
  const salt = webcrypto.getRandomValues(new Uint8Array(params.saltBytes));
  const nonce = webcrypto.getRandomValues(new Uint8Array(16));
  const key = await deriveSivKey(password, salt, params);
  const out = await sivEncrypt(key, [aad, nonce], plaintext); // SIV(16) || ct
  const blob = new Uint8Array(salt.length + nonce.length + out.length);
  blob.set(salt, 0);
  blob.set(nonce, salt.length);
  blob.set(out, salt.length + nonce.length);
  return blob;
}

export async function openSiv(
  password: string,
  blob: Uint8Array,
  aad: Uint8Array,
  params: CryptoParams = DEFAULT_PARAMS,
): Promise<Uint8Array> {
  const salt = blob.slice(0, params.saltBytes);
  const nonce = blob.slice(params.saltBytes, params.saltBytes + 16);
  const out = blob.slice(params.saltBytes + 16);
  const key = await deriveSivKey(password, salt, params);
  return sivDecrypt(key, [aad, nonce], out);
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
