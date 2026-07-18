/**
 * End-to-end pipeline: secret ⇄ cover tokens.
 *
 *   encode:  secret --AEAD--> blob --frame--> bitstream --codec--> cover tokens
 *   decode:  cover tokens --codec--> bitstream --unframe--> blob --AEAD--> secret
 *
 * Frame layout carried by the codec:  [ lengthField(32b) = blob byte length ]
 *                                     [ blob bytes ]
 * The magic string is authenticated as AEAD associated data (it costs no
 * carrier capacity and still makes a corrupt/foreign stream fail the GCM tag).
 */
import type { LogitSource } from "./inference/types.js";
import type { Pins } from "./pins.js";
import { BitReader, BitWriter } from "./util/bits.js";
import { type CoderConfig, type StepTrace } from "./codec/coder.js";
import { encodeEntropy, decodeEntropy } from "./codec/entropy.js";
import { openSiv, sealSiv, type CryptoParams } from "./crypto/aead.js";

export function coderConfigFromPins(pins: Pins): CoderConfig {
  return {
    logitScale: pins.codec.logitScale,
    bucketWidth: pins.codec.bucketWidth,
    maxBitsPerStep: pins.codec.maxBitsPerStep,
    softmaxPrecBits: pins.codec.softmaxPrecBits,
    topK: pins.codec.topK ?? 128,
  };
}

export function cryptoParamsFromPins(pins: Pins): CryptoParams {
  return {
    kdfIterations: pins.crypto.kdfIterations,
    saltBytes: pins.crypto.saltBytes,
    nonceBytes: pins.crypto.nonceBytes,
    keyBits: pins.crypto.keyBits,
  };
}

function framePayload(blob: Uint8Array, lengthFieldBits: number): { bytes: Uint8Array; bitLength: number } {
  const w = new BitWriter();
  w.writeBits(blob.length, lengthFieldBits);
  for (const b of blob) w.writeBits(b, 8);
  return w.finish();
}

function unframePayload(bits: Uint8Array, bitLength: number, lengthFieldBits: number): Uint8Array {
  const r = new BitReader(bits, bitLength);
  const blobLen = r.readBits(lengthFieldBits);
  // A valid frame's blob can't be longer than the bytes actually decoded. Reject a
  // corrupt/adversarial length field before allocating — otherwise a bogus 32-bit
  // length triggers a multi-GB alloc + read loop (hang/OOM) before the AEAD rejects.
  const maxBytes = Math.max(0, (bitLength - lengthFieldBits) >> 3);
  if (blobLen > maxBytes) {
    throw new Error(`frame claims ${blobLen} bytes but only ${maxBytes} decoded — corrupt cover`);
  }
  const blob = new Uint8Array(blobLen);
  for (let i = 0; i < blobLen; i++) blob[i] = r.readBits(8);
  return blob;
}

export interface EncodeSecretResult {
  cover: number[];
  trace: StepTrace[];
  blobBytes: number;
  payloadBits: number;
}

export async function encodeSecret(
  source: LogitSource,
  password: string,
  secret: Uint8Array,
  pins: Pins,
  seedPrefix: number[] = [],
): Promise<EncodeSecretResult> {
  const aad = new TextEncoder().encode(pins.payload.magic);
  const blob = await sealSiv(password, secret, aad, cryptoParamsFromPins(pins));
  const framed = framePayload(blob, pins.payload.lengthFieldBits);
  const result = encodeEntropy(source, framed.bytes, framed.bitLength, coderConfigFromPins(pins), seedPrefix);
  return { cover: result.cover, trace: result.trace, blobBytes: blob.length, payloadBits: framed.bitLength };
}

export interface DecodeSecretResult {
  secret: Uint8Array;
  trace: StepTrace[];
}

export async function decodeSecret(
  source: LogitSource,
  password: string,
  cover: number[],
  pins: Pins,
  seedPrefix: number[] = [],
): Promise<DecodeSecretResult> {
  const dec = decodeEntropy(source, cover, coderConfigFromPins(pins), undefined, seedPrefix);
  const blob = unframePayload(dec.bytes, dec.bitLength, pins.payload.lengthFieldBits);
  const aad = new TextEncoder().encode(pins.payload.magic);
  const secret = await openSiv(password, blob, aad, cryptoParamsFromPins(pins));
  return { secret, trace: dec.trace };
}
