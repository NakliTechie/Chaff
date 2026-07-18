/**
 * Conversation chaining — capability parity with the source project's "message
 * chaining" (detect tampering, deletion, reordering). Clean-room design.
 *
 * A conversation is a single ordered chain shared by both parties (matching the
 * source's "both must process messages in identical order"). Each message hides
 * an authenticated header inside the AEAD plaintext:
 *
 *     [ version(1) ][ seq(uint32, BE) ][ prevChain(16) ][ secret bytes ]
 *
 * and the running chain advances as  chain' = SHA-256(chain || ciphertextBlob)[:16].
 * The genesis chain is derived from the (public) conversation id. Because the
 * header is authenticated by AES-SIV and the conversation id is bound as
 * associated data, an attacker cannot forge it; the chain + sequence then catch
 * drops, reorders, and replays that per-message integrity alone cannot see:
 *
 *   - tampered message  → AES-SIV auth fails (before we even get here)
 *   - deleted message   → next seq skips / prevChain mismatch  → "missing-message"
 *   - reordered/replayed → seq ≤ state.seq or prevChain mismatch → "reorder-or-replay"
 *
 * Pure, portable (WebCrypto SHA-256 + TextEncoder) — shared by the Node tests
 * and the browser app.
 */
const CHAIN_LEN = 16;
const HEADER_LEN = 1 + 4 + CHAIN_LEN; // 21
const VERSION = 1;

function bufOf(u: Uint8Array): ArrayBuffer {
  return u.buffer.slice(u.byteOffset, u.byteOffset + u.byteLength) as ArrayBuffer;
}

export async function sha256(data: Uint8Array): Promise<Uint8Array> {
  return new Uint8Array(await globalThis.crypto.subtle.digest("SHA-256", bufOf(data)));
}

export async function genesisChain(conversationId: string): Promise<Uint8Array> {
  const h = await sha256(new TextEncoder().encode("chaff-chain-v1:" + conversationId));
  return h.slice(0, CHAIN_LEN);
}

export async function nextChain(prev: Uint8Array, blob: Uint8Array): Promise<Uint8Array> {
  const cat = new Uint8Array(prev.length + blob.length);
  cat.set(prev, 0);
  cat.set(blob, prev.length);
  return (await sha256(cat)).slice(0, CHAIN_LEN);
}

export function buildHeader(seq: number, prevChain: Uint8Array): Uint8Array {
  const h = new Uint8Array(HEADER_LEN);
  h[0] = VERSION;
  new DataView(h.buffer).setUint32(1, seq >>> 0, false);
  h.set(prevChain.subarray(0, CHAIN_LEN), 5);
  return h;
}

export interface ParsedHeader {
  seq: number;
  prevChain: Uint8Array;
  body: Uint8Array;
}

export function parseHeader(pt: Uint8Array): ParsedHeader {
  if (pt.length < HEADER_LEN || pt[0] !== VERSION) throw new Error("bad conversation header");
  const seq = new DataView(pt.buffer, pt.byteOffset, pt.byteLength).getUint32(1, false);
  return { seq, prevChain: pt.slice(5, 5 + CHAIN_LEN), body: pt.slice(HEADER_LEN) };
}

export interface ConvState {
  seq: number;
  chain: Uint8Array;
}

export interface ChainStatus {
  seq: number;
  ok: boolean;
  warnings: string[];
}

function bytesEq(a: Uint8Array, b: Uint8Array): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

/** Compare a received header against local state; report drop/reorder/replay. */
export function verifyChain(state: ConvState, header: ParsedHeader): ChainStatus {
  const warnings: string[] = [];
  if (!bytesEq(header.prevChain, state.chain)) warnings.push("chain-mismatch");
  if (header.seq <= state.seq) warnings.push("reorder-or-replay");
  else if (header.seq > state.seq + 1) warnings.push("missing-message");
  return { seq: header.seq, ok: warnings.length === 0, warnings };
}

/** Human-readable one-liner for a chain status. */
export function describeStatus(s: ChainStatus): string {
  if (s.ok) return `message #${s.seq} · chain intact`;
  const parts: string[] = [];
  if (s.warnings.includes("missing-message")) parts.push("a message may be missing before this one");
  if (s.warnings.includes("reorder-or-replay")) parts.push("out of order or replayed");
  if (s.warnings.includes("chain-mismatch")) parts.push("chain broken (tampering, a drop, or a desync)");
  return `message #${s.seq} · ${parts.join("; ")}`;
}
