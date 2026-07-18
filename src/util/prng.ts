/**
 * Deterministic PRNG (splitmix64 → xoshiro-free, minimal) used ONLY to
 * materialize the reference LM's fixed weights from a pinned seed. It never
 * touches the coding path or crypto. Pure integer (BigInt) math so the exact
 * same weights are produced on every machine and every JS engine.
 */

const MASK64 = (1n << 64n) - 1n;

export class SplitMix64 {
  private state: bigint;

  constructor(seed: bigint) {
    this.state = seed & MASK64;
  }

  /** Next raw 64-bit value. */
  nextU64(): bigint {
    this.state = (this.state + 0x9e3779b97f4a7c15n) & MASK64;
    let z = this.state;
    z = ((z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n) & MASK64;
    z = ((z ^ (z >> 27n)) * 0x94d049bb133111ebn) & MASK64;
    z = z ^ (z >> 31n);
    return z & MASK64;
  }

  /** Uniform float in [0,1) with 53 bits of entropy. */
  nextFloat(): number {
    const v = this.nextU64() >> 11n; // top 53 bits
    return Number(v) / 9007199254740992; // 2^53
  }

  /** Standard-normal-ish sample via Box–Muller (deterministic). */
  nextGaussian(): number {
    // Avoid log(0).
    let u1 = this.nextFloat();
    const u2 = this.nextFloat();
    if (u1 < 1e-12) u1 = 1e-12;
    const r = Math.sqrt(-2 * Math.log(u1));
    return r * Math.cos(2 * Math.PI * u2);
  }
}

export function seedFromString(hex: string): bigint {
  // Accepts "0x..." or plain hex/decimal string.
  const cleaned = hex.startsWith("0x") || hex.startsWith("0X") ? hex : "0x" + hex;
  return BigInt(cleaned) & MASK64;
}
