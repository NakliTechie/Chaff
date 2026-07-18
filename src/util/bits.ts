/**
 * MSB-first bit reader/writer over a byte buffer. Deterministic, simple, and
 * the single bit-ordering convention shared by the codec and the pipeline.
 */

export class BitWriter {
  private bytes: number[] = [];
  private cur = 0;
  private nbits = 0;
  private total = 0;

  writeBit(b: number): void {
    this.cur = (this.cur << 1) | (b & 1);
    this.nbits++;
    this.total++;
    if (this.nbits === 8) {
      this.bytes.push(this.cur & 0xff);
      this.cur = 0;
      this.nbits = 0;
    }
  }

  /** Write the low `count` bits of `value`, MSB-first. */
  writeBits(value: number, count: number): void {
    for (let i = count - 1; i >= 0; i--) this.writeBit((value >>> i) & 1);
  }

  get bitLength(): number {
    return this.total;
  }

  finish(): { bytes: Uint8Array; bitLength: number } {
    let last = this.bytes.slice();
    if (this.nbits > 0) {
      last.push((this.cur << (8 - this.nbits)) & 0xff);
    }
    return { bytes: Uint8Array.from(last), bitLength: this.total };
  }
}

export class BitReader {
  private pos = 0;
  constructor(private readonly bytes: Uint8Array, private readonly bitLength: number) {}

  get remaining(): number {
    return this.bitLength - this.pos;
  }

  get consumed(): number {
    return this.pos;
  }

  readBit(): number {
    if (this.pos >= this.bitLength) return 0; // zero-pad past the end
    const byte = this.bytes[this.pos >> 3]!;
    const bit = (byte >> (7 - (this.pos & 7))) & 1;
    this.pos++;
    return bit;
  }

  /** Read `count` bits MSB-first into an unsigned integer. */
  readBits(count: number): number {
    let v = 0;
    for (let i = 0; i < count; i++) v = (v << 1) | this.readBit();
    return v >>> 0;
  }
}
