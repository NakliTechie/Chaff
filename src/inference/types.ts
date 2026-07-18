/**
 * The inference contract. A LogitSource returns raw pre-softmax logits for the
 * next token given a prefix of token ids. This is the single seam the codec
 * depends on; everything downstream is model-agnostic.
 *
 *  - reference-lm.ts   : deterministic fixed-weight transformer (headless gates)
 *  - ort-gpt2 (Chunk B): ONNX-Runtime-Web + Xenova/gpt2 in a real browser (G2)
 *
 * Determinism mandate #1 (WASM/fp32/1-thread/pinned opt level, no WebGPU) lives
 * in the ORT backend. The reference LM is pure-JS fp64 with a fixed operation
 * order, so it is deterministic run-to-run on the machine running the gates.
 */
export interface LogitSource {
  /** Number of tokens in the vocabulary (logits length). */
  readonly vocabSize: number;
  /** Human-readable pinned id, e.g. "chaff-ref-lm@1". */
  readonly id: string;
  /** Hash of the frozen weights / model, folded into the state file and pins check. */
  readonly weightHash: string;
  /**
   * Raw pre-softmax logits (Float32-precision values) for the token that would
   * follow `prefix`. `prefix` is the full sequence so far (token ids). The
   * source may window it to its context length internally.
   */
  logits(prefix: number[]): Float64Array;
}
