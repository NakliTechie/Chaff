/**
 * Crypto boot self-test (G3). Standard known-answer vectors run at boot; any
 * mismatch is a loud, fatal failure — the tool refuses to encrypt with a broken
 * primitive. Also exercises seal/open round-trip and tamper-detection.
 */
import { aesGcmRaw, open, pbkdf2Sha256, seal } from "./aead.js";

function hex(u: Uint8Array): string {
  return Array.from(u, (b) => b.toString(16).padStart(2, "0")).join("");
}
function bytes(hexStr: string): Uint8Array {
  const out = new Uint8Array(hexStr.length / 2);
  for (let i = 0; i < out.length; i++) out[i] = parseInt(hexStr.slice(i * 2, i * 2 + 2), 16);
  return out;
}
const enc = (s: string) => new TextEncoder().encode(s);

export interface VectorResult {
  name: string;
  ok: boolean;
  detail?: string;
}

export async function runCryptoSelfTest(): Promise<{ ok: boolean; results: VectorResult[] }> {
  const results: VectorResult[] = [];
  const check = (name: string, ok: boolean, detail?: string) => results.push({ name, ok, detail });

  // 1. PBKDF2-HMAC-SHA256, P="password" S="salt" c=1 dkLen=32  (widely published KAT).
  {
    const dk = await pbkdf2Sha256(enc("password"), enc("salt"), 1, 32);
    const want = "120fb6cffcf8b32c43e7225256c4f837a86548c92ccc35480805987cb70be17b";
    check("PBKDF2-SHA256 c=1 KAT", hex(dk) === want, `got ${hex(dk)}`);
  }
  // 2. PBKDF2-HMAC-SHA256, c=2 dkLen=32 (second published KAT).
  {
    const dk = await pbkdf2Sha256(enc("password"), enc("salt"), 2, 32);
    const want = "ae4d0c95af6b46d32d0adff928f06dd02a303f8ef3c251dfd6e2d85a95474c43";
    check("PBKDF2-SHA256 c=2 KAT", hex(dk) === want, `got ${hex(dk)}`);
  }
  // 3. AES-256-GCM NIST vector: key=0^256, iv=0^96, PT empty → tag only.
  {
    const out = await aesGcmRaw(new Uint8Array(32), new Uint8Array(12), new Uint8Array(0), new Uint8Array(0));
    const want = "530f8afbc74536b9a963b4f1c4cb738b";
    check("AES-256-GCM empty-PT tag KAT", hex(out) === want, `got ${hex(out)}`);
  }
  // 4. AES-256-GCM NIST vector: key=0^256, iv=0^96, PT=0^128 → CT||tag.
  {
    const out = await aesGcmRaw(new Uint8Array(32), new Uint8Array(12), new Uint8Array(16), new Uint8Array(0));
    const want = "cea7403d4d606b6e074ec5d3baf39d18d0d1c8a799996bf0265b98b5d48ab919";
    check("AES-256-GCM 16-byte-PT KAT", hex(out) === want, `got ${hex(out)}`);
  }
  // 5. seal/open round-trip with real params (reduced iterations for boot speed:
  //    KDF correctness is already covered by the KATs above; this checks framing).
  {
    const params = { kdfIterations: 1000, saltBytes: 16, nonceBytes: 12, keyBits: 256 } as const;
    const pt = enc("chaff boot round-trip");
    const aad = enc("CHF0");
    const blob = await seal("hunter2", pt, aad, params);
    const back = await open("hunter2", blob, aad, params);
    check("seal/open round-trip", hex(back) === hex(pt), `got ${hex(back)}`);
  }
  // 6. Tamper detection: flipping a ciphertext byte must make open() throw.
  {
    const params = { kdfIterations: 1000, saltBytes: 16, nonceBytes: 12, keyBits: 256 } as const;
    const blob = await seal("hunter2", enc("secret"), enc("CHF0"), params);
    blob[blob.length - 1] = (blob[blob.length - 1]! ^ 0x01) & 0xff;
    let threw = false;
    try {
      await open("hunter2", blob, enc("CHF0"), params);
    } catch {
      threw = true;
    }
    check("AEAD tamper is rejected", threw);
  }
  // 7. Wrong-password rejection.
  {
    const params = { kdfIterations: 1000, saltBytes: 16, nonceBytes: 12, keyBits: 256 } as const;
    const blob = await seal("right", enc("secret"), enc("CHF0"), params);
    let threw = false;
    try {
      await open("wrong", blob, enc("CHF0"), params);
    } catch {
      threw = true;
    }
    check("wrong password is rejected", threw);
  }

  const ok = results.every((r) => r.ok);
  return { ok, results };
}

// Allow running standalone: `npm run selftest:crypto`.
if (import.meta.url === `file://${process.argv[1]}`) {
  runCryptoSelfTest().then(({ ok, results }) => {
    for (const r of results) console.log(`${r.ok ? "PASS" : "FAIL"}  ${r.name}${r.ok ? "" : `  (${r.detail ?? ""})`}`);
    console.log(ok ? "\nG3: crypto self-test GREEN" : "\nG3: crypto self-test RED");
    process.exit(ok ? 0 : 1);
  });
}
