import { CORPUS } from "./corpus";
import { Confusions, DEFAULT_DECODER, SimulatedDecoder } from "./decoder";
import { calibration, DEFAULT_TIME, evaluate, lambdaSweep, runPolicy, thresholdSweep } from "./evaluate";
import { BigramLM, BOS, EOS, tokenize } from "./lm";
import { rescore } from "./rescore";
import { Rng } from "./rng";
import { SessionEngine } from "./session";

const lm = new BigramLM(CORPUS);
const vocab = [...lm.vocab];

describe("BigramLM", () => {
  it("is a probability distribution over the vocabulary for any context", () => {
    for (const prev of [BOS, "the", "i", "zebra"]) {
      const total = [...vocab, EOS].reduce((a, w) => a + lm.prob(w, prev), 0);
      expect(total).toBeGreaterThan(0.9);
      expect(total).toBeLessThan(1.05);
    }
  });

  it("prefers seen continuations", () => {
    expect(lm.prob("minutes", "twenty")).toBeGreaterThan(lm.prob("milk", "twenty"));
    expect(lm.prob("to", "going")).toBeGreaterThan(lm.prob("two", "going"));
  });

  it("tokenizes", () => {
    expect(tokenize("I'll be HOME, soon!")).toEqual(["i'll", "be", "home", "soon"]);
  });
});

describe("SimulatedDecoder", () => {
  const dec = () => new SimulatedDecoder(new Confusions(vocab), DEFAULT_DECODER, new Rng(3));

  it("produces normalized n-best lists sorted by probability", () => {
    const d = dec();
    for (const w of ["the", "meeting", "friday", "milk"]) {
      const out = d.decode(w);
      expect(out.nbest.length).toBeGreaterThan(1);
      const sum = out.nbest.reduce((a, h) => a + h.p, 0);
      expect(sum).toBeCloseTo(1, 6);
      for (let i = 1; i < out.nbest.length; i++) expect(out.nbest[i]!.p).toBeLessThanOrEqual(out.nbest[0]!.p);
      expect(new Set(out.nbest.map((h) => h.word)).size).toBe(out.nbest.length);
    }
  });

  it("hits its configured top-1 accuracy", () => {
    const d = dec();
    const words = CORPUS.flatMap(tokenize);
    let correct = 0, n = 0;
    for (let r = 0; r < 5; r++) for (const w of words) { n++; if (d.decode(w).nbest[0]!.word === w) correct++; }
    expect(correct / n).toBeGreaterThan(DEFAULT_DECODER.top1 - 0.04);
    expect(correct / n).toBeLessThan(DEFAULT_DECODER.top1 + 0.04);
  });

  it("confuses words with plausible neighbors", () => {
    const c = new Confusions(vocab);
    expect(c.of("monday")).toContain("friday");
    expect(c.of("to").slice(0, 3)).toEqual(expect.arrayContaining(["two", "too"]));
  });
});

describe("rescore", () => {
  it("lets context fix a confident decoder error", () => {
    const lattice = tokenize("the meeting moved to three").map((w) => ({ truth: w, nbest: [{ word: w, p: 0.9 }, { word: "x" + w, p: 0.1 }] }));
    lattice[3] = { truth: "to", nbest: [{ word: "two", p: 0.55 }, { word: "to", p: 0.45 }] };
    const out = rescore(lattice, lm, 0.6);
    expect(out[3]!.rawWord).toBe("two");
    expect(out[3]!.word).toBe("to");
  });

  it("returns posteriors that sum to one at every position", () => {
    const d = new SimulatedDecoder(new Confusions(vocab), DEFAULT_DECODER, new Rng(9));
    const out = rescore(d.decodeSentence(tokenize(CORPUS[5]!)), lm, 0.3);
    for (const w of out) expect(w.alternatives.reduce((a, h) => a + h.p, 0)).toBeCloseTo(1, 6);
  });

  it("with lambda 0 reproduces the decoder's top-1", () => {
    const d = new SimulatedDecoder(new Confusions(vocab), DEFAULT_DECODER, new Rng(2));
    const lattice = d.decodeSentence(tokenize(CORPUS[7]!));
    expect(rescore(lattice, lm, 0).map((w) => w.word)).toEqual(lattice.map((x) => x.nbest[0]!.word));
  });
});

describe("policies", () => {
  it("review never makes the text worse and proofreading is slowest", () => {
    const r = Object.fromEntries(evaluate({ rounds: 2 }).map((x) => [x.policy, x]));
    expect(r.flagged!.wer).toBeLessThan(r.rescored!.wer);
    expect(r.proofread!.wer).toBeLessThanOrEqual(r.flagged!.wer);
    expect(r.proofread!.effectiveWpm).toBeLessThan(r.flagged!.effectiveWpm);
  });

  it("with a perfect decoder the raw output is exact and rescoring rarely overrides it", () => {
    const r = evaluate({ decoder: { ...DEFAULT_DECODER, top1: 1, sharpness: 1 }, rounds: 1 });
    expect(r[0]!.wer).toBe(0);
    expect(r[0]!.effectiveWpm).toBeCloseTo(DEFAULT_TIME.wpm, 5);
    // A language model can talk a confident decoder out of a right answer; it should be rare.
    expect(r[1]!.wer).toBeLessThan(0.01);
  });

  it("higher thresholds flag more and leave fewer errors", () => {
    const s = thresholdSweep({ rounds: 1 }, [0.5, 0.8, 0.95]);
    expect(s[0]!.flagged).toBeLessThan(s[2]!.flagged);
    expect(s[0]!.wer).toBeGreaterThanOrEqual(s[2]!.wer);
  });

  it("moderate LM weight helps; lambda 0 equals the raw decoder", () => {
    const s = lambdaSweep({ rounds: 2 }, [0, 0.3]);
    const raw = evaluate({ rounds: 2 })[0]!;
    expect(s[0]!.wer).toBeCloseTo(raw.wer, 10);
    expect(s[1]!.wer).toBeLessThan(s[0]!.wer);
  });

  it("charges time for every correction", () => {
    const lattice = [{ truth: "home", nbest: [{ word: "hope", p: 0.5 }, { word: "home", p: 0.5 }] }];
    const rs = rescore(lattice, lm, 0);
    const d = new SimulatedDecoder(new Confusions(vocab), DEFAULT_DECODER, new Rng(1));
    const o = runPolicy(lattice, rs, "flagged", 0.9, DEFAULT_TIME, d, new Rng(1));
    expect(o.errors).toBe(0);
    expect(o.seconds).toBeCloseTo(2 + DEFAULT_TIME.glance + DEFAULT_TIME.pick, 6);
  });

  it("rescored posteriors are informative", () => {
    const c = calibration({}, 5);
    const top = c.rescored.at(-1)!, low = c.rescored[1]!;
    expect(top.accuracy!).toBeGreaterThan(low.accuracy!);
  });
});

describe("SessionEngine", () => {
  it("streams words, revises with context, and locks user picks", () => {
    const e = new SessionEngine(0, DEFAULT_DECODER, 0.3, 0.8, 5);
    let s = e.start();
    while (s.words.length < s.target.length) s = e.step(s);
    expect(s.firstSeen).toHaveLength(s.target.length);
    const v = e.view(s);
    expect(v).toHaveLength(s.target.length);
    const i = v.findIndex((w) => w.flagged);
    if (i >= 0) {
      const truth = s.target[i]!;
      const alt = v[i]!.alternatives.find((a) => a.word === truth) ?? v[i]!.alternatives[0]!;
      s = e.pick(s, i, alt.word);
      const after = e.view(s)[i]!;
      expect(after.locked).toBe(true);
      expect(after.word).toBe(alt.word);
      expect(after.flagged).toBe(false);
    }
  });

  it("re-thinking replaces a word's hypotheses and counts as a correction", () => {
    const e = new SessionEngine(3, DEFAULT_DECODER, 0.3, 0.8, 1);
    let s = e.step(e.start());
    const before = s.words[0]!.decoded;
    s = e.rethink(s, 0);
    expect(s.words[0]!.retries).toBe(1);
    expect(s.corrections).toBe(1);
    expect(s.words[0]!.decoded).not.toBe(before);
  });
});
