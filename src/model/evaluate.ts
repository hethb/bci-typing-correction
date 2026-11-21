/**
 * What does a correction interface buy in words per minute?
 *
 * A raw decoding speed (around 30 words per minute for a thought-to-text
 * decoder) only matters if the text is right. This module simulates a user
 * finishing a message under different correction policies and measures the
 * resulting accuracy and effective throughput: correct words per minute,
 * including the time spent fixing things.
 *
 * Time model (seconds, configurable):
 *   decode        time to think each word (60 / wpm)
 *   glance        time to check one highlighted word
 *   pick          time to select an alternative from the list
 *   rethink       re-decoding a word that is not in the list (decode + overhead)
 *   proofread     time per word to read the whole message
 *
 * Users only fix errors they notice. Words that are not highlighted are
 * caught only when the user proofreads, and then only with probability
 * `proofreadCatch`.
 */
import { CORPUS } from "./corpus";
import { Confusions, DEFAULT_DECODER, SimulatedDecoder, type DecodedWord, type DecoderParams } from "./decoder";
import { BigramLM, tokenize } from "./lm";
import { rescore, type RescoredWord } from "./rescore";
import { Rng } from "./rng";

/** LM weight; chosen by the sweep in the README (lambdaSweep). */
export const DEFAULT_LAMBDA = 0.3;

export type Policy = "raw" | "rescored" | "flagged" | "proofread";

export const POLICY_LABEL: Record<Policy, string> = {
  raw: "Accept decoder output",
  rescored: "Language-model rescoring only",
  flagged: "Rescoring + review flagged words",
  proofread: "Rescoring + proofread every word",
};

export interface TimeModel {
  wpm: number;
  glance: number;
  pick: number;
  rethinkOverhead: number;
  proofread: number;
  proofreadCatch: number;
}

export const DEFAULT_TIME: TimeModel = { wpm: 30, glance: 0.6, pick: 1.0, rethinkOverhead: 1.0, proofread: 0.35, proofreadCatch: 0.9 };

export interface Outcome {
  words: number;
  errors: number;
  seconds: number;
  flagged: number;
  corrections: number;
}

export interface PolicyResult {
  policy: Policy;
  wer: number;
  effectiveWpm: number;
  flaggedPerSentence: number;
  correctionsPerSentence: number;
}

/** Simulate one sentence under one policy. */
export function runPolicy(
  lattice: DecodedWord[],
  rescored: RescoredWord[],
  policy: Policy,
  threshold: number,
  time: TimeModel,
  decoder: SimulatedDecoder,
  rng: Rng,
): Outcome {
  const perWord = 60 / time.wpm;
  let seconds = lattice.length * perWord;
  let flagged = 0, corrections = 0;
  const final = lattice.map((d, i) => (policy === "raw" ? d.nbest[0]!.word : rescored[i]!.word));

  const fix = (i: number) => {
    const truth = lattice[i]!.truth;
    if (final[i] === truth) return;
    if (rescored[i]!.alternatives.some((a) => a.word === truth)) {
      seconds += time.pick;
      final[i] = truth;
      corrections++;
      return;
    }
    // Not in the list: think the word again, up to three tries.
    for (let attempt = 0; attempt < 3; attempt++) {
      seconds += perWord + time.rethinkOverhead;
      corrections++;
      const again = decoder.decode(truth);
      if (again.nbest.some((h) => h.word === truth)) {
        seconds += again.nbest[0]!.word === truth ? 0 : time.pick;
        final[i] = truth;
        return;
      }
    }
  };

  if (policy === "flagged" || policy === "proofread") {
    lattice.forEach((_, i) => {
      if (rescored[i]!.confidence < threshold) {
        flagged++;
        seconds += time.glance;
        fix(i);
      }
    });
  }
  if (policy === "proofread") {
    seconds += lattice.length * time.proofread;
    lattice.forEach((d, i) => {
      if (final[i] !== d.truth && rng.next() < time.proofreadCatch) fix(i);
    });
  }
  const errors = final.filter((w, i) => w !== lattice[i]!.truth).length;
  return { words: lattice.length, errors, seconds, flagged, corrections };
}

export interface EvalOptions {
  decoder?: DecoderParams;
  time?: TimeModel;
  lambda?: number;
  threshold?: number;
  seed?: number;
  /** Repeat the corpus this many times with different decoder noise. */
  rounds?: number;
}

/**
 * Leave-one-out evaluation: for each sentence, train the LM on all the others,
 * decode the sentence, and run every policy on the same decoded lattice.
 */
export function evaluate(opts: EvalOptions = {}): PolicyResult[] {
  const time = opts.time ?? DEFAULT_TIME;
  const threshold = opts.threshold ?? 0.8;
  const rounds = opts.rounds ?? 3;
  const seed = opts.seed ?? 11;
  // Separate streams: the decoded lattices are identical for any lambda or
  // threshold, so comparisons across settings are paired.
  const rng = new Rng(seed);
  const userRng = new Rng(seed ^ 0x9e3779b9);
  const totals = new Map<Policy, Outcome>();
  const policies: Policy[] = ["raw", "rescored", "flagged", "proofread"];
  for (const p of policies) totals.set(p, { words: 0, errors: 0, seconds: 0, flagged: 0, corrections: 0 });
  const lms = CORPUS.map((_, i) => new BigramLM(CORPUS.filter((__, j) => j !== i)));
  const vocab = [...new BigramLM(CORPUS).vocab];
  const confusions = new Confusions(vocab);
  const decoder = new SimulatedDecoder(confusions, opts.decoder ?? DEFAULT_DECODER, rng);
  const redecoder = new SimulatedDecoder(confusions, opts.decoder ?? DEFAULT_DECODER, new Rng(seed + 1));

  let sentences = 0;
  for (let r = 0; r < rounds; r++) {
    CORPUS.forEach((s, i) => {
      const lattice = decoder.decodeSentence(tokenize(s));
      const rs = rescore(lattice, lms[i]!, opts.lambda ?? DEFAULT_LAMBDA);
      for (const p of policies) {
        const o = runPolicy(lattice, rs, p, threshold, time, redecoder, userRng);
        const t = totals.get(p)!;
        t.words += o.words; t.errors += o.errors; t.seconds += o.seconds; t.flagged += o.flagged; t.corrections += o.corrections;
      }
      sentences++;
    });
  }
  return policies.map((policy) => {
    const t = totals.get(policy)!;
    return {
      policy,
      wer: t.errors / t.words,
      effectiveWpm: ((t.words - t.errors) / t.seconds) * 60,
      flaggedPerSentence: t.flagged / sentences,
      correctionsPerSentence: t.corrections / sentences,
    };
  });
}

/** Effective WPM and WER of the flagged policy across review thresholds. */
export function thresholdSweep(opts: EvalOptions = {}, thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]) {
  return thresholds.map((threshold) => {
    const r = evaluate({ ...opts, threshold }).find((x) => x.policy === "flagged")!;
    return { threshold, wer: r.wer, effectiveWpm: r.effectiveWpm, flagged: r.flaggedPerSentence };
  });
}

/**
 * Calibration of a confidence signal: bucket words by confidence and report
 * accuracy per bucket. Used to show that LM posteriors are better calibrated
 * than raw decoder scores.
 */
export function calibration(opts: EvalOptions = {}, bins = 5) {
  const rng = new Rng(opts.seed ?? 11);
  const vocab = [...new BigramLM(CORPUS).vocab];
  const decoder = new SimulatedDecoder(new Confusions(vocab), opts.decoder ?? DEFAULT_DECODER, rng);
  const raw = Array.from({ length: bins }, () => ({ n: 0, correct: 0 }));
  const post = Array.from({ length: bins }, () => ({ n: 0, correct: 0 }));
  const bin = (p: number) => Math.min(bins - 1, Math.floor(p * bins));
  CORPUS.forEach((s, i) => {
    const lm = new BigramLM(CORPUS.filter((_, j) => j !== i));
    const lattice = decoder.decodeSentence(tokenize(s));
    rescore(lattice, lm, opts.lambda ?? DEFAULT_LAMBDA).forEach((w, k) => {
      const truth = lattice[k]!.truth;
      const rb = raw[bin(w.rawConfidence)]!; rb.n++; if (w.rawWord === truth) rb.correct++;
      const pb = post[bin(w.confidence)]!; pb.n++; if (w.word === truth) pb.correct++;
    });
  });
  const fmt = (xs: { n: number; correct: number }[]) => xs.map((x, i) => ({ lo: i / bins, hi: (i + 1) / bins, n: x.n, accuracy: x.n ? x.correct / x.n : null }));
  return { raw: fmt(raw), rescored: fmt(post) };
}

/** WER of LM rescoring alone across LM weights. Lambda 0 is the raw decoder. */
export function lambdaSweep(opts: EvalOptions = {}, lambdas = [0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0]) {
  return lambdas.map((lambda) => {
    const r = evaluate({ ...opts, lambda }).find((x) => x.policy === "rescored")!;
    return { lambda, wer: r.wer };
  });
}
