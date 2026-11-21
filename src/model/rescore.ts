/**
 * Combine decoder n-best lists with a language model.
 *
 * Each decoded word is a small set of hypotheses with decoder probabilities.
 * Treating the sentence as a lattice (one column per word, one node per
 * hypothesis), the score of a path is
 *
 *     sum_i  log p_dec(w_i)  +  lambda * log p_lm(w_i | w_{i-1})
 *
 * Viterbi gives the best path. Forward-backward over the same lattice gives,
 * for every position, the posterior of each hypothesis given the whole
 * sentence so far, which is a far better confidence signal than the decoder's
 * own score: "the meeting moved to tree" has a confident decoder and a very
 * unhappy language model.
 *
 * The lattice is decoded left to right, so it also works incrementally: when
 * a new word arrives, earlier words can change, which is what a live
 * correction UI shows.
 */
import { BOS, type BigramLM } from "./lm";
import type { DecodedWord, Hypothesis } from "./decoder";

export interface RescoredWord {
  /** Chosen word after rescoring. */
  word: string;
  /** Posterior of the chosen word, 0..1. */
  confidence: number;
  /** All hypotheses re-ranked by posterior. */
  alternatives: Hypothesis[];
  /** Decoder's own top-1, for comparison. */
  rawWord: string;
  rawConfidence: number;
}

function logSumExp(xs: number[]): number {
  const m = Math.max(...xs);
  if (m === -Infinity) return m;
  return m + Math.log(xs.reduce((a, x) => a + Math.exp(x - m), 0));
}

export function rescore(lattice: DecodedWord[], lm: BigramLM, lambda = 0.6): RescoredWord[] {
  const n = lattice.length;
  if (!n) return [];
  const cols = lattice.map((d) => d.nbest);
  const emit = cols.map((c) => c.map((h) => Math.log(Math.max(h.p, 1e-12))));
  const trans = (prev: string, w: string) => lambda * lm.logprob(w, prev);

  // Viterbi
  const score: number[][] = [];
  const back: number[][] = [];
  score[0] = cols[0]!.map((h, j) => emit[0]![j]! + trans(BOS, h.word));
  back[0] = cols[0]!.map(() => -1);
  for (let i = 1; i < n; i++) {
    score[i] = [];
    back[i] = [];
    cols[i]!.forEach((h, j) => {
      let best = -Infinity, arg = 0;
      cols[i - 1]!.forEach((p, k) => {
        const s = score[i - 1]![k]! + trans(p.word, h.word);
        if (s > best) { best = s; arg = k; }
      });
      score[i]![j] = best + emit[i]![j]!;
      back[i]![j] = arg;
    });
  }
  const path = new Array<number>(n);
  let last = score[n - 1]!.indexOf(Math.max(...score[n - 1]!));
  for (let i = n - 1; i >= 0; i--) {
    path[i] = last;
    last = back[i]![last]!;
  }

  // Forward-backward (log space)
  const fwd: number[][] = [];
  fwd[0] = cols[0]!.map((h, j) => emit[0]![j]! + trans(BOS, h.word));
  for (let i = 1; i < n; i++) {
    fwd[i] = cols[i]!.map((h, j) => emit[i]![j]! + logSumExp(cols[i - 1]!.map((p, k) => fwd[i - 1]![k]! + trans(p.word, h.word))));
  }
  const bwd: number[][] = new Array(n);
  bwd[n - 1] = cols[n - 1]!.map(() => 0);
  for (let i = n - 2; i >= 0; i--) {
    bwd[i] = cols[i]!.map((h) => logSumExp(cols[i + 1]!.map((nx, k) => trans(h.word, nx.word) + emit[i + 1]![k]! + bwd[i + 1]![k]!)));
  }

  return cols.map((c, i) => {
    const logs = c.map((_, j) => fwd[i]![j]! + bwd[i]![j]!);
    const z = logSumExp(logs);
    const post = logs.map((l) => Math.exp(l - z));
    const alternatives = c.map((h, j) => ({ word: h.word, p: post[j]! })).sort((a, b) => b.p - a.p);
    const chosen = c[path[i]!]!.word;
    return {
      word: chosen,
      confidence: post[path[i]!]!,
      alternatives,
      rawWord: c[0]!.word,
      rawConfidence: c[0]!.p,
    };
  });
}
