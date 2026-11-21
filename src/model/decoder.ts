/**
 * A stand-in for a thought-to-text decoder.
 *
 * A real non-invasive decoder emits one word at a time with uncertainty. This
 * simulator reproduces the properties a correction interface has to cope
 * with:
 *
 *   - top-1 accuracy well below 100 percent (configurable)
 *   - errors that are not random: the wrong word is usually a near neighbor
 *     of the right one, by spelling or sound (then/than, for/four) or by
 *     meaning (morning/evening, monday/friday)
 *   - an n-best list in which the correct word is often present even when it
 *     is not first
 *   - confidence scores that are informative but imperfectly calibrated
 *
 * Every number is seeded so runs are reproducible.
 */
import { Rng } from "./rng";

export interface Hypothesis {
  word: string;
  p: number;
}

export interface DecodedWord {
  /** What the user intended. Only the simulator and the evaluator see this. */
  truth: string;
  /** Decoder n-best, highest first, probabilities summing to 1. */
  nbest: Hypothesis[];
}

export interface DecoderParams {
  /** Probability the top hypothesis is correct. */
  top1: number;
  /** Probability the correct word appears somewhere in the n-best when top-1 is wrong. */
  inNbest: number;
  nbestSize: number;
  /** 0 = confidences carry no information, 1 = well separated. */
  sharpness: number;
}

export const DEFAULT_DECODER: DecoderParams = { top1: 0.8, inNbest: 0.75, nbestSize: 4, sharpness: 0.7 };

/** Hand-picked semantic and phonetic neighbors that a decoder could plausibly confuse. */
const SEMANTIC: string[][] = [
  ["morning", "evening", "afternoon", "night", "tonight"],
  ["monday", "friday", "saturday", "tomorrow", "today"],
  ["three", "six", "seven", "eight", "nine", "ten", "twenty"],
  ["mom", "dad", "boss"],
  ["milk", "bread", "coffee", "tea"],
  ["home", "office", "school", "store", "station"],
  ["call", "text", "message", "send", "email"],
  ["late", "early", "earlier", "soon", "later"],
  ["dinner", "lunch", "breakfast"],
  ["car", "train", "flight", "bus"],
  ["dog", "cat", "kids"],
  ["leave", "arrive", "start", "finish"],
  ["good", "fine", "better", "great"],
  ["need", "want", "have"],
  ["meeting", "class", "call", "game", "party"],
  ["to", "two", "too"],
  ["for", "four"],
  ["the", "a", "this", "that"],
  ["is", "was", "are"],
  ["i", "we", "you"],
  ["in", "on", "at"],
  ["will", "would", "can", "should"],
  ["be", "get"],
  ["me", "you", "us"],
];

function editDistance(a: string, b: string): number {
  const dp = Array.from({ length: a.length + 1 }, (_, i) => [i, ...new Array<number>(b.length).fill(0)]);
  for (let j = 1; j <= b.length; j++) dp[0]![j] = j;
  for (let i = 1; i <= a.length; i++) {
    for (let j = 1; j <= b.length; j++) {
      dp[i]![j] = Math.min(dp[i - 1]![j]! + 1, dp[i]![j - 1]! + 1, dp[i - 1]![j - 1]! + (a[i - 1] === b[j - 1] ? 0 : 1));
    }
  }
  return dp[a.length]![b.length]!;
}

export class Confusions {
  private cache = new Map<string, string[]>();
  constructor(private readonly vocab: string[]) {}

  /** Plausible confusions for a word, nearest first. */
  of(word: string): string[] {
    const hit = this.cache.get(word);
    if (hit) return hit;
    const semantic = new Set(SEMANTIC.filter((g) => g.includes(word)).flat());
    semantic.delete(word);
    const spelled = this.vocab
      .filter((w) => w !== word && !semantic.has(w))
      .map((w) => ({ w, d: editDistance(w, word) / Math.max(w.length, word.length) }))
      .filter((x) => x.d <= 0.5)
      .sort((a, b) => a.d - b.d || a.w.localeCompare(b.w))
      .map((x) => x.w);
    const out = [...semantic, ...spelled];
    if (out.length < 3) {
      const filler = this.vocab.filter((w) => w !== word && !out.includes(w)).sort((a, b) => Math.abs(a.length - word.length) - Math.abs(b.length - word.length));
      out.push(...filler.slice(0, 3 - out.length));
    }
    this.cache.set(word, out);
    return out;
  }
}

export class SimulatedDecoder {
  constructor(
    private readonly confusions: Confusions,
    private readonly params: DecoderParams = DEFAULT_DECODER,
    private readonly rng: Rng = new Rng(1),
  ) {}

  decode(truth: string): DecodedWord {
    const { top1, inNbest, nbestSize, sharpness } = this.params;
    const rng = this.rng;
    const neighbors = this.confusions.of(truth);
    const pick = (exclude: Set<string>) => {
      // Prefer near neighbors, but not always the nearest.
      const pool = neighbors.filter((w) => !exclude.has(w));
      if (!pool.length) return null;
      const i = Math.min(pool.length - 1, Math.floor(-Math.log(Math.max(rng.next(), 1e-9)) * 1.5));
      return pool[i]!;
    };

    const correct = rng.next() < top1;
    const words: string[] = [];
    const used = new Set<string>([truth]);
    if (correct) {
      words.push(truth);
    } else {
      const wrong = pick(used);
      if (wrong) { words.push(wrong); used.add(wrong); } else words.push(truth);
    }
    const truthInList = correct || rng.next() < inNbest;
    while (words.length < nbestSize) {
      if (!words.includes(truth) && truthInList && (words.length === 1 || rng.next() < 0.6)) {
        words.push(truth);
        continue;
      }
      const w = pick(used);
      if (!w) break;
      words.push(w);
      used.add(w);
    }
    if (truthInList && !words.includes(truth)) words[words.length - 1] = truth;

    // Scores: the top hypothesis gets a margin that depends on whether it is
    // right, blurred by noise. Lower sharpness means less separation.
    const margin = (correct ? 2.2 : 0.7) * sharpness + rng.normal() * 0.9 * (1.3 - sharpness);
    const logits = words.map((_, i) => (i === 0 ? Math.max(margin, 0.05) : -0.5 * i + rng.normal() * 0.25));
    for (let i = 1; i < logits.length; i++) logits[i] = Math.min(logits[i]!, logits[0]! - 0.01);
    const z = logits.map((l) => Math.exp(l));
    const sum = z.reduce((a, b) => a + b, 0);
    return { truth, nbest: words.map((w, i) => ({ word: w, p: z[i]! / sum })) };
  }

  decodeSentence(words: string[]): DecodedWord[] {
    return words.map((w) => this.decode(w));
  }
}
