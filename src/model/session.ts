/**
 * State for one live dictation session: the decoder emits words one at a
 * time, the whole prefix is rescored after each word, and the user can
 * accept, swap, or re-think any word. Pure functions over an immutable state
 * so the React layer stays thin and the behavior is testable.
 */
import { CORPUS } from "./corpus";
import { Confusions, SimulatedDecoder, type DecodedWord, type DecoderParams } from "./decoder";
import { BigramLM, tokenize } from "./lm";
import { rescore, type RescoredWord } from "./rescore";
import { Rng } from "./rng";

export interface WordState {
  decoded: DecodedWord;
  /** Set when the user picked or confirmed a word; rescoring no longer changes it. */
  locked: string | null;
  /** Number of times the user re-thought this word. */
  retries: number;
}

export interface Session {
  sentenceIndex: number;
  target: string[];
  words: WordState[];
  /** The word shown when each position first appeared, to mark later revisions. */
  firstSeen: string[];
  corrections: number;
}

export interface SessionView {
  word: string;
  confidence: number;
  alternatives: { word: string; p: number }[];
  flagged: boolean;
  locked: boolean;
  /** True when context changed this word after it first appeared. */
  revised: boolean;
  correct: boolean;
}

export class SessionEngine {
  readonly lm: BigramLM;
  private readonly decoder: SimulatedDecoder;

  constructor(
    readonly sentenceIndex: number,
    params: DecoderParams,
    readonly lambda: number,
    readonly threshold: number,
    seed: number,
  ) {
    this.lm = new BigramLM(CORPUS.filter((_, j) => j !== sentenceIndex));
    const vocab = [...new BigramLM(CORPUS).vocab];
    this.decoder = new SimulatedDecoder(new Confusions(vocab), params, new Rng(seed));
  }

  start(): Session {
    return { sentenceIndex: this.sentenceIndex, target: tokenize(CORPUS[this.sentenceIndex]!), words: [], firstSeen: [], corrections: 0 };
  }

  /** Decode the next intended word. */
  step(s: Session): Session {
    if (s.words.length >= s.target.length) return s;
    const decoded = this.decoder.decode(s.target[s.words.length]!);
    const next = { ...s, words: [...s.words, { decoded, locked: null, retries: 0 }] };
    return { ...next, firstSeen: [...s.firstSeen, this.view(next).at(-1)!.word] };
  }

  pick(s: Session, i: number, word: string): Session {
    const words = s.words.map((w, k) => (k === i ? { ...w, locked: word } : w));
    const changed = this.view(s)[i]?.word !== word;
    return { ...s, words, corrections: s.corrections + (changed ? 1 : 0) };
  }

  /** The user thinks the word again; the new n-best replaces the old one. */
  rethink(s: Session, i: number): Session {
    const w = s.words[i]!;
    const decoded = this.decoder.decode(w.decoded.truth);
    const words = s.words.map((x, k) => (k === i ? { decoded, locked: null, retries: x.retries + 1 } : x));
    const next = { ...s, words, corrections: s.corrections + 1 };
    return { ...next, firstSeen: next.firstSeen.map((w, k) => (k === i ? this.view(next)[i]!.word : w)) };
  }

  view(s: Session): SessionView[] {
    // Locked words become single-hypothesis columns so they condition their neighbors.
    const lattice = s.words.map((w) => (w.locked ? { truth: w.decoded.truth, nbest: [{ word: w.locked, p: 1 }] } : w.decoded));
    const rs: RescoredWord[] = rescore(lattice, this.lm, this.lambda);
    return rs.map((r, i) => {
      const locked = s.words[i]!.locked !== null;
      const alternatives = locked ? s.words[i]!.decoded.nbest.map((h) => ({ word: h.word, p: h.word === r.word ? 1 : 0 })) : r.alternatives;
      return {
        word: r.word,
        confidence: r.confidence,
        alternatives,
        flagged: !locked && r.confidence < this.threshold,
        locked,
        revised: s.firstSeen[i] !== undefined && s.firstSeen[i] !== r.word,
        correct: r.word === s.words[i]!.decoded.truth,
      };
    });
  }
}
