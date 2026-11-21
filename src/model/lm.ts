/**
 * Interpolated bigram language model with absolute discounting
 * (Kneser-Ney style continuation counts for the unigram backoff).
 *
 * The point is not state of the art; a production system would use a
 * neural LM. The point is the interface: given the previous word, a
 * probability for every candidate, cheap enough to rescore every word as it
 * is decoded.
 */
export const BOS = "<s>";
export const EOS = "</s>";

export class BigramLM {
  private bigram = new Map<string, Map<string, number>>();
  private context = new Map<string, number>();
  private continuation = new Map<string, number>();
  private followers = new Map<string, number>();
  private totalBigramTypes = 0;
  readonly vocab = new Set<string>();

  constructor(
    sentences: string[],
    private readonly discount = 0.75,
  ) {
    for (const s of sentences) {
      const words = [BOS, ...tokenize(s), EOS];
      for (let i = 1; i < words.length; i++) {
        const a = words[i - 1]!, b = words[i]!;
        if (b !== EOS) this.vocab.add(b);
        let row = this.bigram.get(a);
        if (!row) this.bigram.set(a, (row = new Map()));
        const prev = row.get(b) ?? 0;
        row.set(b, prev + 1);
        this.context.set(a, (this.context.get(a) ?? 0) + 1);
        if (prev === 0) {
          this.continuation.set(b, (this.continuation.get(b) ?? 0) + 1);
          this.followers.set(a, (this.followers.get(a) ?? 0) + 1);
          this.totalBigramTypes++;
        }
      }
    }
  }

  /** Continuation probability, with a small floor for unseen words. */
  private pCont(w: string): number {
    const V = this.vocab.size + 1;
    return ((this.continuation.get(w) ?? 0) + 0.5) / (this.totalBigramTypes + 0.5 * V);
  }

  /** P(w | prev). */
  prob(w: string, prev: string): number {
    const c = this.context.get(prev) ?? 0;
    if (c === 0) return this.pCont(w);
    const cab = this.bigram.get(prev)?.get(w) ?? 0;
    const lambda = (this.discount * (this.followers.get(prev) ?? 0)) / c;
    return Math.max(cab - this.discount, 0) / c + lambda * this.pCont(w);
  }

  logprob(w: string, prev: string): number {
    return Math.log(this.prob(w, prev));
  }
}

export function tokenize(s: string): string[] {
  return s.toLowerCase().replace(/[^a-z' ]/g, " ").split(/\s+/).filter(Boolean);
}
