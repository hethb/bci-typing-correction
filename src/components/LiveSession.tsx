import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { CORPUS } from "../model/corpus";
import type { DecoderParams } from "../model/decoder";
import { SessionEngine, type Session } from "../model/session";

interface Props {
  params: DecoderParams;
  lambda: number;
  threshold: number;
}

const pct = (p: number) => `${Math.round(p * 100)}%`;

export function LiveSession({ params, lambda, threshold }: Props) {
  const [sentence, setSentence] = useState(2);
  const [seed, setSeed] = useState(4);
  const [wpm, setWpm] = useState(30);
  const [playing, setPlaying] = useState(true);
  const [showTarget, setShowTarget] = useState(false);
  const [selected, setSelected] = useState<number | null>(null);
  const engine = useMemo(() => new SessionEngine(sentence, params, lambda, threshold, seed), [sentence, params, lambda, threshold, seed]);
  const [session, setSession] = useState<Session>(() => engine.start());
  const [elapsed, setElapsed] = useState(0);
  const tickRef = useRef<number | null>(null);

  useEffect(() => {
    setSession(engine.start());
    setSelected(null);
    setElapsed(0);
    setPlaying(true);
  }, [engine]);

  const done = session.words.length >= session.target.length;
  useEffect(() => {
    if (!playing || done) return;
    tickRef.current = window.setTimeout(() => {
      setSession((s) => engine.step(s));
      setElapsed((e) => e + 60 / wpm);
    }, (60 / wpm) * 1000);
    return () => window.clearTimeout(tickRef.current ?? undefined);
  }, [playing, done, session, engine, wpm]);

  const view = useMemo(() => engine.view(session), [engine, session]);
  const flaggedIdx = view.flatMap((v, i) => (v.flagged ? [i] : []));
  const errors = view.filter((v) => !v.correct).length;

  const pick = useCallback(
    (i: number, word: string) => {
      setSession((s) => engine.pick(s, i, word));
      setElapsed((e) => e + 1.0);
      const next = flaggedIdx.find((k) => k > i);
      setSelected(next ?? null);
    },
    [engine, flaggedIdx],
  );

  const rethink = useCallback(
    (i: number) => {
      setSession((s) => engine.rethink(s, i));
      setElapsed((e) => e + 60 / wpm + 1.0);
    },
    [engine, wpm],
  );

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement).tagName === "INPUT" || (e.target as HTMLElement).tagName === "SELECT") return;
      if (e.key === "Tab" && flaggedIdx.length) {
        e.preventDefault();
        const cur = selected ?? -1;
        const next = e.shiftKey ? [...flaggedIdx].reverse().find((k) => k < cur) ?? flaggedIdx.at(-1)! : flaggedIdx.find((k) => k > cur) ?? flaggedIdx[0]!;
        setSelected(next);
      } else if (selected !== null && /^[1-9]$/.test(e.key)) {
        const alt = view[selected]?.alternatives[Number(e.key) - 1];
        if (alt) pick(selected, alt.word);
      } else if (selected !== null && e.key === "Enter") {
        pick(selected, view[selected]!.word);
      } else if (selected !== null && e.key.toLowerCase() === "r") {
        rethink(selected);
      } else if (e.key === "Escape") {
        setSelected(null);
      } else if (e.key === " ") {
        e.preventDefault();
        setPlaying((p) => !p);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [flaggedIdx, selected, view, pick, rethink]);

  // ?demo opens the first flagged word once decoding finishes (used for screenshots).
  useEffect(() => {
    if (done && selected === null && new URLSearchParams(window.location.search).has("demo")) {
      const first = view.findIndex((v) => v.flagged);
      if (first >= 0) setSelected(first);
    }
    // Only when decoding completes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [done]);

  const sel = selected !== null ? view[selected] : null;
  const correctWords = view.length - errors;

  return (
    <section className="card">
      <div className="card-head">
        <h2>Live dictation</h2>
        <div className="controls">
          <select value={sentence} onChange={(e) => setSentence(Number(e.target.value))} aria-label="Intended sentence">
            {CORPUS.map((s, i) => (
              <option key={i} value={i}>{s}</option>
            ))}
          </select>
          <button onClick={() => { setSentence(Math.floor(Math.random() * CORPUS.length)); setSeed((x) => x + 1); }}>New message</button>
          <button onClick={() => setSeed((x) => x + 1)}>Decode again</button>
          <button className="primary" onClick={() => setPlaying((p) => !p)} disabled={done}>{playing && !done ? "Pause" : "Resume"}</button>
          <label className="inline">
            Speed
            <select value={wpm} onChange={(e) => setWpm(Number(e.target.value))}>
              {[15, 30, 60, 120].map((w) => <option key={w} value={w}>{w} wpm</option>)}
            </select>
          </label>
        </div>
      </div>

      <div className="dictation" aria-live="polite">
        {view.map((v, i) => (
          <button
            key={i}
            className={["word", v.flagged ? "flagged" : "", v.locked ? "locked" : "", v.revised ? "revised" : "", selected === i ? "selected" : "", showTarget && !v.correct ? "wrong" : ""].join(" ")}
            style={{ ["--conf" as string]: v.confidence }}
            onClick={() => setSelected(i === selected ? null : i)}
            title={`confidence ${pct(v.confidence)}`}
          >
            {v.word}
          </button>
        ))}
        {!done && <span className="cursor" aria-hidden />}
      </div>
      {showTarget && <div className="target">Intended: {session.target.join(" ")}</div>}

      <div className="session-foot">
        <div className="stats">
          <span><b>{view.length}</b>/{session.target.length} words</span>
          <span><b>{flaggedIdx.length}</b> flagged</span>
          <span><b>{session.corrections}</b> corrections</span>
          <span><b>{elapsed.toFixed(0)} s</b> simulated time</span>
          <span><b>{elapsed > 0 ? ((correctWords / elapsed) * 60).toFixed(1) : "0.0"}</b> correct wpm</span>
        </div>
        <label className="inline"><input type="checkbox" checked={showTarget} onChange={(e) => setShowTarget(e.target.checked)} /> Show intended text (evaluation only)</label>
      </div>

      {sel && selected !== null && (
        <div className="alts">
          <div className="alts-head">
            Word {selected + 1}: <b>{sel.word}</b> <span className="muted">confidence {pct(sel.confidence)}{sel.revised ? ", changed by context" : ""}</span>
          </div>
          <ol>
            {sel.alternatives.map((a, k) => (
              <li key={a.word}>
                <button onClick={() => pick(selected, a.word)} className={a.word === sel.word ? "current" : ""}>
                  <kbd>{k + 1}</kbd>
                  <span className="alt-word">{a.word}</span>
                  <span className="bar"><span style={{ width: pct(a.p) }} /></span>
                  <span className="muted">{pct(a.p)}</span>
                </button>
              </li>
            ))}
          </ol>
          <div className="alts-actions">
            <button onClick={() => pick(selected, sel.word)}><kbd>Enter</kbd> Keep</button>
            <button onClick={() => rethink(selected)}><kbd>R</kbd> Think it again</button>
          </div>
        </div>
      )}
      <p className="hint">
        Words with low confidence after language-model rescoring are underlined. <kbd>Tab</kbd> jumps between them,
        <kbd>1</kbd>-<kbd>4</kbd> picks an alternative, <kbd>Enter</kbd> keeps the word, <kbd>R</kbd> re-thinks it, <kbd>Space</kbd> pauses.
        Dotted words were revised by later context.
      </p>
    </section>
  );
}
