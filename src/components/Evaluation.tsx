import { useMemo } from "react";
import type { DecoderParams } from "../model/decoder";
import { calibration, evaluate, POLICY_LABEL, thresholdSweep } from "../model/evaluate";

interface Props {
  params: DecoderParams;
  lambda: number;
  threshold: number;
}

function SweepChart({ points, current }: { points: { threshold: number; wer: number; effectiveWpm: number }[]; current: number }) {
  const W = 420, H = 220, l = 44, r = 14, t = 14, b = 36;
  const maxWer = Math.max(0.15, ...points.map((p) => p.wer));
  const wpms = points.map((p) => p.effectiveWpm);
  const lo = Math.floor(Math.min(...wpms) - 1), hi = Math.ceil(Math.max(...wpms) + 1);
  const x = (w: number) => l + (w / maxWer) * (W - l - r);
  const y = (v: number) => t + (1 - (v - lo) / (hi - lo)) * (H - t - b);
  const sorted = [...points].sort((a, c) => a.wer - c.wer);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label="Effective words per minute against word error rate across review thresholds">
      {[0, 0.05, 0.1, 0.15].filter((v) => v <= maxWer + 1e-9).map((v) => (
        <g key={v}>
          <line x1={x(v)} x2={x(v)} y1={t} y2={H - b} className="grid" />
          <text x={x(v)} y={H - b + 16} textAnchor="middle" className="ax">{Math.round(v * 100)}%</text>
        </g>
      ))}
      {Array.from({ length: hi - lo + 1 }, (_, i) => lo + i).filter((v) => v % 1 === 0).map((v) => (
        <g key={v}>
          <line x1={l} x2={W - r} y1={y(v)} y2={y(v)} className="grid" />
          <text x={l - 6} y={y(v) + 4} textAnchor="end" className="ax">{v}</text>
        </g>
      ))}
      <text x={(l + W - r) / 2} y={H - 4} textAnchor="middle" className="ax">word error rate after review</text>
      <text x={12} y={(t + H - b) / 2} textAnchor="middle" className="ax" transform={`rotate(-90 12 ${(t + H - b) / 2})`}>correct wpm</text>
      <polyline fill="none" stroke="var(--accent)" strokeWidth={2} points={sorted.map((p) => `${x(p.wer)},${y(p.effectiveWpm)}`).join(" ")} />
      {points.map((p, i) => (
        <g key={p.threshold}>
          <circle cx={x(p.wer)} cy={y(p.effectiveWpm)} r={p.threshold === current ? 6 : 4} fill={p.threshold === current ? "var(--accent)" : "var(--card)"} stroke="var(--accent)" strokeWidth={2} />
          <text x={x(p.wer) + (i % 2 ? 9 : -9)} y={y(p.effectiveWpm) + (i % 2 ? 14 : -8)} textAnchor={i % 2 ? "start" : "end"} className="ax">{p.threshold}</text>
        </g>
      ))}
    </svg>
  );
}

function Calibration({ data }: { data: ReturnType<typeof calibration> }) {
  const W = 420, H = 180, l = 44, r = 10, t = 10, b = 34;
  const bins = data.raw.length;
  const bw = (W - l - r) / bins;
  const y = (v: number) => t + (1 - v) * (H - t - b);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label="Accuracy by confidence bucket for raw decoder scores and rescored posteriors">
      {[0, 0.5, 1].map((v) => (
        <g key={v}>
          <line x1={l} x2={W - r} y1={y(v)} y2={y(v)} className="grid" />
          <text x={l - 6} y={y(v) + 4} textAnchor="end" className="ax">{v * 100}%</text>
        </g>
      ))}
      <line x1={l} y1={y(0)} x2={W - r} y2={y(1)} stroke="var(--muted)" strokeDasharray="3 4" />
      {data.raw.map((bRaw, i) => {
        const bPost = data.rescored[i]!;
        const cx = l + i * bw;
        return (
          <g key={i}>
            {bRaw.accuracy !== null && bRaw.n >= 5 && <rect x={cx + bw * 0.18} y={y(bRaw.accuracy)} width={bw * 0.3} height={y(0) - y(bRaw.accuracy)} fill="var(--muted-2)"><title>raw {bRaw.n} words</title></rect>}
            {bPost.accuracy !== null && bPost.n >= 5 && <rect x={cx + bw * 0.52} y={y(bPost.accuracy)} width={bw * 0.3} height={y(0) - y(bPost.accuracy)} fill="var(--accent)"><title>rescored {bPost.n} words</title></rect>}
            <text x={cx + bw / 2} y={H - b + 16} textAnchor="middle" className="ax">{Math.round(bRaw.lo * 100)}-{Math.round(bRaw.hi * 100)}%</text>
          </g>
        );
      })}
    </svg>
  );
}

export function Evaluation({ params, lambda, threshold }: Props) {
  const results = useMemo(() => evaluate({ decoder: params, lambda, threshold, rounds: 3 }), [params, lambda, threshold]);
  const sweep = useMemo(() => thresholdSweep({ decoder: params, lambda, rounds: 2 }, [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]), [params, lambda]);
  const calib = useMemo(() => calibration({ decoder: params, lambda }), [params, lambda]);
  const raw = results[0]!;
  return (
    <section className="card">
      <div className="card-head"><h2>What correction buys, measured</h2></div>
      <p className="muted small">
        Every sentence in the corpus is decoded three times with fresh noise, using a language model trained on the other sentences only.
        Each policy is run on the same decoded words. Correct wpm counts only words that end up right, including time spent fixing.
      </p>
      <table>
        <thead>
          <tr><th>Policy</th><th>Word error rate</th><th>Correct wpm</th><th>Flagged / message</th></tr>
        </thead>
        <tbody>
          {results.map((r) => (
            <tr key={r.policy} className={r.policy === "flagged" ? "highlight" : ""}>
              <td>{POLICY_LABEL[r.policy]}</td>
              <td>{(r.wer * 100).toFixed(1)}%</td>
              <td>{r.effectiveWpm.toFixed(1)} <span className="muted small">{r.policy !== "raw" ? `${r.effectiveWpm >= raw.effectiveWpm ? "+" : ""}${(r.effectiveWpm - raw.effectiveWpm).toFixed(1)}` : ""}</span></td>
              <td>{r.flaggedPerSentence.toFixed(1)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="two">
        <div>
          <h3>Review threshold trade-off</h3>
          <SweepChart points={sweep} current={threshold} />
          <p className="muted small">Each point is a confidence threshold for flagging. Higher thresholds flag more words: fewer errors survive, at some cost in speed.</p>
        </div>
        <div>
          <h3>Is the confidence trustworthy?</h3>
          <Calibration data={calib} />
          <p className="muted small"><span className="sw muted-2" /> raw decoder score <span className="sw accent" /> posterior after rescoring. Bars on the dashed line mean a word marked 80% confident is right about 80% of the time.</p>
        </div>
      </div>
    </section>
  );
}
