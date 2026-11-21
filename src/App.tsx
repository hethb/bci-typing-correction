import { useMemo, useState } from "react";
import { Evaluation } from "./components/Evaluation";
import { LiveSession } from "./components/LiveSession";
import { DEFAULT_DECODER, type DecoderParams } from "./model/decoder";
import { DEFAULT_LAMBDA } from "./model/evaluate";

function Slider(props: { label: string; value: number; min: number; max: number; step: number; onChange: (v: number) => void; fmt?: (v: number) => string; help: string }) {
  const { label, value, min, max, step, onChange, fmt, help } = props;
  return (
    <label className="slider" title={help}>
      <span className="slider-top"><span>{label}</span><b>{fmt ? fmt(value) : value}</b></span>
      <input type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} />
      <span className="muted small">{help}</span>
    </label>
  );
}

export default function App() {
  const [top1, setTop1] = useState(DEFAULT_DECODER.top1);
  const [sharpness, setSharpness] = useState(DEFAULT_DECODER.sharpness);
  const [lambda, setLambda] = useState(DEFAULT_LAMBDA);
  const [threshold, setThreshold] = useState(0.8);
  const params: DecoderParams = useMemo(() => ({ ...DEFAULT_DECODER, top1, sharpness }), [top1, sharpness]);

  return (
    <div className="page">
      <header>
        <h1>Correcting thought-to-text</h1>
        <p className="muted">
          A correction interface for a word-by-word neural decoder: confidence-aware highlighting, language-model rescoring,
          one-key alternatives, and a measurement of what each correction policy does to real throughput. The decoder here is
          simulated. Independent project, not affiliated with Sabi.
        </p>
      </header>
      <section className="card knobs">
        <Slider label="Decoder top-1 accuracy" value={top1} min={0.5} max={0.98} step={0.01} onChange={setTop1} fmt={(v) => `${Math.round(v * 100)}%`} help="How often the decoder's first guess is right" />
        <Slider label="Confidence quality" value={sharpness} min={0.1} max={1} step={0.05} onChange={setSharpness} fmt={(v) => v.toFixed(2)} help="How well decoder scores separate right from wrong" />
        <Slider label="Language model weight" value={lambda} min={0} max={1} step={0.05} onChange={setLambda} fmt={(v) => v.toFixed(2)} help="0 uses the decoder alone" />
        <Slider label="Review threshold" value={threshold} min={0.5} max={0.95} step={0.05} onChange={setThreshold} fmt={(v) => v.toFixed(2)} help="Flag words below this confidence" />
      </section>
      <LiveSession params={params} lambda={lambda} threshold={threshold} />
      <Evaluation params={params} lambda={lambda} threshold={threshold} />
      <footer className="muted small">
        Source and method: see the README. All sentences, the decoder model and the language model are synthetic and run in the browser.
      </footer>
    </div>
  );
}
