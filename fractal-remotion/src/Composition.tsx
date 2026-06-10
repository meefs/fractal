import {
  AbsoluteFill,
  Easing,
  Img,
  interpolate,
  staticFile,
  useCurrentFrame,
} from "remotion";
import { loadFont } from "@remotion/google-fonts/JetBrainsMono";

const { fontFamily: MONO } = loadFont("normal", {
  weights: ["400", "500", "700"],
});

// Sampled from the logo PNG so the mark blends into the canvas.
const BG = "#160230";

const C = {
  bg: BG,
  text: "#ece4ff",
  lavender: "#c4b5fd",
  violet: "#a78bfa",
  accent: "#8b5cf6",
  dim: "#71619e",
  faint: "#4a3a78",
  border: "rgba(167, 139, 250, 0.30)",
};

const clamp = {
  extrapolateLeft: "clamp" as const,
  extrapolateRight: "clamp" as const,
};

const easeOut = Easing.bezier(0.16, 1, 0.3, 1);

const fadeIn = (frame: number, start: number, dur = 12) =>
  interpolate(frame, [start, start + dur], [0, 1], { ...clamp, easing: easeOut });

const typed = (text: string, frame: number, start: number, cps = 1.4) => {
  const chars = Math.floor(
    interpolate(frame, [start, start + text.length / cps], [0, text.length], clamp),
  );
  return text.slice(0, chars);
};

const blink = (frame: number) => (frame % 16 < 9 ? 1 : 0);

// Terminal redraws don't fade; chrome appears the frame it is printed.
const on = (frame: number, at: number) => (frame >= at ? 1 : 0);

const Cursor: React.FC<{ opacity: number }> = ({ opacity }) => (
  <span
    style={{
      display: "inline-block",
      width: "0.52em",
      height: "1.02em",
      marginLeft: "0.06em",
      verticalAlign: "-0.12em",
      background: "#a78bfa",
      opacity,
    }}
  />
);

const SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧";

// ---------------------------------------------------------------------------
// Self-similar geometry: each session embeds the next one as the whole canvas
// scaled by CHILD_SCALE, top-left pinned at T — indented in the log flow,
// right below the "spawning sub-lm" line — so the recursion reads as session
// output, not a floating window. P is the fixed point: T = P * (1 - scale).
// ---------------------------------------------------------------------------
const CHILD_SCALE = 0.38;
const T = { x: 82, y: 568 };
const P = { x: T.x / (1 - CHILD_SCALE), y: T.y / (1 - CHILD_SCALE) };

// The cold open owns frames 0-74; the whole session timeline shifts by this.
const SESSION_SHIFT = 86;
const SESSION_APPEAR = 100;
// Frame offset between a session's flow and the flow of the session it spawns.
const FLOW_SHIFT = 144;
const CHILD_AT = 112;
const ZOOM_START = 134 + SESSION_SHIFT;
const ZOOM_END = 182 + SESSION_SHIFT;

type Line = {
  at: number;
  text: string;
  color?: string;
  indent?: number;
  bullet?: string;
};

type CodeSegment = { text: string; color: string };

const PY = {
  kw: C.violet,
  fn: C.text,
  str: C.lavender,
  num: C.lavender,
  plain: "#b3a6d6",
  punct: C.dim,
};

const PROMPT_AT = 42;

type Script = {
  prompt: string;
  before: Line[];
  code: { at: number; segments: CodeSegment[] }[];
  after: Line[];
};

// Depth 0 fans the big task out; every level below works one shard of it.
const SCRIPTS: Script[] = [
  {
    prompt: "inspect this 123 page contract and extract key points",
    before: [
      { at: 54, text: "RLM turn 1/30 (ok)", color: C.violet },
      { at: 60, text: "reasoning: 123 pages won't fit one context — split it", indent: 1 },
      { at: 66, text: "python: fanning out one sub-lm per page", indent: 1 },
    ],
    code: [
      {
        at: 70,
        segments: [
          { text: "pages = load(", color: PY.plain },
          { text: '"contract.pdf"', color: PY.str },
          { text: ").pages", color: PY.plain },
        ],
      },
      {
        at: 78,
        segments: [
          { text: "for ", color: PY.kw },
          { text: "page ", color: PY.plain },
          { text: "in ", color: PY.kw },
          { text: "pages:", color: PY.plain },
        ],
      },
      {
        at: 86,
        segments: [
          { text: "    spawn(", color: PY.plain },
          { text: '"extract key points"', color: PY.str },
          { text: ", doc=page)", color: PY.plain },
        ],
      },
    ],
    after: [
      { at: 94, text: "123 sub-lms queued", indent: 1, color: C.lavender },
      { at: 100, text: "RLM turn 2/30 (ok)", color: C.violet },
      { at: 106, text: "sub-lm 1/123 · page 1", indent: 1, color: C.text, bullet: "↳ " },
    ],
  },
  {
    prompt: "here is one page, extract key points",
    before: [
      { at: 54, text: "RLM turn 1/30 (ok)", color: C.violet },
      { at: 60, text: "reasoning: scan the clauses, pull terms and obligations", indent: 1 },
      { at: 66, text: "python: collecting key points", indent: 1 },
    ],
    code: [
      {
        at: 70,
        segments: [{ text: "clauses = parse(page).clauses", color: PY.plain }],
      },
      {
        at: 78,
        segments: [
          { text: "points = [summarize(c) ", color: PY.plain },
          { text: "for ", color: PY.kw },
          { text: "c ", color: PY.plain },
          { text: "in ", color: PY.kw },
          { text: "clauses]", color: PY.plain },
        ],
      },
      {
        at: 86,
        segments: [
          { text: "emit", color: PY.fn },
          { text: "(points)", color: PY.plain },
        ],
      },
    ],
    after: [
      { at: 94, text: "3 key points · payment · term · liability", indent: 1, color: C.lavender },
      { at: 100, text: "RLM turn 2/30 (ok)", color: C.violet },
      { at: 106, text: "returning key points to parent", indent: 1, color: C.text, bullet: "↳ " },
    ],
  },
];

const typedSegments = (
  segments: CodeSegment[],
  frame: number,
  start: number,
  cps = 3,
) => {
  const total = segments.reduce((sum, segment) => sum + segment.text.length, 0);
  let visible = Math.floor(
    interpolate(frame, [start, start + total / cps], [0, total], clamp),
  );
  const out: CodeSegment[] = [];
  for (const segment of segments) {
    if (visible <= 0) break;
    out.push({ ...segment, text: segment.text.slice(0, visible) });
    visible -= segment.text.length;
  }
  return out;
};

const LogLine: React.FC<{ line: Line; frame: number; offset: number }> = ({
  line,
  frame,
  offset,
}) => {
  const at = line.at + offset;
  const opacity = fadeIn(frame, at, 8);
  const y = interpolate(frame, [at, at + 10], [8, 0], { ...clamp, easing: easeOut });
  return (
    <div
      style={{
        opacity,
        transform: `translateY(${y}px)`,
        color: line.color ?? C.dim,
        paddingLeft: (line.indent ?? 0) * 38,
        fontSize: 27,
        lineHeight: 1.65,
        whiteSpace: "pre",
      }}
    >
      {line.bullet ? <span style={{ color: C.accent }}>{line.bullet}</span> : null}
      {typed(line.text, frame, at, 2.4)}
    </div>
  );
};

const PromptLine: React.FC<{
  frame: number;
  offset: number;
  appear: number;
  text: string;
}> = ({ frame, offset, appear, text }) => {
  const at = PROMPT_AT + offset;
  const doneAt = at + text.length / 2.4 + 10;
  return (
    <div
      style={{
        opacity: on(frame, appear + 2),
        color: C.text,
        fontSize: 27,
        lineHeight: 1.65,
        whiteSpace: "pre",
      }}
    >
      <span style={{ color: C.accent }}>fractal› </span>
      {typed(text, frame, at, 2.4)}
      <Cursor opacity={frame < doneAt ? blink(frame) : 0} />
    </div>
  );
};

const CodeBlock: React.FC<{
  frame: number;
  offset: number;
  code: Script["code"];
}> = ({ frame, offset, code }) => (
  <div
    style={{
      margin: "8px 0 8px 38px",
      padding: "12px 26px",
      borderLeft: `3px solid rgba(139, 92, 246, 0.55)`,
      background: "rgba(167, 139, 250, 0.05)",
      opacity: fadeIn(frame, code[0].at + offset - 4, 8),
    }}
  >
    {code.map((line, i) => (
      <div
        key={i}
        style={{
          fontSize: 25,
          lineHeight: 1.6,
          whiteSpace: "pre",
          opacity: fadeIn(frame, line.at + offset, 6),
        }}
      >
        {typedSegments(line.segments, frame, line.at + offset).map((segment, j) => (
          <span key={j} style={{ color: segment.color }}>
            {segment.text}
          </span>
        ))}
      </div>
    ))}
  </div>
);

const Footer: React.FC<{
  frame: number;
  offset: number;
  appear: number;
  idleAfter?: number;
}> = ({ frame, offset, appear, idleAfter }) => {
  const running =
    frame >= SCRIPTS[0].before[0].at + offset &&
    (idleAfter === undefined || frame < idleAfter);
  const spinner = SPINNER[Math.floor(frame / 3) % SPINNER.length];
  return (
    <div
      style={{
        position: "absolute",
        left: 0,
        right: 0,
        bottom: 0,
        height: 56,
        display: "flex",
        alignItems: "center",
        padding: "0 28px",
        borderTop: `1px solid ${C.border}`,
        color: C.faint,
        fontSize: 22,
        opacity: on(frame, appear),
        whiteSpace: "pre",
      }}
    >
      {running ? (
        <>
          <span style={{ color: C.dim }}>{`${spinner} running RLM`}</span>
          <span>{" · model "}</span>
          <span style={{ color: C.dim }}>fable-5</span>
          <span>{" · sub "}</span>
          <span style={{ color: C.dim }}>haiku-4.5</span>
        </>
      ) : (
        <>
          <span>{"model "}</span>
          <span style={{ color: C.dim }}>fable-5</span>
          <span>{" · sub "}</span>
          <span style={{ color: C.dim }}>haiku-4.5</span>
          <span>{" · verbose off"}</span>
        </>
      )}
    </div>
  );
};

const Session: React.FC<{ depth: number; offset: number; appear: number }> = ({
  depth,
  offset,
  appear,
}) => {
  const frame = useCurrentFrame();
  const opacity = depth === 0 ? on(frame, appear) : 1;
  const script = SCRIPTS[Math.min(depth, SCRIPTS.length - 1)];

  return (
    <AbsoluteFill style={{ opacity }}>
      <AbsoluteFill style={{ background: C.bg }}>
        {/* header */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            height: 58,
            padding: "0 28px",
            borderBottom: `1px solid ${C.border}`,
            color: C.faint,
            fontSize: 22,
            opacity: on(frame, appear),
          }}
        >
          <span>
            <span style={{ color: C.violet }}>fractal</span>
            {` · session ${["8f31c2", "c09d4e", "77ab10"][depth % 3]}`}
          </span>
          <span>{`~/deals/acme · depth ${depth}`}</span>
        </div>

        {/* log stream */}
        <div style={{ padding: "30px 44px" }}>
          <PromptLine
            frame={frame}
            offset={offset}
            appear={appear}
            text={script.prompt}
          />
          {script.before.map((line) => (
            <LogLine key={line.at} line={line} frame={frame} offset={offset} />
          ))}
          <CodeBlock frame={frame} offset={offset} code={script.code} />
          {script.after.map((line) => (
            <LogLine key={line.at} line={line} frame={frame} offset={offset} />
          ))}
        </div>

        {/* the mini session in the log shows no footer; it appears once
            the zoom lands and this becomes the active screen */}
        <Footer
          frame={frame}
          offset={offset}
          appear={depth === 0 ? appear : ZOOM_END - 6}
          idleAfter={depth >= 1 ? offset + 114 : undefined}
        />

        {/* the spawned session: the whole canvas scaled toward the fixed
            point, sitting in the log flow right below "spawning sub-lm" */}
        {depth < 1 ? (
          <div
            style={{
              position: "absolute",
              inset: 0,
              transform: `scale(${CHILD_SCALE})`,
              transformOrigin: `${P.x}px ${P.y}px`,
              opacity: on(frame, offset + CHILD_AT),
            }}
          >
            <Session
              depth={depth + 1}
              offset={offset + FLOW_SHIFT}
              appear={offset + CHILD_AT}
            />
          </div>
        ) : null}
      </AbsoluteFill>
    </AbsoluteFill>
  );
};

// Beat sheet (16-frame blink cycle), paced like the outro: the bare cursor
// fades in and blinks three times, "fractal" types slowly, two blinks,
// then hand over to the session.
const ColdOpen: React.FC = () => {
  const frame = useCurrentFrame();
  const opacity = frame >= SESSION_APPEAR ? 0 : fadeIn(frame, 0, 24);
  const typing = frame >= 48 && frame < 68;
  return (
    <AbsoluteFill style={{ justifyContent: "center", paddingLeft: 220, opacity }}>
      <div style={{ fontSize: 132, color: C.text, whiteSpace: "pre" }}>
        <span style={{ color: C.accent }}>$ </span>
        {typed("fractal", frame, 48, 0.35)}
        <Cursor opacity={typing ? 1 : blink(frame)} />
      </div>
    </AbsoluteFill>
  );
};

const ZoomStage: React.FC = () => {
  const frame = useCurrentFrame();
  // Dive exactly one nesting level into the freshly spawned, still-empty
  // session; its flow starts once we have arrived.
  const p = interpolate(frame, [ZOOM_START, ZOOM_END], [0, 1], {
    ...clamp,
    easing: Easing.bezier(0.65, 0, 0.35, 1),
  });
  const scale = Math.pow(CHILD_SCALE, -p);
  const opacity =
    on(frame, SESSION_APPEAR) * interpolate(frame, [396, 410], [1, 0], clamp);

  return (
    <AbsoluteFill
      style={{
        opacity,
        transform: `scale(${scale})`,
        transformOrigin: `${P.x}px ${P.y}px`,
      }}
    >
      <Session depth={0} offset={SESSION_SHIFT} appear={SESSION_APPEAR} />
    </AbsoluteFill>
  );
};

// The outro takes its time: logo, wordmark, tagline, and install command
// arrive one after another with long fades, so each gets read on its own.
const OUTRO_AT = 408;

const Outro: React.FC = () => {
  const frame = useCurrentFrame();
  const rise = interpolate(frame, [OUTRO_AT, OUTRO_AT + 70], [26, 0], {
    ...clamp,
    easing: easeOut,
  });
  const install = typed("uv tool install fractal", frame, OUTRO_AT + 120, 0.6);
  const installVisible = frame >= OUTRO_AT + 112;

  return (
    <AbsoluteFill
      style={{
        alignItems: "center",
        justifyContent: "center",
        opacity: fadeIn(frame, OUTRO_AT, 40),
        transform: `translateY(${rise}px)`,
      }}
    >
      <Img
        src={staticFile("logo-mark.png")}
        style={{ width: 290, height: 290, marginBottom: 4 }}
      />
      <div
        style={{
          fontSize: 108,
          fontWeight: 700,
          color: C.text,
          lineHeight: 1,
          letterSpacing: -2,
          opacity: fadeIn(frame, OUTRO_AT + 30, 36),
        }}
      >
        fractal
      </div>
      <div
        style={{
          marginTop: 22,
          fontSize: 30,
          color: C.dim,
          opacity: fadeIn(frame, OUTRO_AT + 62, 36),
        }}
      >
        the recursive language model cli agent
      </div>
      <div
        style={{
          marginTop: 52,
          fontSize: 32,
          color: C.lavender,
          whiteSpace: "pre",
          opacity: fadeIn(frame, OUTRO_AT + 106, 24),
        }}
      >
        <span style={{ color: C.accent }}>$ </span>
        {install}
        <Cursor opacity={installVisible ? blink(frame) : 0} />
      </div>
      <div
        style={{
          position: "absolute",
          bottom: 64,
          display: "flex",
          alignItems: "center",
          gap: 14,
          opacity: fadeIn(frame, OUTRO_AT + 150, 36),
        }}
      >
        <span style={{ fontSize: 22, color: C.faint }}>by</span>
        <Img
          src={staticFile("trampoline.svg")}
          style={{ height: 26, opacity: 0.85 }}
        />
      </div>
    </AbsoluteFill>
  );
};

export const FractalRelease: React.FC = () => {
  return (
    <AbsoluteFill style={{ background: BG, fontFamily: MONO }}>
      {/* one quiet vignette, nothing else */}
      <AbsoluteFill
        style={{
          background:
            "radial-gradient(ellipse at 50% 42%, rgba(124, 58, 237, 0.10), transparent 62%), radial-gradient(ellipse at 50% 110%, rgba(0, 0, 0, 0.45), transparent 60%)",
        }}
      />
      <ZoomStage />
      <ColdOpen />
      <Outro />
    </AbsoluteFill>
  );
};
