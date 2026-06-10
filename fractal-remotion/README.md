# Fractal release video

Remotion composition for a short landscape release bumper introducing Fractal
as the first RLM-based CLI coding agent.

The closing CTA uses:

```console
uv tool install fractal
```

## Commands

```console
npm i
npm run dev
npm run still
npm run render
```

`npm run render` writes `out/fractal-release.mp4`.
`npm run still` writes a midpoint layout check to
`out/fractal-release-frame.png`.

## Composition

- Composition id: `FractalRelease`
- Duration: 7 seconds
- Format: 1920x1080, 30 fps
- Source: `src/Composition.tsx`

The animation uses Remotion frame-based interpolation and springs. There are no
CSS transitions or keyframe animations, so Studio preview and renders should
match.
