# Fractal release video

Remotion composition for a short landscape release bumper introducing Fractal
as the first RLM-based CLI coding agent.

The closing CTA uses:

```console
curl -LsSf https://fractal.trampoline.ai/install.sh | sh
```

## Commands

```console
npm i
npm run dev
npm run still
npm run still:mobile
npm run render
npm run render:mobile
```

`npm run render` writes `out/fractal-release.mp4`.
`npm run render:mobile` writes `out/fractal-release-mobile-4x5.mp4`.
`npm run still` writes a midpoint layout check to
`out/fractal-release-frame.png`.
`npm run still:mobile` writes a mobile outro layout check to
`out/fractal-release-mobile-4x5-frame.png`.

## Composition

- Composition id: `FractalRelease`
- Composition id: `FractalReleaseMobile`
- Duration: 7 seconds
- Formats: 1920x1080 and 1080x1350, 30 fps
- Source: `src/Composition.tsx`

The animation uses Remotion frame-based interpolation and springs. There are no
CSS transitions or keyframe animations, so Studio preview and renders should
match.
