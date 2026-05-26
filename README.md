Fractal
=======

Fractal is an interactive coding-agent CLI built around a Recursive Language
Model. Each user turn is one RLM call over a direct `Workspace` input mounted
into a Docker Sandbox through predict-rlm's SBX backend, so Python subprocesses
and project commands operate on the real workspace path.

Development
-----------

This project depends on a local editable checkout of predict-rlm:

```bash
uv run fractal --help
uv run pytest
```

The RLM-facing imports require `predict_rlm.WorkspaceMode` and direct workspace
support from the local `/Users/emile/git/predict-rlm` checkout.
