#!/bin/sh
# Fractal installer.
#
#   curl -LsSf https://fractal.trampoline.ai/install.sh | sh
#
# What it does:
#   1. Ensures `uv` is available (bootstraps it if missing).
#   2. Installs the Fractal CLI as an isolated uv tool, putting `fractal`
#      on your PATH.
#   3. Checks the runtime prerequisites (Docker + the `sbx` CLI) and tells
#      you what's still needed.
#
# Environment overrides:
#   FRACTAL_VERSION=0.1.0   install a specific version instead of the latest

set -eu

# --- output helpers ---------------------------------------------------------
if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
    BOLD="$(printf '\033[1m')"; DIM="$(printf '\033[2m')"
    RED="$(printf '\033[31m')"; GREEN="$(printf '\033[32m')"
    YELLOW="$(printf '\033[33m')"; RESET="$(printf '\033[0m')"
else
    BOLD=""; DIM=""; RED=""; GREEN=""; YELLOW=""; RESET=""
fi

info()  { printf '%s\n' "${BOLD}fractal:${RESET} $*"; }
warn()  { printf '%s\n' "${YELLOW}fractal: $*${RESET}" >&2; }
err()   { printf '%s\n' "${RED}fractal: $*${RESET}" >&2; }
step()  { printf '%s\n' "${DIM}  -> $*${RESET}"; }

has() { command -v "$1" >/dev/null 2>&1; }

# --- 1. ensure uv -----------------------------------------------------------
ensure_uv() {
    if has uv; then
        step "uv found ($(uv --version 2>/dev/null || echo present))"
        return 0
    fi
    info "uv not found; installing it..."
    if ! has curl; then
        err "curl is required to bootstrap uv. Install curl and re-run."
        exit 1
    fi
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # uv installs to ~/.local/bin (or $XDG_BIN_HOME / $CARGO_HOME/bin).
    for d in "$HOME/.local/bin" "${XDG_BIN_HOME:-}" "${CARGO_HOME:-$HOME/.cargo}/bin"; do
        [ -n "$d" ] && [ -d "$d" ] && case ":$PATH:" in *":$d:"*) ;; *) PATH="$d:$PATH";; esac
    done
    export PATH
    if ! has uv; then
        err "uv was installed but isn't on PATH. Open a new shell and re-run, or see https://docs.astral.sh/uv/."
        exit 1
    fi
}

# --- 2. install fractal -----------------------------------------------------
install_fractal() {
    _pkg="fractal"
    [ -n "${FRACTAL_VERSION:-}" ] && _pkg="fractal==$FRACTAL_VERSION"
    info "installing $_pkg ..."
    # --force so re-running the script upgrades an existing install.
    uv tool install --force "$_pkg"
    uv tool update-shell >/dev/null 2>&1 || true
}

# --- 3. prerequisite checks (warn only; don't fail the install) -------------
check_prereqs() {
    info "checking runtime prerequisites..."
    if has docker; then
        if docker info >/dev/null 2>&1; then
            step "${GREEN}Docker is running${RESET}"
        else
            warn "Docker is installed but not running. Start Docker before your first turn."
        fi
    else
        warn "Docker not found. Fractal runs each turn in a Docker Sandbox; install Docker."
    fi

    if has sbx; then
        step "${GREEN}sbx CLI found${RESET} (run 'sbx login' if you haven't)"
    else
        warn "sbx CLI not found. Install and log in:"
        printf '         brew install docker/tap/sbx && sbx login\n' >&2
    fi
}

# --- main -------------------------------------------------------------------
main() {
    info "installing the Fractal agentic CLI"
    ensure_uv
    if ! install_fractal; then
        err "installation failed."
        exit 1
    fi
    check_prereqs

    printf '\n'
    info "${GREEN}done.${RESET} Verify with: ${BOLD}fractal --help${RESET}"
    info "Get started:  ${BOLD}cd your-project && fractal${RESET}"
    if ! has fractal; then
        warn "'fractal' isn't on your PATH yet — open a new shell, then run 'fractal --help'."
    fi
}

main "$@"
