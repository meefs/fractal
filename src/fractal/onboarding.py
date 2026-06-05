from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, TextIO


CUSTOM_MODEL_SENTINEL = "__custom_model__"


class SetupInputError(ValueError):
    """Raised when interactive setup cannot collect a required answer."""


@dataclass(frozen=True)
class MenuChoice:
    value: str
    label: str
    detail: str | None = None


@dataclass
class InlineMenuState:
    choices: Sequence[MenuChoice]
    active_index: int
    selected_index: int

    @classmethod
    def create(
        cls,
        *,
        choices: Sequence[MenuChoice],
        default: str,
    ) -> "InlineMenuState":
        if not choices:
            raise SetupInputError("setup menu has no choices")
        default_index = next(
            (index for index, choice in enumerate(choices) if choice.value == default),
            0,
        )
        return cls(
            choices=choices,
            active_index=default_index,
            selected_index=default_index,
        )

    def move_up(self) -> None:
        self.active_index = (self.active_index - 1) % len(self.choices)

    def move_down(self) -> None:
        self.active_index = (self.active_index + 1) % len(self.choices)

    def select_active(self) -> None:
        self.selected_index = self.active_index

    def confirmed_value(self) -> str:
        return self.choices[self.selected_index].value


def prompt_for_config(*, stdin: TextIO, stdout: TextIO) -> Any:
    if _should_use_inline_menu(stdin=stdin, stdout=stdout):
        return _prompt_for_config_interactive(stdout=stdout)
    return _prompt_for_config_line(stdin=stdin, stdout=stdout)


def _prompt_for_config_interactive(*, stdout: TextIO) -> Any:
    from .config import FractalConfig
    from .providers import get_provider, list_providers

    providers = list_providers()
    print("Fractal global config setup", file=stdout)
    provider_id = _choose_from_menu(
        title="Fractal setup",
        text="Choose a provider. Use the arrow keys to move and Enter to select.",
        choices=[
            MenuChoice(
                value=provider.id,
                label=provider.display_name,
                detail=f"{provider.id} - default model {provider.default_model}",
            )
            for provider in providers
        ],
        default=providers[0].id,
    )
    provider = get_provider(provider_id)
    model = _choose_model(provider=provider, stdout=stdout)
    provider_config = _prompt_provider_settings_interactive(
        provider_id=provider.id,
        stdout=stdout,
    )
    return FractalConfig(
        active_provider=provider.id,
        active_model=model,
        providers={provider.id: provider_config},
    )


def _prompt_for_config_line(*, stdin: TextIO, stdout: TextIO) -> Any:
    from .config import FractalConfig
    from .providers import list_providers

    providers = list_providers()
    print("Fractal global config setup", file=stdout)
    print("Choose a provider:", file=stdout)
    for index, provider in enumerate(providers, start=1):
        model_options = ", ".join(provider.model_options)
        print(
            f"{index}. {provider.display_name} ({provider.id})",
            file=stdout,
        )
        print(f"   default model: {provider.default_model}", file=stdout)
        if model_options:
            print(f"   model options: {model_options}", file=stdout)

    provider = _prompt_provider_line(stdin=stdin, stdout=stdout, providers=providers)
    model = _prompt_model_line(stdin=stdin, stdout=stdout, provider=provider)
    provider_config = _prompt_provider_settings_line(
        stdin=stdin,
        stdout=stdout,
        provider_id=provider.id,
    )
    return FractalConfig(
        active_provider=provider.id,
        active_model=model,
        providers={provider.id: provider_config},
    )


def _prompt_provider_line(
    *,
    stdin: TextIO,
    stdout: TextIO,
    providers: list[Any],
) -> Any:
    from .providers import get_provider

    provider_by_index = {
        str(index): provider for index, provider in enumerate(providers, start=1)
    }
    provider_ids = {provider.id for provider in providers}
    while True:
        answer = _prompt(
            stdin=stdin,
            stdout=stdout,
            label="Provider number or id",
            default=providers[0].id,
        )
        if answer in provider_by_index:
            return provider_by_index[answer]
        if answer in provider_ids:
            return get_provider(answer)
        print(
            f"Unknown provider {answer!r}. Choose one of the listed providers.",
            file=stdout,
        )


def _prompt_model_line(*, stdin: TextIO, stdout: TextIO, provider: Any) -> str:
    from .providers import CUSTOM_OPENAI_COMPATIBLE, model_choices

    choices = model_choices(provider)
    print(f"Choose a model for {provider.display_name}:", file=stdout)
    for index, model in enumerate(choices, start=1):
        print(f"{index}. {model}", file=stdout)

    model_by_index = {
        str(index): model for index, model in enumerate(choices, start=1)
    }
    while True:
        answer = _prompt(
            stdin=stdin,
            stdout=stdout,
            label="Model number or id",
            default=choices[0],
        )
        if answer in model_by_index:
            return model_by_index[answer]
        if answer in choices:
            return answer
        if provider.id == CUSTOM_OPENAI_COMPATIBLE:
            return answer
        print(
            f"Unknown model {answer!r}. Choose one of the listed models.",
            file=stdout,
        )


def _prompt_provider_settings_line(
    *,
    stdin: TextIO,
    stdout: TextIO,
    provider_id: str,
) -> Any:
    from .config import ProviderConfig
    from .providers import get_provider

    provider = get_provider(provider_id)
    for message in provider.setup_messages:
        print(message, file=stdout)

    base_url = None
    if provider.base_url_label is not None:
        base_url = _prompt_required(
            stdin=stdin,
            stdout=stdout,
            label=provider.base_url_label,
        )

    api_key_env = None
    if provider.auth_type == "api_key_env":
        if provider.default_api_key_env is None:
            raise SetupInputError(
                f"provider {provider.id!r} requires a default API key env var"
            )
        api_key_env = _prompt(
            stdin=stdin,
            stdout=stdout,
            label="API key environment variable",
            default=provider.default_api_key_env,
        )

    return ProviderConfig(
        auth_source=provider.auth_source,
        api_key_env=api_key_env,
        base_url=base_url,
    )


def _prompt_provider_settings_interactive(
    *,
    provider_id: str,
    stdout: TextIO,
) -> Any:
    from .config import ProviderConfig
    from .providers import get_provider

    provider = get_provider(provider_id)
    for message in provider.setup_messages:
        _show_message(title=provider.display_name, text=message, stdout=stdout)

    base_url = None
    if provider.base_url_label is not None:
        base_url = _prompt_text_interactive(
            title=provider.display_name,
            label=provider.base_url_label,
            stdout=stdout,
            required=True,
        )

    api_key_env = None
    if provider.auth_type == "api_key_env":
        if provider.default_api_key_env is None:
            raise SetupInputError(
                f"provider {provider.id!r} requires a default API key env var"
            )
        api_key_env = _prompt_text_interactive(
            title=provider.display_name,
            label="API key environment variable",
            stdout=stdout,
            default=provider.default_api_key_env,
            required=True,
        )

    return ProviderConfig(
        auth_source=provider.auth_source,
        api_key_env=api_key_env,
        base_url=base_url,
    )


def _prompt(
    *,
    stdin: TextIO,
    stdout: TextIO,
    label: str,
    default: str | None = None,
) -> str:
    suffix = f" [{default}]" if default else ""
    print(f"{label}{suffix}: ", end="", flush=True, file=stdout)
    answer = stdin.readline()
    if answer == "":
        raise SetupInputError("setup requires interactive input")
    value = answer.strip()
    if value:
        return value
    if default:
        return default
    return ""


def _prompt_required(*, stdin: TextIO, stdout: TextIO, label: str) -> str:
    value = _prompt(stdin=stdin, stdout=stdout, label=label)
    if not value:
        raise SetupInputError(f"{label} is required")
    return value


def _choose_model(*, provider: Any, stdout: TextIO) -> str:
    from .providers import CUSTOM_OPENAI_COMPATIBLE, model_choices

    choices = model_choices(provider)
    menu_choices = [MenuChoice(value=model, label=model) for model in choices]
    if provider.id == CUSTOM_OPENAI_COMPATIBLE:
        menu_choices.append(
            MenuChoice(
                value=CUSTOM_MODEL_SENTINEL,
                label="Custom model...",
                detail="Enter a model id for this endpoint",
            )
        )
    selected = _choose_from_menu(
        title=f"{provider.display_name} model",
        text="Choose a model. Use the arrow keys to move and Enter to select.",
        choices=menu_choices,
        default=choices[0],
    )
    if selected == CUSTOM_MODEL_SENTINEL:
        return _prompt_text_interactive(
            title=provider.display_name,
            label="Model id",
            stdout=stdout,
            required=True,
        )
    return selected


def _choose_from_menu(
    *,
    title: str,
    text: str,
    choices: Sequence[MenuChoice],
    default: str,
) -> str:
    from prompt_toolkit.application import Application
    from prompt_toolkit.layout import Layout, Window
    from prompt_toolkit.layout.controls import FormattedTextControl

    state = InlineMenuState.create(choices=choices, default=default)
    control = FormattedTextControl(
        lambda: _menu_fragments(title=title, text=text, state=state),
        focusable=True,
    )
    app = Application(
        layout=Layout(
            Window(
                content=control,
                height=len(choices) + 5,
                dont_extend_height=True,
                always_hide_cursor=True,
            )
        ),
        key_bindings=_inline_menu_key_bindings(state),
        style=_interactive_style(),
        full_screen=False,
        mouse_support=False,
        erase_when_done=False,
    )
    try:
        result = app.run()
    except (KeyboardInterrupt, EOFError) as exc:
        raise SetupInputError("setup canceled") from exc
    if result is None:
        raise SetupInputError("setup canceled")
    return result


def _prompt_text_interactive(
    *,
    title: str,
    label: str,
    stdout: TextIO,
    default: str | None = None,
    required: bool = False,
) -> str:
    from prompt_toolkit.shortcuts import prompt

    print(f"\n{title}", file=stdout)
    try:
        result = prompt(
            [("class:prompt", f"{label}: ")],
            default=default or "",
            style=_interactive_style(),
        )
    except (KeyboardInterrupt, EOFError) as exc:
        raise SetupInputError("setup canceled") from exc
    value = result.strip()
    if value:
        return value
    if default:
        return default
    if required:
        raise SetupInputError(f"{label} is required")
    return ""


def _show_message(*, title: str, text: str, stdout: TextIO) -> None:
    print(f"\n{title}", file=stdout)
    print(text, file=stdout)


def _interactive_style() -> Any:
    from prompt_toolkit.styles import Style

    return Style.from_dict({
        "title": "bold #c4b5fd",
        "hint": "#94a3b8",
        "active": "reverse bold #ffffff",
        "selected": "bold #ffffff",
        "marker": "#64748b",
        "selected-marker": "bold #a78bfa",
        "muted": "#94a3b8",
        "prompt": "bold #c4b5fd",
    })


def _should_use_inline_menu(*, stdin: TextIO, stdout: TextIO) -> bool:
    return _is_tty(stdin) and _is_tty(stdout)


def _is_tty(stream: TextIO) -> bool:
    try:
        return stream.isatty()
    except AttributeError:
        return False


def _choice_fragments(choice: MenuChoice) -> list[tuple[str, str]]:
    fragments = [("bold", choice.label)]
    if choice.detail:
        fragments.extend([("", " "), ("class:muted", choice.detail)])
    return fragments


def _menu_fragments(
    *,
    title: str,
    text: str,
    state: InlineMenuState,
) -> list[tuple[str, str]]:
    fragments: list[tuple[str, str]] = [
        ("class:title", title),
        ("", "\n"),
        ("class:hint", text),
        ("", "\n"),
        ("class:hint", "Up/Down move | Space select | Enter confirm | Esc cancel"),
        ("", "\n\n"),
    ]
    for index, choice in enumerate(state.choices):
        active = index == state.active_index
        selected = index == state.selected_index
        row_style = "class:active" if active else ""
        marker_style = "class:selected-marker" if selected else "class:marker"
        prefix = "> " if active else "  "
        marker = "[x]" if selected else "[ ]"
        fragments.extend([
            (row_style, prefix),
            (marker_style, marker),
            (row_style, " "),
        ])
        for style, value in _choice_fragments(choice):
            fragments.append((row_style or style, value))
        fragments.append(("", "\n"))
    return fragments


def _inline_menu_key_bindings(state: InlineMenuState) -> Any:
    from prompt_toolkit.key_binding import KeyBindings

    bindings = KeyBindings()

    @bindings.add("up")
    def _(event: Any) -> None:
        state.move_up()
        event.app.invalidate()

    @bindings.add("down")
    def _(event: Any) -> None:
        state.move_down()
        event.app.invalidate()

    @bindings.add(" ")
    def _(event: Any) -> None:
        state.select_active()
        event.app.invalidate()

    @bindings.add("enter")
    def _(event: Any) -> None:
        event.app.exit(result=state.confirmed_value())

    @bindings.add("escape")
    @bindings.add("c-c")
    def _(event: Any) -> None:
        event.app.exit(result=None)

    return bindings
