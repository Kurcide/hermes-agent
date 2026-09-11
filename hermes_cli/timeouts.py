from __future__ import annotations


def _coerce_timeout(raw: object) -> float | None:
    try:
        timeout = float(raw)
    except (TypeError, ValueError):
        return None
    return timeout if timeout > 0 else None


def _configured_timeout(provider_id: str, model: str | None, model_key: str, provider_key: str,
                        requested_provider: str | None = None) -> float | None:
    """Per-model ``providers.<id>.models.<model>.<model_key>`` wins over ``providers.<id>.<provider_key>``."""
    if not provider_id:
        return None
    try:
        from hermes_cli.config import load_config_readonly
        config = load_config_readonly()
    except Exception:
        return None
    providers = config.get("providers", {}) if isinstance(config, dict) else {}
    if not isinstance(providers, dict):
        return None
    # Custom transport canonicalization must not discard the named endpoint's
    # timeout policy. Keep the transport-wide entry as the existing fallback.
    candidates = []
    named_custom = provider_id if provider_id.startswith("custom:") else (
        requested_provider if provider_id == "custom" else None
    )
    if named_custom and named_custom != "custom":
        from hermes_cli.config import is_provider_enabled
        from hermes_cli.providers import custom_provider_aliases

        requested = named_custom.strip().lower().replace(" ", "-")
        for name, entry in providers.items():
            if not isinstance(entry, dict) or not is_provider_enabled(entry):
                continue
            if not (entry.get("api") or entry.get("url") or entry.get("base_url")):
                continue
            aliases = custom_provider_aliases(str(entry.get("name", "") or name), str(name))
            if requested in aliases:
                candidates.append(entry)
                break
    candidates.append(providers.get("custom" if provider_id.startswith("custom:") else provider_id, {}))
    for provider_config in candidates:
        if not isinstance(provider_config, dict):
            continue
        model_config = _get_model_config(provider_config, model)
        timeout = _coerce_timeout(model_config.get(model_key)) if model_config is not None else None
        if timeout is None:
            timeout = _coerce_timeout(provider_config.get(provider_key))
        if timeout is not None:
            return timeout
    return None


def get_provider_request_timeout(provider_id: str, model: str | None = None, *,
                                 requested_provider: str | None = None) -> float | None:
    """Return a configured provider request timeout in seconds, if any."""
    return _configured_timeout(provider_id, model, "timeout_seconds", "request_timeout_seconds", requested_provider)


def get_provider_stale_timeout(provider_id: str, model: str | None = None, *,
                               requested_provider: str | None = None) -> float | None:
    """Return a configured non-stream stale timeout in seconds, if any."""
    return _configured_timeout(provider_id, model, "stale_timeout_seconds", "stale_timeout_seconds", requested_provider)


def _get_model_config(provider_config: dict[str, object], model: str | None) -> dict[str, object] | None:
    if not model:
        return None
    models = provider_config.get("models", {})
    model_config = models.get(model, {}) if isinstance(models, dict) else {}
    return model_config if isinstance(model_config, dict) else None
