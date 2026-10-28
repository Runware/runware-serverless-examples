"""Presence, and the absence of the value, are the two things worth pinning."""

from __future__ import annotations

import os

from model import EXPECTED, ConfigAndSecrets, Report


def _loaded() -> ConfigAndSecrets:
    model = ConfigAndSecrets()
    model.load()

    return model


def test_a_name_that_is_set_is_reported_present(monkeypatch) -> None:
    monkeypatch.setenv("LOG_LEVEL", "info")
    report = _loaded().predict()

    assert next(b for b in report.bindings if b.name == "LOG_LEVEL").present


def test_a_name_that_is_absent_is_listed_as_missing(monkeypatch) -> None:
    monkeypatch.delenv("API_TOKEN", raising=False)
    report = _loaded().predict()

    assert "API_TOKEN" in report.missing


def test_every_expected_name_is_accounted_for() -> None:
    report = _loaded().predict()

    assert [b.name for b in report.bindings] == list(EXPECTED)


def test_no_value_can_reach_the_response(monkeypatch) -> None:
    """The one assertion this example exists for.

    A handler that returned the value would publish a credential to every caller
    and into the task record, so the response carries presence and nothing else.
    """
    monkeypatch.setenv("API_TOKEN", "a-value-that-must-not-travel")
    body = _loaded().predict().model_dump_json()

    assert "a-value-that-must-not-travel" not in body


def test_the_published_response_declares_no_value_field() -> None:
    fields = Report.model_json_schema()["$defs"]["Binding"]["properties"]

    assert set(fields) == {"name", "present"}


def test_the_configuration_is_read_once(monkeypatch) -> None:
    """Changing a variable rolls the app, so a request never needs to re-read."""
    monkeypatch.delenv("CACHE_DIR", raising=False)
    model = _loaded()
    monkeypatch.setenv("CACHE_DIR", "/weights")

    assert "CACHE_DIR" in model.predict().missing
