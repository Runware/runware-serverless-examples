"""The request shape is the contract, and it is the part a test can reach."""

from __future__ import annotations

from model import DEFAULT_TEMPERATURE, DEFAULT_TOKENS, ChatEndpoint, Message, Reply, Request


def test_the_file_imports_without_the_model_s_own_dependencies() -> None:
    """The build imports this file with none of the app's requirements installed.

    Importing the test module is the same position, since torch and transformers
    are absent here too. The assertion names what that proves.
    """
    assert ChatEndpoint.predict.__name__ == "predict"


def test_requirements_are_declared_and_pinned() -> None:
    assert all("==" in line for line in ChatEndpoint.requirements)


def test_a_role_outside_the_three_is_rejected() -> None:
    """The roles are published as an enum, so the boundary refuses a fourth."""
    roles = Request.model_json_schema()["$defs"]["Message"]["properties"]["role"]["enum"]

    assert roles == ["system", "user", "assistant"]


def test_the_conversation_is_a_list_of_described_messages() -> None:
    messages = Request.model_json_schema()["properties"]["messages"]

    assert messages["type"] == "array"
    assert messages["items"]["$ref"] == "#/$defs/Message"
    assert messages["description"]


def test_the_sampling_fields_publish_their_defaults_and_bounds() -> None:
    props = Request.model_json_schema()["properties"]

    assert (props["max_tokens"]["default"], props["max_tokens"]["maximum"]) == (DEFAULT_TOKENS, 4096)
    assert (props["temperature"]["default"], props["temperature"]["maximum"]) == (DEFAULT_TEMPERATURE, 2)


def test_a_conversation_validates_into_messages() -> None:
    request = Request(messages=[{"role": "user", "content": "hi"}])

    assert isinstance(request.messages[0], Message)
    assert request.max_tokens == DEFAULT_TOKENS


def test_the_reply_reports_how_much_it_generated() -> None:
    assert set(Reply.model_json_schema()["required"]) == {"content", "tokens"}
