from __future__ import annotations

import argparse
import sys
import types
import unittest
from unittest.mock import patch
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from api_client_qwen_v2 import call_model_api
from common import ApiConfig, add_api_cli_args, call_chat_completion


class _FakeResponses:
    def __init__(self) -> None:
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return types.SimpleNamespace(
            output_text="OK",
            usage=types.SimpleNamespace(input_tokens=3, output_tokens=1, total_tokens=4),
        )


class _FakeClient:
    def __init__(self) -> None:
        self.responses = _FakeResponses()

    def with_options(self, **_kwargs):
        return self


class ReasoningConfigTests(unittest.TestCase):
    def test_pipeline_defaults_to_medium(self):
        parser = argparse.ArgumentParser()
        add_api_cli_args(parser)
        args = parser.parse_args([])

        self.assertEqual(args.reasoning_effort, "medium")
        self.assertEqual(ApiConfig().reasoning_effort, "medium")

    def test_responses_request_enables_thinking_at_medium_effort(self):
        client = _FakeClient()
        text, usage = call_model_api(
            client,
            model_name="test-model",
            payload_input=[{"role": "user", "content": "test"}],
            reasoning_effort="medium",
        )

        self.assertEqual(text, "OK")
        self.assertEqual(usage["total_tokens"], 4)
        self.assertEqual(client.responses.kwargs["reasoning"], {"effort": "medium"})
        self.assertEqual(
            client.responses.kwargs["extra_body"],
            {"thinking": {"type": "enabled"}},
        )

    def test_invalid_reasoning_effort_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Invalid Ark reasoning effort"):
            call_model_api(
                _FakeClient(),
                model_name="test-model",
                payload_input=[{"role": "user", "content": "test"}],
                reasoning_effort="disabled",
            )

    def test_token_limit_error_is_not_retried(self):
        cfg = ApiConfig()
        cfg.api_call_retries = 3
        with patch(
            "common.call_model_api",
            side_effect=RuntimeError(
                "Total tokens of multi-modal content and text exceed max message tokens."
            ),
        ) as model_call:
            with self.assertRaisesRegex(RuntimeError, "Non-retryable model input error"):
                call_chat_completion(
                    object(), cfg, [{"role": "user", "content": "x"}], max_tokens=100
                )
        self.assertEqual(model_call.call_count, 1)


if __name__ == "__main__":
    unittest.main()
