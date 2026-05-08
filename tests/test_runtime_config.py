# -*- coding: utf-8 -*-

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src import runtime_config
from src.processor import LiveVideoProcessor, ProcessingConfig
from src.slicing_ui import build_slicing_processing_config


class RuntimeConfigTests(unittest.TestCase):
    def setUp(self):
        runtime_config.load_local_runtime_config.cache_clear()

    def tearDown(self):
        runtime_config.load_local_runtime_config.cache_clear()

    def test_env_overrides_local_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "local_config.json"
            path.write_text(json.dumps({"doubao": {"appid": "local-app", "access_token": "local-token"}}), encoding="utf-8")
            with patch.dict(
                os.environ,
                {
                    "VIDEO_CLIP_LOCAL_CONFIG": str(path),
                    "DOUBAO_APPID": "env-app",
                    "DOUBAO_ACCESS_TOKEN": "env-token",
                },
                clear=False,
            ):
                runtime_config.load_local_runtime_config.cache_clear()
                self.assertEqual(runtime_config.get_doubao_credentials(), ("env-app", "env-token"))

    def test_processor_config_overrides_runtime_credentials(self):
        processor = LiveVideoProcessor.__new__(LiveVideoProcessor)
        processor.config = ProcessingConfig(doubao_appid="ui-app", doubao_access_token="ui-token")
        with patch("src.processor.get_doubao_credentials", return_value=("runtime-app", "runtime-token")):
            with patch("src.processor.DoubaoASR") as doubao_cls:
                doubao_cls.return_value.recognize.return_value = {"code": 0, "utterances": []}
                result = processor._run_doubao_asr(audio=[0.0, 0.0], sr=16000)
                self.assertNotIn("error", result)
                doubao_cls.assert_called_once_with(appid="ui-app", access_token="ui-token")

    def test_slicing_config_carries_ui_doubao_credentials(self):
        config = build_slicing_processing_config(
            {"min_dur": 8.0, "max_dur": 15.0, "enable_subtitle": True},
            output_dir="output",
            songformer_device="cuda",
            doubao_appid="ui-app",
            doubao_access_token="ui-token",
        )

        self.assertTrue(config.enable_subtitle)
        self.assertEqual(config.doubao_appid, "ui-app")
        self.assertEqual(config.doubao_access_token, "ui-token")


if __name__ == "__main__":
    unittest.main()
