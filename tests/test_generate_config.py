"""Tests for generate_config.py top-level llama-swap settings."""

import os
import unittest
from unittest import mock

from generate_config import generate_config, get_config


class TestListenAddress(unittest.TestCase):
    """The generated llama-swap config must listen on BIND_HOST."""

    def listen_for_env(self, env):
        with mock.patch.dict(os.environ, env, clear=True):
            config = get_config()
        return generate_config([], config, {})['listen']

    def test_defaults_to_loopback(self):
        self.assertEqual(self.listen_for_env({}), '127.0.0.1:5847')

    def test_uses_bind_host_and_port_from_env(self):
        listen = self.listen_for_env({'BIND_HOST': '0.0.0.0', 'LLAMA_SWAP_PORT': '6001'})
        self.assertEqual(listen, '0.0.0.0:6001')


class TestCaptureBuffer(unittest.TestCase):
    """llama-swap request/response captures are off unless enabled via env."""

    def capture_buffer_for_env(self, env):
        with mock.patch.dict(os.environ, env, clear=True):
            config = get_config()
        return generate_config([], config, {})['captureBuffer']

    def test_captures_disabled_by_default(self):
        self.assertEqual(self.capture_buffer_for_env({}), 0)

    def test_capture_buffer_from_env(self):
        self.assertEqual(self.capture_buffer_for_env({'LLAMA_SWAP_CAPTURE_BUFFER_MB': '5'}), 5)


if __name__ == '__main__':
    unittest.main()


class TestSkipEntry(unittest.TestCase):
    """`skip: true` keeps a discovered file out of the config entirely."""

    def test_skipped_file_is_not_served_and_others_are(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            config = get_config()
        models = [
            ('gemma4-25.8b-q4_k_m', '/m/gemma4-25.8B-Q4_K_M.gguf', '/blobs/a'),
            ('qwen3-4.0b-q8_0', '/m/qwen3-4.0B-Q8_0.gguf', '/blobs/b'),
        ]
        with mock.patch('generate_config.estimate_ctx_size', return_value=8192):
            generated = generate_config(models, config, {'gemma4-25.8b-q4_k_m': {'skip': True}})
        prefix = config['model_prefix']
        self.assertNotIn(f'{prefix}gemma4-25.8b-q4_k_m', generated['models'])
        self.assertIn(f'{prefix}qwen3-4.0b-q8_0', generated['models'])
