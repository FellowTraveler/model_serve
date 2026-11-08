"""Tests for generate_config.py listen address."""

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


if __name__ == '__main__':
    unittest.main()
