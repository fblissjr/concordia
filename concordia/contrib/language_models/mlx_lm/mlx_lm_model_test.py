# Copyright 2025 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the MLX-LM language model provider."""

import unittest

# Check if MLX is available before importing the model
try:
  import mlx.core as mx

  MLX_AVAILABLE = True
except ImportError:
  MLX_AVAILABLE = False


@unittest.skipUnless(MLX_AVAILABLE, 'MLX-LM not available')
class MLXLMLanguageModelTest(unittest.TestCase):
  """Tests for MLXLMLanguageModel."""

  @classmethod
  def setUpClass(cls):
    """Import modules that require MLX."""
    # These imports are deferred to avoid import errors when MLX is not installed
    from unittest import mock

    from concordia.contrib.language_models.mlx_lm import mlx_lm_model

    cls.mlx_lm_model = mlx_lm_model
    cls.mock = mock
    # The patch target must be where the name is looked up, not where it's
    # defined
    cls.load_patch_target = (
        'concordia.contrib.language_models.mlx_lm.mlx_lm_model.load'
    )

  def _create_mock_tokenizer(self):
    """Create a mock tokenizer for testing."""

    class MockTokenizer:
      """Mock tokenizer for testing."""

      def __init__(self):
        self.eos_token_ids = {2}  # Mock EOS token ID

      def encode(self, text: str) -> list[int]:
        # Simple mock: return list of character ordinals
        return [ord(c) for c in text]

      def decode(self, tokens: list[int]) -> str:
        # Simple mock: convert ordinals back to characters
        return ''.join(chr(t) for t in tokens if t < 256)

      def apply_chat_template(
          self, messages, tokenize=False, add_generation_prompt=True
      ) -> str:
        # Simple mock chat template
        parts = []
        for msg in messages:
          role = msg['role']
          content = msg['content']
          parts.append(f'<{role}>{content}</{role}>')
        if add_generation_prompt:
          parts.append('<assistant>')
        return ''.join(parts)

    return MockTokenizer()

  def _create_mock_model(self):
    """Create a mock MLX model for testing."""

    class MockModel:
      """Mock MLX model for testing."""

      def __init__(self):
        self._call_count = 0

      def __call__(self, input_ids):
        # Return mock logits
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1] if len(input_ids.shape) > 1 else 1
        vocab_size = 256

        # Create mock logits that favor token 65 ('A')
        logits = mx.zeros((batch_size, seq_len, vocab_size))
        # Make token 65 most likely
        logits = logits.at[:, :, 65].add(10.0)

        self._call_count += 1
        return logits

    return MockModel()

  def test_initialization(self):
    """Test that the model initializes correctly."""
    mock_model = self._create_mock_model()
    mock_tokenizer = self._create_mock_tokenizer()

    with self.mock.patch(self.load_patch_target) as mock_load:
      mock_load.return_value = (mock_model, mock_tokenizer)

      model = self.mlx_lm_model.MLXLMLanguageModel(
          model_name='test-model',
      )

      mock_load.assert_called_once_with(
          'test-model',
          tokenizer_config=None,
          model_config=None,
          adapter_path=None,
      )
      self.assertEqual(model._model_name, 'test-model')

  def test_initialization_with_adapter(self):
    """Test initialization with LoRA adapter."""
    mock_model = self._create_mock_model()
    mock_tokenizer = self._create_mock_tokenizer()

    with self.mock.patch(self.load_patch_target) as mock_load:
      mock_load.return_value = (mock_model, mock_tokenizer)

      self.mlx_lm_model.MLXLMLanguageModel(
          model_name='test-model',
          adapter_path='/path/to/adapter',
      )

      mock_load.assert_called_once_with(
          'test-model',
          tokenizer_config=None,
          model_config=None,
          adapter_path='/path/to/adapter',
      )

  def test_sample_text_returns_string(self):
    """Test that sample_text returns a string."""
    mock_model = self._create_mock_model()
    mock_tokenizer = self._create_mock_tokenizer()

    with self.mock.patch(self.load_patch_target) as mock_load:
      mock_load.return_value = (mock_model, mock_tokenizer)

      model = self.mlx_lm_model.MLXLMLanguageModel(model_name='test-model')

      result = model.sample_text('Hello', max_tokens=5)

      self.assertIsInstance(result, str)

  def test_sample_text_respects_max_tokens(self):
    """Test that sample_text respects the max_tokens limit."""
    mock_model = self._create_mock_model()
    mock_tokenizer = self._create_mock_tokenizer()

    with self.mock.patch(self.load_patch_target) as mock_load:
      mock_load.return_value = (mock_model, mock_tokenizer)

      model = self.mlx_lm_model.MLXLMLanguageModel(model_name='test-model')

      # Generate with small max_tokens
      result = model.sample_text('Hello', max_tokens=3)

      # Result should be limited (mock generates 'A' tokens)
      self.assertLessEqual(len(result), 10)  # Generous upper bound

  def test_sample_text_handles_terminators(self):
    """Test that sample_text handles terminators correctly."""
    mock_model = self._create_mock_model()
    mock_tokenizer = self._create_mock_tokenizer()

    with self.mock.patch(self.load_patch_target) as mock_load:
      mock_load.return_value = (mock_model, mock_tokenizer)

      model = self.mlx_lm_model.MLXLMLanguageModel(model_name='test-model')

      # The mock generates 'A' tokens, so we test with a terminator
      result = model.sample_text('Hello', max_tokens=10, terminators=['AA'])

      # If 'AA' was generated, it should be truncated
      self.assertNotIn('AA', result)

  def test_sample_choice_returns_valid_tuple(self):
    """Test that sample_choice returns a valid tuple."""
    mock_model = self._create_mock_model()
    mock_tokenizer = self._create_mock_tokenizer()

    with self.mock.patch(self.load_patch_target) as mock_load:
      mock_load.return_value = (mock_model, mock_tokenizer)

      model = self.mlx_lm_model.MLXLMLanguageModel(model_name='test-model')

      responses = ['yes', 'no', 'maybe']
      idx, response, debug = model.sample_choice('Is it raining?', responses)

      self.assertIsInstance(idx, int)
      self.assertIn(idx, range(len(responses)))
      self.assertIn(response, responses)
      self.assertEqual(response, responses[idx])
      self.assertIsInstance(debug, dict)
      self.assertIn('logprobs', debug)
      self.assertIn('method', debug)
      self.assertEqual(debug['method'], 'logprobs')

  def test_sample_choice_empty_responses_raises(self):
    """Test that sample_choice raises ValueError for empty responses."""
    mock_model = self._create_mock_model()
    mock_tokenizer = self._create_mock_tokenizer()

    with self.mock.patch(self.load_patch_target) as mock_load:
      mock_load.return_value = (mock_model, mock_tokenizer)

      model = self.mlx_lm_model.MLXLMLanguageModel(model_name='test-model')

      with self.assertRaises(ValueError) as context:
        model.sample_choice('Is it raining?', [])

      self.assertIn('No responses provided', str(context.exception))

  def test_sample_choice_debug_info_contains_all_responses(self):
    """Test that debug info contains log probs for all responses."""
    mock_model = self._create_mock_model()
    mock_tokenizer = self._create_mock_tokenizer()

    with self.mock.patch(self.load_patch_target) as mock_load:
      mock_load.return_value = (mock_model, mock_tokenizer)

      model = self.mlx_lm_model.MLXLMLanguageModel(model_name='test-model')

      responses = ['red', 'blue', 'green']
      _, _, debug = model.sample_choice('What color is the sky?', responses)

      self.assertIn('logprobs', debug)
      for response in responses:
        self.assertIn(response, debug['logprobs'])
        self.assertIsInstance(debug['logprobs'][response], float)

  def test_format_prompt_with_chat_template(self):
    """Test prompt formatting with chat template."""
    mock_model = self._create_mock_model()
    mock_tokenizer = self._create_mock_tokenizer()

    with self.mock.patch(self.load_patch_target) as mock_load:
      mock_load.return_value = (mock_model, mock_tokenizer)

      model = self.mlx_lm_model.MLXLMLanguageModel(
          model_name='test-model',
          system_message='You are helpful.',
      )

      formatted = model._format_prompt('Hello')

      self.assertIn('You are helpful.', formatted)
      self.assertIn('Hello', formatted)
      self.assertIn('<assistant>', formatted)

  def test_temperature_zero_handling(self):
    """Test that zero temperature is handled correctly."""
    mock_model = self._create_mock_model()
    mock_tokenizer = self._create_mock_tokenizer()

    with self.mock.patch(self.load_patch_target) as mock_load:
      mock_load.return_value = (mock_model, mock_tokenizer)

      model = self.mlx_lm_model.MLXLMLanguageModel(model_name='test-model')

      # Should not raise with temperature=0
      result = model.sample_text('Hello', max_tokens=3, temperature=0.0)
      self.assertIsInstance(result, str)


class MLXLMLanguageModelSkipTest(unittest.TestCase):
  """Test that verifies the skip message when MLX is unavailable."""

  @unittest.skipIf(MLX_AVAILABLE, 'MLX is available, skip unavailable test')
  def test_skip_message_when_mlx_unavailable(self):
    """Verify that tests are skipped when MLX is not installed."""
    # This test only runs when MLX is not available
    # It verifies that the module gracefully handles the missing dependency
    self.assertFalse(MLX_AVAILABLE)


if __name__ == '__main__':
  unittest.main()
