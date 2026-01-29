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

"""Language Model that uses MLX-LM for local inference on Apple Silicon.

MLX-LM is a fast inference library for large language models using Apple's MLX
framework, optimized for Apple Silicon (M1/M2/M3/M4 chips).

Example usage:
  # Using a HuggingFace model (will be downloaded automatically)
  model = mlx_lm_model.MLXLMLanguageModel(
      model_name="mlx-community/Llama-3.2-3B-Instruct-4bit",
  )

  # Using a local model directory
  model = mlx_lm_model.MLXLMLanguageModel(
      model_name="./models/my-local-model",
  )

  # With LoRA adapter
  model = mlx_lm_model.MLXLMLanguageModel(
      model_name="mlx-community/Llama-3.2-3B-Instruct-4bit",
      adapter_path="/path/to/lora/adapter",
  )
"""

from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import Any, override

from concordia.language_model import language_model
from concordia.utils.deprecated import measurements as measurements_lib
import mlx.core as mx
from mlx_lm import generate as mlx_generate
from mlx_lm import load
from mlx_lm.sample_utils import make_sampler


_DEFAULT_SYSTEM_MESSAGE = (
    'You always continue sentences provided by the user and you never repeat '
    'what the user already said.'
)


def _resolve_model_path(model_name: str) -> str:
  """Resolve model name to an absolute path if it's a local directory.

  MLX-LM's load function checks if a path exists locally before attempting
  to download from HuggingFace. However, relative paths with multiple slashes
  (e.g., "models/org/model-name") can trigger HuggingFace's repo ID validation
  before the existence check. This function resolves such paths to absolute
  paths to avoid this issue.

  Args:
    model_name: Either a HuggingFace repo ID (e.g., "mlx-community/model")
      or a local path (e.g., "./models/my-model" or "models/my-model").

  Returns:
    If the path exists locally, returns the resolved absolute path.
    Otherwise, returns the original model_name unchanged (for HuggingFace).
  """
  path = Path(model_name)

  # Check if it looks like a local path and exists
  if path.exists():
    # Return absolute path to avoid HuggingFace validation issues
    return str(path.resolve())

  # Check common indicators of local paths that might not exist yet
  # (useful for providing better error messages)
  if model_name.startswith(('./', '../', '/')):
    # Explicitly looks like a path, resolve it even if it doesn't exist
    # (mlx_lm will give a clearer error about missing files)
    return str(path.resolve())

  # Looks like a HuggingFace repo ID, return as-is
  return model_name


class MLXLMLanguageModel(language_model.LanguageModel):
  """Language Model that uses MLX-LM for local inference."""

  def __init__(
      self,
      model_name: str,
      *,
      adapter_path: str | None = None,
      tokenizer_config: dict[str, Any] | None = None,
      model_config: dict[str, Any] | None = None,
      system_message: str = _DEFAULT_SYSTEM_MESSAGE,
      measurements: measurements_lib.Measurements | None = None,
      channel: str = language_model.DEFAULT_STATS_CHANNEL,
  ):
    """Initialize the MLX-LM language model.

    Args:
      model_name: The name or path of the model to load. Can be a HuggingFace
        repo ID (e.g., "mlx-community/Llama-3.2-3B-Instruct-4bit") or a local
        path to model weights (e.g., "./models/my-model" or an absolute path).
      adapter_path: Optional path to LoRA adapter weights.
      tokenizer_config: Optional configuration for the tokenizer.
      model_config: Optional configuration for the model.
      system_message: System message to prepend to prompts when using chat
        templates.
      measurements: Measurements object for logging statistics.
      channel: Channel name for measurements.
    """
    self._model_name = model_name
    self._adapter_path = adapter_path
    self._system_message = system_message
    self._measurements = measurements
    self._channel = channel

    # Resolve local paths to absolute paths to avoid HuggingFace validation
    resolved_model_name = _resolve_model_path(model_name)

    # Load model and tokenizer via mlx_lm
    self._model, self._tokenizer = load(
        resolved_model_name,
        tokenizer_config=tokenizer_config,
        model_config=model_config,
        adapter_path=adapter_path,
    )

  def _format_prompt(self, prompt: str) -> str:
    """Format the prompt using chat template if available."""
    messages = [
        {'role': 'system', 'content': self._system_message},
        {'role': 'user', 'content': prompt},
    ]

    if hasattr(self._tokenizer, 'apply_chat_template'):
      try:
        formatted = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return formatted
      except Exception:
        # Fall back to simple formatting if chat template fails
        pass

    return f'{self._system_message}\n\n{prompt}'

  def _generate_text(
      self,
      prompt: str,
      max_tokens: int,
      temperature: float,
      top_p: float,
      top_k: int,
      seed: int | None,
  ) -> str:
    """Generate text using the MLX-LM model."""
    # Set random seed if provided
    if seed is not None:
      mx.random.seed(seed)

    # Format prompt
    formatted_prompt = self._format_prompt(prompt)

    # Create sampler with appropriate parameters
    sampler = make_sampler(
        temp=temperature,
        top_p=top_p if top_p < 1.0 else 0.0,
        top_k=top_k if top_k > 0 else 0,
    )

    # Use mlx_lm's generate function which handles caching properly
    result = mlx_generate(
        model=self._model,
        tokenizer=self._tokenizer,
        prompt=formatted_prompt,
        max_tokens=max_tokens,
        sampler=sampler,
        verbose=False,
    )

    return result

  @override
  def sample_text(
      self,
      prompt: str,
      *,
      max_tokens: int = language_model.DEFAULT_MAX_TOKENS,
      terminators: Collection[str] = language_model.DEFAULT_TERMINATORS,
      temperature: float = language_model.DEFAULT_TEMPERATURE,
      top_p: float = language_model.DEFAULT_TOP_P,
      top_k: int = language_model.DEFAULT_TOP_K,
      timeout: float = language_model.DEFAULT_TIMEOUT_SECONDS,
      seed: int | None = None,
  ) -> str:
    """Sample text from the MLX-LM model.

    Args:
      prompt: The input prompt to condition on.
      max_tokens: Maximum number of tokens to generate.
      terminators: Strings that will terminate generation if encountered.
      temperature: Sampling temperature. Higher values increase randomness.
      top_p: Nucleus sampling threshold.
      top_k: Top-k sampling limit.
      timeout: Timeout for the request (not used by MLX-LM).
      seed: Random seed for reproducibility.

    Returns:
      The generated text response.
    """
    del timeout  # MLX-LM doesn't support timeout

    # Ensure temperature is positive
    if temperature <= 0:
      temperature = 0.1

    # Generate text
    result = self._generate_text(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        seed=seed,
    )

    # Handle terminators by truncating at first occurrence
    for term in terminators:
      if term in result:
        result = result.split(term)[0]

    result = result.strip()

    # Log statistics
    if self._measurements is not None:
      self._measurements.publish_datum(
          self._channel,
          {'raw_text_length': len(result)},
      )

    return result

  def _compute_log_probability(self, prompt: str, response: str) -> float:
    """Compute the log probability of generating response given prompt.

    This method tokenizes the combined prompt+response, runs the model forward
    pass, and sums the log probabilities of the response tokens.

    Args:
      prompt: The input prompt.
      response: The response to score.

    Returns:
      The total log probability of the response tokens.
    """
    # Format and tokenize
    formatted_prompt = self._format_prompt(prompt)
    prompt_tokens = self._tokenizer.encode(formatted_prompt)
    full_text = formatted_prompt + response
    full_tokens = self._tokenizer.encode(full_text)

    prompt_length = len(prompt_tokens)

    # If response adds no new tokens, return 0
    if len(full_tokens) <= prompt_length:
      return 0.0

    # Run model forward pass
    input_ids = mx.array(full_tokens)[None]  # Add batch dimension
    logits = self._model(input_ids)

    # Compute log probabilities
    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)

    # Sum log probs for response tokens
    # For token at position i, we use logits at position i-1
    total_log_prob = 0.0
    for i in range(prompt_length, len(full_tokens)):
      token_id = full_tokens[i]
      # Logits at position i-1 predict token at position i
      token_log_prob = log_probs[0, i - 1, token_id].item()
      total_log_prob += token_log_prob

    return total_log_prob

  @override
  def sample_choice(
      self,
      prompt: str,
      responses: Sequence[str],
      *,
      seed: int | None = None,
  ) -> tuple[int, str, Mapping[str, Any]]:
    """Sample a choice from responses using log probability scoring.

    This method computes the log probability of each response given the prompt
    and returns the response with the highest probability.

    Args:
      prompt: The input prompt to condition on.
      responses: The candidate responses to choose from.
      seed: Random seed (not used for log-prob scoring).

    Returns:
      A tuple of (index, response, debug_info) where index is the selected
      response index, response is the selected string, and debug_info contains
      the log probabilities for all responses.

    Raises:
      ValueError: If no responses are provided.
    """
    del seed  # Not used for log-probability based selection

    if not responses:
      raise ValueError('No responses provided to choose from.')

    # Compute log probability for each response
    log_probs = []
    for response in responses:
      log_prob = self._compute_log_probability(prompt, response)
      log_probs.append(log_prob)

    # Select response with highest log probability
    max_idx = log_probs.index(max(log_probs))

    # Create debug info
    debug_info = {
        'logprobs': {
            response: log_probs[i] for i, response in enumerate(responses)
        },
        'method': 'logprobs',
    }

    if self._measurements is not None:
      self._measurements.publish_datum(
          self._channel,
          {'choice_method': 'logprobs', 'num_choices': len(responses)},
      )

    return max_idx, responses[max_idx], debug_info
