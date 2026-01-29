# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Concordia is a library for generative agent-based social simulations. It uses a Game Master (GM) pattern borrowed from tabletop RPGs: agents describe actions in natural language, and the GM resolves their effects in the simulated environment.

This is a fork of `google-deepmind/concordia`. When adding new functionality (like LLM providers), tests should be added for that new functionality.

## Build & Development Commands

This project uses `uv` for dependency management. Do not use `pip` directly.

```bash
# Development installation
uv pip install --editable .[dev]

# Install with LLM provider (google, openai, huggingface, together, vllm)
uv pip install --editable .[google]

# Sync dependencies (preferred)
uv sync

# Run tests (uses pytest-xdist for parallel execution)
uv run pytest concordia              # Library tests only
uv run pytest examples               # Example tests only
uv run pytest                        # All tests

# Run single test file
uv run pytest concordia/path/to/file_test.py

# Linting and type checking
uv run pylint --errors-only concordia
uv run pytype concordia

# Code formatting
uv run pyink .
uv run isort --resolve-all-configs .

# Full validation suite (test + lint + typecheck)
./bin/test.sh
```

## Architecture

### Core Abstractions

- **Entity**: Agents that observe the world and take actions (implemented via `EntityAgent`)
- **Game Master**: Controls simulation flow, resolves actions, generates observations
- **Components**: Modular building blocks composed into entities (Memory, Observation, Planning, Actuation)
- **Prefab**: Reusable recipes for building entities/GMs with specific component configurations
- **Engine**: Orchestrates the simulation loop (`Sequential` for turn-based, `Simultaneous` for parallel)

### Component Lifecycle

Components execute in phases during agent actions:
```
READY -> PRE_ACT -> POST_ACT -> UPDATE
READY -> PRE_OBSERVE -> POST_OBSERVE -> UPDATE
```

Components communicate by looking each other up via `get_entity().get_component(key)`.

### Key Directories

| Directory | Purpose |
|-----------|---------|
| `concordia/agents/` | Agent implementations, primarily `EntityAgent` |
| `concordia/components/agent/` | Agent components (memory, observation, instructions, acting) |
| `concordia/components/game_master/` | GM components (flow control, resolution, observations) |
| `concordia/environment/` | Engine implementations and GM core |
| `concordia/prefabs/` | Pre-assembled agent and GM recipes |
| `concordia/typing/` | Core interfaces: `Entity`, `ActionSpec`, `OutputType`, component protocols |
| `concordia/document/` | LLM prompt management with `Document` and `InteractiveDocument` |
| `concordia/thought_chains/` | Modular reasoning pipelines for event resolution |
| `concordia/language_model/` | LLM API wrappers with retry, limiting, profiling |
| `concordia/contrib/language_models/` | Provider implementations (OpenAI, Google, HuggingFace, etc.) |

### Simulation Flow

1. GM decides next acting entity
2. Agent's `act()` generates action via components
3. Engine calls GM to resolve action (`EventResolution`)
4. GM generates observations for entities (`MakeObservation`)
5. Entities call `observe()` to update memory/state
6. Loop until termination condition

## Code Style

- **Style Guide**: Google Python Style Guide
- **Indentation**: 2 spaces (configured via pyink)
- **Line Length**: 80 characters
- **Test Pattern**: `*_test.py` files colocated with source
- **Imports**: Google-style ordering via isort

## LLM Requirements

Simulations require:
1. A `LanguageModel` instance (any LLM API that supports text sampling)
2. A text embedder for associative memory (fixed-dimensional embeddings for semantic search)

LLM integrations are in `concordia/contrib/language_models/`.

## Key Entry Points

- **Tutorial**: `examples/tutorial.ipynb`
- **Cheat Sheet**: `CHEATSHEET.md` - comprehensive code reference
- **Prefab Examples**: `concordia/prefabs/entity/` and `concordia/prefabs/game_master/`

---

## Adding New LLM Providers

### Provider Implementation Pattern

All providers implement the `LanguageModel` abstract base class at `concordia/language_model/language_model.py`:

```python
class LanguageModel(abc.ABC):
  @abc.abstractmethod
  def sample_text(
      self,
      prompt: str,
      *,
      max_tokens: int = DEFAULT_MAX_TOKENS,
      terminators: Sequence[str] = DEFAULT_TERMINATORS,
      temperature: float = DEFAULT_TEMPERATURE,
      timeout: float = -1.0,
      seed: int | None = None,
  ) -> str:
    """Returns a string completion for the given prompt."""

  @abc.abstractmethod
  def sample_choice(
      self,
      prompt: str,
      responses: Sequence[str],
      *,
      seed: int | None = None,
  ) -> tuple[int, str, dict[str, Any]]:
    """Returns (index, response, debug_info) for the best matching choice."""
```

### Directory Structure for New Provider

```
concordia/contrib/language_models/
  mlx/                          # New provider directory
    __init__.py                 # Empty or minimal exports
    mlx_model.py                # Main implementation
    mlx_model_test.py           # Tests for the provider
```

### Registration

Add to the registry in `concordia/contrib/language_models/__init__.py`:

```python
_REGISTRY = types.MappingProxyType({
    # ... existing providers ...
    'mlx': 'mlx.mlx_model.MLXLanguageModel',
})
```

### Dependencies

Add optional dependency group in `setup.py` (or `pyproject.toml` after migration):

```python
extras_require={
    # ... existing groups ...
    'mlx': [
        'mlx',
        'mlx-lm',
    ],
}
```

Then install with: `uv pip install --editable .[mlx]`

### Example: MLX-LM Provider Reference

The `coderef/mlx-lm` directory contains the mlx-lm library for reference. Key APIs:

```python
from mlx_lm import load, generate, stream_generate

# Load model
model, tokenizer = load("mlx-community/Llama-3.2-3B-Instruct-4bit")

# Generate text
text = generate(model, tokenizer, prompt="Hello", max_tokens=256)

# Stream generation
for response in stream_generate(model, tokenizer, prompt, max_tokens=256):
    print(response.text, end="", flush=True)
```

MLX-LM supports:
- Temperature, top_p, top_k, min_p sampling
- KV cache quantization for long sequences
- Speculative decoding with draft models
- Batch generation

### Implementation Tips

1. **sample_choice strategies**:
   - Prompt model to respond with exact choice string, retry on mismatch
   - Use log probabilities to score each choice (preferred for local models)

2. **Measurements/Telemetry**: Accept optional `measurements` parameter for logging:
   ```python
   if self._measurements is not None:
       self._measurements.publish_datum(self._channel, {'raw_text_length': len(response)})
   ```

3. **Error handling**: Raise `language_model.InvalidResponseError` after max retries

---

## Migration Notes: setup.py to pyproject.toml

The project uses `setup.py` for package configuration. To migrate to modern `pyproject.toml`:

### Current State
- `setup.py`: Contains all package metadata and dependencies
- `pyproject.toml`: Contains only tool configurations (pyink, isort, pytest, pytype)

### Migration Path

Move package metadata from `setup.py` to `pyproject.toml`:

```toml
[project]
name = "gdm-concordia"
version = "2.2.0"
description = "A library for building a generative model of social interactions."
readme = "README.md"
license = {text = "Apache 2.0"}
requires-python = ">=3.12"
authors = [
    {name = "DeepMind", email = "noreply@google.com"}
]
keywords = ["multi-agent", "agent-based-simulation", "generative-agents", "python", "machine-learning"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "Intended Audience :: Education",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: Apache Software License",
    "Operating System :: POSIX :: Linux",
    "Operating System :: MacOS :: MacOS X",
    "Programming Language :: Python :: 3 :: Only",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Programming Language :: Python :: 3.14",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]

dependencies = [
    "absl-py",
    "ipython",
    "matplotlib",
    "numpy>=1.26",
    "pandas",
    "python-dateutil",
    "reactivex",
    "tenacity",
    "termcolor",
]

[project.optional-dependencies]
dev = [
    "build",
    "isort",
    "jupyter",
    "pip-tools",
    "pyink",
    "pylint",
    "pytest-xdist",
    "pytype",
    "twine",
]
google = ["google-cloud-aiplatform", "google-generativeai"]
huggingface = ["accelerate", "torch", "transformers"]
openai = ["openai>=1.3.0"]
together = ["together"]
vllm = ["vllm"]
mlx = ["mlx", "mlx-lm"]

[project.urls]
Homepage = "https://github.com/google-deepmind/concordia"
Download = "https://github.com/google-deepmind/concordia/releases"

[build-system]
requires = ["setuptools>=42"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["concordia", "concordia.*"]
```

After migration, `setup.py` can be reduced to a minimal shim or removed entirely.

With `pyproject.toml`, use `uv sync` to install dependencies and `uv run` to execute scripts.
