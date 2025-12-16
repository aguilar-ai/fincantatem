```
▗▄▄▄▖    ▗▄▄▄▖▄▄▄▄  ▗▞▀▘▗▞▀▜▌▄▄▄▄     ■  ▗▞▀▜▌   ■  ▗▞▀▚▖▄▄▄▄
▐▌         █  █   █ ▝▚▄▖▝▚▄▟▌█   █ ▗▄▟▙▄▖▝▚▄▟▌▗▄▟▙▄▖▐▛▀▀▘█ █ █
▐▛▀▀▘      █  █   █          █   █   ▐▌         ▐▌  ▝▚▄▄▖█   █
▐▌       ▗▄█▄▖                       ▐▌         ▐▌
                                     ▐▌         ▐▌
```

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://github.com/aguilar-ai/fincantatem/blob/main/pyproject.toml)
[![Last commit](https://img.shields.io/github/last-commit/aguilar-ai/fincantatem)](https://github.com/aguilar-ai/fincantatem/commits/main)
[![Repo size](https://img.shields.io/github/repo-size/aguilar-ai/fincantatem)](https://github.com/aguilar-ai/fincantatem)
[![PyPI version](https://img.shields.io/pypi/v/fincantatem.svg)](https://pypi.org/project/fincantatem/)
[![Python versions](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Python is honest, but not generous.

When code breaks, it hands you an exception type and a message. You see where it failed, but the reasoning that led there is scattered across stack frames, source files, and the half-forgotten state sitting in locals. The error itself is real. The understanding is not.

F. Incantatem reconstructs that understanding. It reads the exception in full context, including the stack trace, the relevant source, the local state at the point of rupture, and produces an explanation of why it happened, what caused it, and what to do about it. The difference between knowing a crash occurred and knowing why it occurred is the difference between debugging and guessing.

It integrates without ceremony: decorator, CLI, or IPython extension. Add it where you want. Leave the rest alone.

## Features

![F. Incantatem](images/header.png)
_The tool displays a condensed traceback, identifies the root cause, and provides actionable fixes — all automatically. You can even chat with your exception._

- **Contextual Analysis** — Examines stack traces, source code, and local variables to produce reasoned explanations
- **Multiple Integration Points** — Use as a decorator, command-line tool, or IPython extension
- **Inference Flexibility** — OpenRouter, OpenAI, or any OpenAI-compatible API; support for local inference via Ollama or vLLM
- **Rich Output** — Optional Markdown formatting and interactive chat for follow-up questions
- **Zero Core Dependencies** — The library itself is lightweight; optional features remain modular
- **Security Conscious** — Optional cautious mode to automatically redact secrets and PII before transmission

## Table of Contents

- [Features](#features)
- [Table of Contents](#table-of-contents)
- [Installation](#installation)
  - [Configuration](#configuration)
    - [Environment Variables](#environment-variables)
    - [Presets](#presets)
  - [Optional Features](#optional-features)
- [Usage](#usage)
  - [As a Decorator](#as-a-decorator)
    - [Decorator Options](#decorator-options)
  - [From the Command Line](#from-the-command-line)
    - [Options](#options)
  - [In Jupyter/IPython](#in-jupyteripython)
- [Examples](#examples)
  - [1. The "Silent API Change"](#1-the-silent-api-change)
  - [2. The "Mutable Default Argument" Trap](#2-the-mutable-default-argument-trap)
  - [3. The "Unicode Normalization Bomb"](#3-the-unicode-normalization-bomb)
- [Data Risks](#data-risks)
  - [Cautious Mode](#cautious-mode)
    - [How It Works](#how-it-works)
    - [Important Caveats](#important-caveats)
  - [Alternative Mitigation Strategies](#alternative-mitigation-strategies)
- [Performance Considerations](#performance-considerations)
- [Key Features](#key-features)
  - [Basic Debugging](#basic-debugging)
  - [Interactive Chat](#interactive-chat)
  - [Cautious Mode](#cautious-mode-1)
- [Requirements](#requirements)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [Status](#status)

## Installation

The library installs with zero dependencies:

```bash
pip install fincantatem
```

Or with uv:

```bash
uv add fincantatem
```

### Configuration

When using the library as a decorator or as the IPython extension, you can configure the inference settings using environment variables. You can also provide them as arguments when using the command-line interface.

#### Environment Variables

| Variable     | Default                                         | Description                         |
| ------------ | ----------------------------------------------- | ----------------------------------- |
| `FI_PRESET`  | `openrouter`                                    | Preset name or custom identifier    |
| `FI_API_KEY` | None                                            | API key for the inference service   |
| `FI_URL`     | `https://openrouter.ai/api/v1/chat/completions` | OpenAI-compatible API endpoint      |
| `FI_MODEL`   | `google/gemini-2.5-flash`                       | Model identifier (overrides preset) |

#### Presets

**OpenRouter** (default)

```bash
FI_API_KEY=sk_your_key python -m fincantatem script.py
```

**OpenAI**

```bash
FI_PRESET=openai FI_API_KEY=sk_your_key python -m fincantatem script.py
```

**Local Inference** (e.g., Ollama)

```bash
FI_PRESET=local \
  FI_URL=http://localhost:11434/v1/chat/completions \
  FI_MODEL=llama2 \
  python -m fincantatem script.py
```

### Optional Features

For rich Markdown output in your terminal:

```bash
pip install "fincantatem[pretty]"
```

For automatic redaction of secrets and PII:

```bash
pip install "fincantatem[cautious]"
python -m spacy download en_core_web_lg
```

## Usage

### As a Decorator

Wrap any function to get AI-powered error analysis when exceptions occur:

```python
from fincantatem import finite

@finite
def process_data(value: int):
    return 100 / value

process_data(0)  # Exception triggers analysis
```

#### Decorator Options

```python
@finite(
    preset="openrouter",  # "openrouter" or "openai" (default: "openrouter")
    snippets=True,        # Send code snippets or full source (default: True)
    chat=False,           # Enable interactive chat after analysis (default: False)
    cautious=False,       # Redact secrets and PII (default: False)
)
def my_function():
    pass
```

### From the Command Line

Run any Python script with automatic error analysis:

```bash
python -m fincantatem script.py
```

#### Options

```bash
python -m fincantatem script.py \
  --preset openrouter \
  --snippets true \
  --cautious false
```

- `-p, --preset` — Inference preset to use (default: `openrouter`)
- `-s, --snippets` — Show code snippets instead of full source (default: `true`)
- `-c, --cautious` — Enable secret and PII redaction (default: `false`)

### In Jupyter/IPython

Load as an extension to get automatic error analysis on cell failures:

```python
%load_ext fincantatem
```

Every exception can be analyzed automatically without (onerous) modification to your code. See the [example notebook](examples/05_data_science/main.ipynb) for more details.

## Examples

### 1. The "Silent API Change"

**Scenario:** You are consuming a third-party API. The code has worked for months. Suddenly, it breaks with a `KeyError`, but the API status code is still 200.

```python
@finite
def sync_user_data(user_id: str):
    response = requests.get(f"https://api.vendor.com/users/{user_id}")
    payload = response.json()

    # CRASH: KeyError: 'active_subscription'
    if payload["data"]["active_subscription"]["status"] == "active":
        update_local_db(user_id)
```

**Why this is hard:** Standard debugging requires you to add print statements to dump `payload`, rerun the script, and inspect the JSON structure. But here's the insidious part: _you might not be able to reproduce it_ because the API behavior depends on the specific user_id, time of day, or rate limit state.

**F. Incantatem Insight:** It captures the _actual response body_ that caused the crash—no reproduction needed.

**The Explanation:**

> "The code expects `payload['data']['active_subscription']['status']`. However, the actual `payload` captured during the crash contains `{'error': 'RateLimitExceeded', 'retry_after': 60, 'request_id': 'req_8x2k9'}`.
>
> The API vendor is returning application-level errors with HTTP 200 status codes—a common but poor API design pattern. Your code assumes `response.status_code == 200` means success, but you need defensive key checking or schema validation. The `request_id` in the payload suggests you can report this specific failure to their support."

---

### 2. The "Mutable Default Argument" Trap

**Scenario:** A classic Python foot-gun that manifests as "data bleeding" between requests in a long-running process.

```python
@finite
def add_audit_log(event, _buffer=[]):  # The bug is here
    _buffer.append(event)
    if len(_buffer) >= 3:
        flush_to_disk(_buffer)
        return True
    return False

# Imagine this running in a server context
add_audit_log("login")
add_audit_log("logout")
add_audit_log("view_page")  # Flushes
add_audit_log("login_admin")  # CRASH or security leak
```

**Why this is hard:** This is a _Heisenbug_—it disappears when you try to debug it with unit tests because each test starts fresh. The failure only emerges in production after the process has been running long enough. You might see symptoms like: duplicate database entries, security logs showing events from different users mixed together, or memory growing unbounded.

**F. Incantatem Insight:** It shows you the _identity_ of the buffer object and its complete contents at crash time.

**The Explanation:**

> "The local variable \_buffer contains ['login', 'logout', 'view_page', 'login_admin']—4 items when you expected 1.
>
> This is the mutable default argument trap. Your function signature is def add_audit_log(event, \_buffer=[]). The list contains events from previous function calls because Python evaluates default arguments once at definition time, then reuses the same object. You can verify this yourself by adding print(id(\_buffer)) at the start of the function—you'll see the same memory address across calls. In this case, the presence of 4 accumulated items instead of 1 is the smoking gun."

---

### 3. The "Unicode Normalization Bomb"

**Scenario:** A web application that's been running fine suddenly starts having bizarre duplicate key violations in your database, but only for certain users with international names.

```python
@finite
def create_user(username: str, email: str):
    # CRASH: IntegrityError - duplicate key value violates unique constraint "users_username_key"
    # But you just checked and the username doesn't exist!
    user = User.objects.create(username=username, email=email)
    return user
```

**Why this is hard:** Looking at your database, you see one entry for "José" and you're trying to insert "José"—they look identical. Your uniqueness check passes because Python's `==` operator says they're different. The bug appears random because it only affects certain character combinations.

**F. Incantatem Insight:** It shows you the actual byte representation of the `username` variable.

**The Explanation:**

> "The variable `username` contains the string `'José'` which appears as `'Jos\\u00e9'` (5 characters, with a single codepoint é). However, your database already contains a user with username `'Jose\\u0301'` (5 characters, with e + combining acute accent).
>
> These are two different Unicode representations of the same visual character—NFC (composed) vs. NFD (decomposed) normalization. Your form input is receiving NFD-normalized text (common on macOS), while your database contains NFC (common on Windows/Linux). Python's `User.objects.filter(username=username)` doesn't find a match because you're comparing different byte sequences. **The fix:** Apply `unicodedata.normalize('NFC', username)` before any database operation. This affects approximately 2,184 Unicode characters with multiple representations."

---

### Optional Features

For rich Markdown output in your terminal:

```bash
pip install "fincantatem[pretty]"
```

For automatic redaction of secrets and PII:

```bash
pip install "fincantatem[cautious]"
```

Or with uv:

```bash
uv add "fincantatem[pretty]"
uv add "fincantatem[cautious]"
```

## Data Risks

By default, the following is transmitted to the inference API:

- Source code and file paths
- Full stack trace
- Exception messages
- Local variables in the call stack
- Any values or secrets present in your code

This may be undesirable in environments handling sensitive information. F. Incantatem offers several mitigation strategies.

### Cautious Mode

When enabled, cautious mode attempts to automatically redact secrets and personally identifiable information from source code and local variables before transmission. Enable it with:

```python
@finite(cautious=True)
def my_function():
    pass
```

Or via the CLI:

```bash
python -m fincantatem script.py --cautious true
```

#### How It Works

Cautious mode employs two complementary detection mechanisms:

**1. Secret Detection** — Uses pattern matching and entropy analysis (via `detect-secrets`) to identify API keys, tokens, private keys, and other cryptographic material. Detected secrets are replaced with a deterministic placeholder derived from their SHA-256 hash, allowing correlation without exposing the actual value.

**2. PII Detection** — Uses Presidio, a Microsoft library trained to recognize personally identifiable information including names, email addresses, phone numbers, credit card numbers, and more. Detected entities are anonymized in-place.

#### Important Caveats

Cautious mode is a _best-effort_ mechanism, not a guarantee. Users should understand its limitations:

- **Incompleteness** — Detection patterns may miss secrets in unusual formats or custom PII that doesn't match known patterns. A secret hidden in a comment, a name disguised as a variable, or a partially exposed credential may slip through.

- **False Positives** — Legitimate values (e.g., a string containing "password") may be incorrectly flagged as sensitive. This can result in unnecessary redaction of benign data.

- **Heuristic Nature** — Both `detect-secrets` and Presidio rely on heuristics rather than definitive signatures. They are probabilistic tools designed for high recall at the cost of some precision.

- **No Guarantee** — Cautious mode reduces risk but does not eliminate it. **Never enable cautious mode as a substitute for data governance.** If transmitting source code to any external service is unacceptable in your environment, cautious mode is insufficient—use local inference instead.

- **Performance Cost** — Secret and PII scanning adds latency to exception processing, particularly on large codebases or with complex stack traces.

**Recommendation:** Cautious mode is suitable for development workflows where occasional false redactions are tolerable and where the code does not contain highly sensitive information by design. For production systems handling financial data, credentials, or protected health information, consider local inference or private endpoints instead.

### Alternative Mitigation Strategies

1. **Local Inference** — Run a model locally via [Ollama](https://ollama.ai) or [vLLM](https://docs.vllm.ai), keeping all data on your machine
2. **Private API** — Route requests through a trusted organizational endpoint with appropriate access controls and data handling policies
3. **Selective Decoration** — Use `@finite` only on non-sensitive functions, or on functions unlikely to encounter data exposure

## Performance Considerations

We have not benchmarked FI meaningfully yet, but we can make a very reliable estimate of its performance characterstics at runtime. There are two performance profiles:

| Case         | Performance Characteristics                                                                                                                                                                                                                                                                                                                                                                                             |
| ------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| No exception | If the wrapped function does microseconds or more of work (I/O, JSON, allocations, network, DB, nontrivial Python logic), this overhead is usually negligible.<br><br>If the wrapped function is a tiny hot-loop primitive (simple arithmetic, attribute access) called millions of times, `@finite` can be a noticeable slowdown (often tens of percent or more) because the baseline call is already extremely small. |
| Exception    | All of the characteristics of the no exception case, plus the additional overhead of the exception handling, analysis and LLM call. You're likely to notice a 2-3s delay in wind-down at exception time.                                                                                                                                                                                                                |

## Key Features

### Basic Debugging

```python
from fincantatem import finite

@finite
def divide(a: int, b: int):
    return a / b

divide(10, 0)  # Receives AI analysis of ZeroDivisionError
```

### Interactive Chat

```python
@finite(chat=True)
def load_config(path: str):
    with open(path) as f:
        return json.loads(f.read())

# After analysis, continue conversing with the AI
load_config("missing.json")
```

### Cautious Mode

```python
@finite(cautious=True)
def query_api(token: str):
    response = requests.get("https://api.example.com", headers={
        "Authorization": f"Bearer {token}"
    })
    return response.json()

# Token will be redacted before transmission
query_api("secret_key_123")
```

## Requirements

- Python 3.10+
- Optional: `rich` for formatted output
- Optional: `presidio-analyzer`, `presidio-anonymizer`, `spacy`, `detect-secrets`, `bip-utils` for cautious mode

## License

Apache License 2.0

## Roadmap

Planned features and improvements:

- **Richer Context Collection:**

  - Analyze type annotations and compare them to runtime values for better error diagnosis.
  - Capture dependency versions used in the traceback for context on API or environment issues.

- **Smarter Code Analysis:**

  - Use AST to retrieve full function/class definitions and key imports.
  - Summarize data and control flow to pinpoint where and how values led to the crash.
  - Show related code, assignments, and similar patterns across the codebase.

- **External Knowledge Integration:**

  - Fetch documentation for involved exceptions/functions.
  - Suggest relevant GitHub issues or Stack Overflow threads.
  - Check changelogs for breaking changes in third-party libraries.

- **Enhanced Output:**

  - Include severity and impact ratings.
  - Offer ready-to-apply code patches.
  - Generate suggested test cases.

- **Context-Aware Features:**

  - Detect common frameworks (Django, FastAPI, Pytest, etc.) and display specialized context.
  - Optionally surface recent git history and CI/environment info.

- **Advanced and Performance Features:**
  - Let the AI request additional context in multi-turn analysis mode.
  - Suggest reproduction steps, hypotheses, and incremental context loading.
  - Pre-classify and summarize exceptions locally before making LLM calls.

## Acknowledgements

The `pipe` utility is lifted from the [toolz](https://github.com/pytoolz/toolz) library.

## Status

F. Incantatem is actively maintained and used in production environments.
