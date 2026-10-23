# MLE routed model

Use MLE as one OpenAI-compatible model named `mle`. A request is automatically routed to the
Math, Search, Tau, or instruction-following (IF) expert, and that expert generates
the answer. Clients do not select or load experts.

## Setup

Use Python 3.12 on a CUDA-capable Linux machine. From the repository root:

```bash
python3.12 -m venv outputs/mle/venv
source outputs/mle/venv/bin/activate
python -m pip install -r slime/examples/mle/requirements.txt
```

Set the four empty `model_id` fields in [experts.json](experts.json) to the
Hugging Face repository IDs of the merged expert checkpoints. The classifier's
public model ID is already configured in [router.json](router.json).

## Start the model

Choose one loading mode. Both expose `http://127.0.0.1:30610/v1` and accept the
same requests.

```bash
MLE_GPUS=0,1 bash slime/examples/mle/run_mle.sh
```

## Send a request

With the OpenAI Python client:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:30610/v1", api_key="unused", timeout=7200)
response = client.chat.completions.create(
    model="mle",
    messages=[{"role": "user", "content": "What is 17 times 23?"}],
    max_tokens=16384,
)
print(response.choices[0].message.content)
```

`GET /v1/models` lists `mle`. `POST /v1/chat/completions` returns a standard chat
completion whose `model` is also `mle`. This example supports text messages and
non-streaming responses. Generation parameters supplied by the client are
forwarded to the selected expert.

For a multi-turn conversation, send the full message history and reuse an
`X-Conversation-ID` header (for example, through the client's `extra_headers`).
The gateway creates and returns an ID when none is supplied. The selected expert
is fixed for that ID and persists across restarts. Without reusing the ID, each
request is routed again from the first user message in its history.

Tools and tool-choice settings are forwarded to the expert. As with an ordinary
model API, the client executes returned tool calls and sends the tool results
back with the conversation history and the same conversation ID.

## Routing and generation

The router first inspects available function schemas for recognizable search or
business operations. All other inputs are classified by Base using the
[prompt](mle/prompt.txt). Rules do not inspect question wording. The
classifier sees the original system prompt, available tools, any assistant
messages preceding the first user request, and that request. Its instructions
and classification output are never added to the expert's answer context.

| Classifier label | Answering expert |
|---|---|
| `reasoning` | Math |
| `search` | Search |
| `customer_service` | Tau |
| `instruction_following` | IF |

Classification uses plain completion, greedy decoding, seed 42, and a 64-token
output limit. Regex-constrained decoding restricts the output to the four label
objects, without a separate thinking stage.

The pinned runtime is SGLang 0.5.10, BF16, a 32,768-token context, and disabled
prefix caching. Experts use their chat templates with `enable_thinking=true`,
seed 42, `stop=["<|im_end|>"]`, and `skip_special_tokens=false`. Other generation
parameters use the expert service defaults unless supplied by the client.
