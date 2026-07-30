# Vendored chat templates

Fetched verbatim from the HF model repos:

- `glm52.jinja`  — `zai-org/GLM-5.2` / `chat_template.jinja`
- `qwen36.jinja` — `Qwen/Qwen3.6-35B-A3B` / `chat_template.jinja`

`test_ir.py` renders IR output through both. They are the authority for the turn
grammar in `anyharness.ir_grammar`: each rule there cites the line here that
motivates it, so re-verifying against a new model means diffing a new template in
rather than trusting the docstring.

Vendored rather than fetched at test time so the suite runs offline and a silent
upstream template change cannot turn into a mysterious CI failure.
