# Prompt templates

Default templates are packaged in [`src/ecg_few/prompts/templates/`](../src/ecg_few/prompts/templates/). Edit them there to keep one source of truth.

Use `--system-prompt-file PATH` and `--prompt-file PATH` for local overrides. Templates support the row placeholders used by the VLM pipeline, including `{lead}`.
