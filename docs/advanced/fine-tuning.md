# Training Data Export & Fine-Tuning

Every interaction accumulates signal in the database. Export it as JSONL and fine-tune your local models on **your** specific vocabulary, projects, and routing preferences.

## Export

```
/hub-export-training                      # default dir
/hub-export-training ~/my-training-data   # custom dir
```

```
Training data exported to ~/.claude/mcp-skill-hub/training/
  feedback.jsonl:  12 pairs   ← (query, skill, helpful) preference pairs
  triage.jsonl:    43 pairs   ← (message, action) classification pairs
  compact.jsonl:   21 pairs   ← (summary, digest) compaction pairs

Total: 76 training pairs
```

## Three signal types

| File | Source | Trains the model to… |
|------|--------|----------------------|
| `feedback.jsonl` | `record_feedback()` + implicit feedback | Identify relevant skills for your queries |
| `triage.jsonl` | `triage_log` table | Route **your** specific phrasing (`"FAO catalog status"` → `local_action`) |
| `compact.jsonl` | Closed tasks | Produce digests in **your** style with **your** terminology |

## Recommended path — Apple Silicon, no GPU

```bash
pip install mlx-lm

# Fine-tune the triage model (1.5b, fastest)
mlx_lm.lora \
  --model mlx-community/deepseek-r1-distill-qwen-1.5b-4bit \
  --train --data ~/.claude/mcp-skill-hub/training \
  --num-layers 8 --iters 200 --batch-size 4

# Fuse and save as Ollama model
mlx_lm.fuse --model mlx-community/deepseek-r1-distill-qwen-1.5b-4bit \
  --adapter-path adapters --save-path ~/my-triage-model

# Create Ollama modelfile
echo 'FROM ~/my-triage-model' > Modelfile
ollama create skill-hub-triage -f Modelfile

# Activate
configure(key="reason_model", value="skill-hub-triage")
```

## What you get

At ~200+ examples the fine-tuned model will:

- Recognize your project names (e.g. `geoid`, `dynastore`, `FAO`)
- Know your domain vocabulary
- Route your preferred phrasing correctly
- Produce digests that sound like **you** write

…making triage significantly more accurate than a generic base model.

## Alternative — OpenAI-compatible endpoints

If you use OpenRouter, LiteLLM, or a Fireworks/Together account:

1. Run the same export
2. Convert JSONL to the fine-tuning format of your chosen provider
3. Point `remote_llm` at the resulting custom model

```
configure(key="remote_llm",
  value='{"base_url":"https://api.provider.com","api_key":"sk-...","model":"ft:your-triage-v1","timeout":120}')
configure(key="local_models", value='{"level_4":"remote:https://api.provider.com"}')
```

## Related

- [advanced/context-bridge.md](context-bridge.md) — same signal fuels live context and training
- [features/learning.md](../features/learning.md) — how the signal accumulates in the first place
