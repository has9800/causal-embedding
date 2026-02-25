# Causal Embeddings Training Pipeline

This repository implements a modular training workflow for steering transformer embeddings toward causal structure.

## Pipeline Stages

1. **Probe dataset build** (`scripts/build_probe_dataset.py`)
   - Extracts hidden-state features from the frozen student base model for labeled probe sentences.
2. **Probe training** (`scripts/train_probe.py`)
   - Trains a frozen linear probe over intermediate hidden-state features.
3. **SFT warm-start** (`scripts/run_sft.py`)
   - Performs supervised fine-tuning on causal reasoning traces.
4. **RL-style orchestration** (`scripts/run_pipeline.py`)
   - Runs LangGraph nodes: multi-generation fan-out, filtering/scoring, probe scoring, reward ranking, preference pair construction, and checkpointing.
5. **Model export** (`scripts/save_model.py`)
   - Saves LoRA adapter and merged model artifacts.

## Key Design Choices

- **Student model**: GPT-Neo-2.7B for iteration, with GPT-J-6B production option in config.
- **LoRA**: Applied to attention projection modules (`q_proj`, `v_proj`) by default.
- **Dual critic path**:
  - Always run local filter.
  - Optional Anthropic critic (`critic.use_premium_critic=true`) for high-quality traces.
  - If premium critic is disabled, the local-only warm-trial path remains fully functional.
- **Reward split**: static 50/50 blend between normalized trace quality and probe-derived embedding score.
- **Checkpointability**: LangGraph state object stores history, rewards, and preference rows.

## Run Order

```bash
python scripts/build_probe_dataset.py
python scripts/train_probe.py
python scripts/run_sft.py
python scripts/run_pipeline.py
python scripts/save_model.py
```

## Configuration

- `config/training_config.yaml`: model, LoRA, critic, reward, and pipeline settings.
- `config/probe_config.yaml`: probe metadata and probe training hyperparameters.

## Notes

- Premium critic requires `ANTHROPIC_API_KEY` and enabling `critic.use_premium_critic`.
- Data files under `data/` are starter examples; replace with COPA/e-CARE/synthetic causal graph data for real training.
