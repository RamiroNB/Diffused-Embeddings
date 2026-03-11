# Diffused-Embeddings

This repository contains two Jupyter notebooks used to extract sentence embeddings from large language models and evaluate _layer selection_ and _pooling strategies_ on MTEB tasks (e.g., STSBenchmark). The goal is to compare how different choices (which layer(s), how to pool tokens, whether to use IDF weighting) affect downstream embedding quality, while keeping the pipeline simple and reproducible.

## Repository Layout

- `llama-embeddings-extraction.ipynb`: Experiments using the LLaMA 3.1 8B model family (causal LM).
- `llada-embeddings-extraction.ipynb`: Experiments using the LLaDA 8B model family (encoder or causal fallback, depending on the checkpoint).
- `requirements.txt`: Minimal pinned dependencies (Transformers + MTEB).
- `results/`: Precomputed artifacts (CSV leaderboards, ZIP archives, and plots).
  - `results/csvs/`: Aggregated CSV outputs.
  - `results/*.zip`: Packaged result folders/CSVs for easy download from notebook environments.
  - `results/*.png`: Figures used for reporting/illustration.

## What The Notebooks Do

Each notebook follows the same high-level flow:

1. **Install dependencies** (Colab-style cells).
2. **Load model + tokenizer** and enable `output_hidden_states`.
3. **Define pooling functions** to convert token representations into a single sentence embedding:
   - `mean`, `max`, `eos`, `mean_no_special`, and `idf_mean` (IDF-weighted mean).
4. **Select layer representations**:
   - last layer, last-k average, or explicit indices (e.g., first/middle/last windows).
5. **Wrap everything in an encoder class** compatible with MTEB (`encode()` method).
6. **Build an experiment grid** over layer strategies × pooling strategies.
7. **Run MTEB evaluation** (typically STSBenchmark test split), collect `main_score`.
8. **Rank + export results** (print top configs, save CSV, optionally zip).
9. **(Optional) Plot** a quick comparison chart.

## Quick Start (Local)

### 1) Create an environment

Use any standard Python environment manager. Example with `venv`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Run the notebooks

Open one notebook at a time and run the cells top-to-bottom:

- `llama-embeddings-extraction.ipynb`
- `llada-embeddings-extraction.ipynb`

Notes:

- These notebooks are written to work well in **GPU notebook runtimes** (e.g., Colab). Running locally may require CUDA + appropriate PyTorch installation.
- Model downloads happen via Hugging Face; you may need to authenticate depending on model access.

## Reproducibility Notes

- Dependencies are pinned in `requirements.txt`.
- For exact reproducibility across machines, also pin Python version and CUDA/PyTorch build.
- The notebooks intentionally keep the code in-notebook (no hidden scripts) for transparency.

## Results and Artifacts

Precomputed results live under `results/`:

- CSV leaderboards (examples):
  - `results/csvs/results_llama_sts.csv`
  - `results/csvs/results_llada_sts.csv`
  - `results/csvs/all_tasks_results.csv`
- Zipped bundles for download/sharing (examples):
  - `results/results_llama_sts.zip`
  - `results/results_llada_sts.zip`

The top-level metric reported in the notebooks is `main_score` as provided by MTEB for the given task/split.

## Hardware and Runtime Expectations

- These experiments typically require a GPU with enough VRAM for 8B models (26GB VRAM will be sufficient).
- If you run in a notebook runtime:
  - set `device = "cuda"` (or update to CPU for debugging only).
  - expect model loading and MTEB evaluation to take minutes to hours depending on tasks/grid size.

## Common Customizations

- **Change model checkpoint**:
  - edit the `model_name` / `NAME` variable in the corresponding notebook.
- **Change grid size**:
  - adjust `layer_cfgs`, `poolings`, or batch size (`bs`) and max sequence length (`max_len`).
- **Add tasks**:
  - extend the MTEB task list in the runner cell(s).


## License

Add a license before public release (e.g., MIT/Apache-2.0) and ensure model usage complies with the corresponding model licenses and terms.
