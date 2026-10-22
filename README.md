# LawBot

LawBot is an educational Indian law Q&A assignment. The repository is being built phase by phase.

## Phase 1: Dataset preparation

The supplied source data uses `question` and `answer` fields. Run preprocessing from the repository root with Python 3.10 or newer:

```powershell
python src/preprocess.py
```

This creates `data/processed/lawbot_cleaned.jsonl`, `train.jsonl`, `val.jsonl`, and `reports/preprocessing_report.md`. The processor uses only the Python standard library and a fixed seed of 42.

## Phase 2: Fine-tuning (Colab/Kaggle)

Open `notebooks/finetune_lawbot.ipynb` in Colab or Kaggle, enable a T4 GPU, and make the `lawbot/` project folder available in the runtime. The notebook installs the pinned Phase 2 packages from `requirements-training.txt` while retaining the runtime's CUDA-matched PyTorch installation. Set `LAWBOT_PROJECT_DIR` if automatic project discovery does not find the repository. The default is Qwen2.5-1.5B-Instruct with 4-bit QLoRA; the notebook config also includes Phi-3-mini and Mistral-7B choices.

If the runtime throws a `libtorchaudio.so`/`undefined symbol` error, it is a broken `torch`/`torchaudio` ABI mismatch rather than a model bug. The notebook automatically reinstalls a matching `torchaudio` build for the active `torch` version before importing the training stack; when running locally, use a supported CUDA runtime or reinstall `torch` and `torchaudio` together from the same release line instead of mixing wheels.

The notebook saves the adapter under `adapters/`, optionally attempts a merged 16-bit export for the default Qwen run, generates base-versus-fine-tuned metrics and five examples, and creates `reports/manual_rubric.csv` with 30 review samples. After you run the notebook, it populates `reports/eval_results.md` and `reports/eval_results.json`. TensorBoard logs are written to `runs/`. Adapter and merged files live in the notebook runtime unless you copy them to persistent storage.

The notebook has not been run on a GPU in this workspace. Its result report is a pending template until a Colab/Kaggle run completes. The free T4 target and CUDA-specific package compatibility should be confirmed in that runtime.

## Phase 3: Retrieval-augmented Q&A

Open `notebooks/rag_lawbot.ipynb` in Colab/Kaggle and run the cells in order. The notebook installs the pinned packages in `requirements-rag.txt`, downloads three public-law PDFs from India Code, extracts page text, chunks around article/section boundaries, builds a normalized MiniLM FAISS index, and generates a chunk-size chart. PDF downloads and embedding/model downloads require network access. The generated text, chunks, and index are written under `data/rag_corpus/`; they are not bundled in this project.

The demo uses the Phase 2 Qwen adapter from `adapters/lawbot-qwen1.5b` if available; otherwise it reports that it is using the base model. It rewrites follow-up questions, retrieves with MMR, optionally applies a cross-encoder reranker, cites sources, and demonstrates abstention. The notebook reports Hit@5/MRR on a small reference-bearing validation sample and chooses an illustrative similarity-score floor using that same sample plus synthetic out-of-scope prompts. These are prototype diagnostics, not evidence of legal correctness. Similarity scores are not calibrated probabilities. The Supreme Court case records are brief paraphrases of its official summaries page; the Court states those summaries are not part of its decisions or reasons and are not for use in legal proceedings.

On successful execution, the notebook writes `reports/chunk_statistics.png` and `reports/rag_eval_results.md`. It has not been run here because the workspace lacks network access and a GPU, so those generated results are not included.

## Phase 4: Optional GRPO/RLHF experiment

Open `notebooks/reward_model_lawbot.ipynb` in Colab/Kaggle with a T4 GPU and run it in order. The notebook uses the pinned TRL/Unsloth stack in `requirements-training.txt`, creates 40 base-Qwen rejected-answer candidates, then trains a short 4-bit QLoRA GRPO run from the Phase 2 adapter when available (otherwise from the base model). Its four rule-based reward functions score citation overlap, answer-text similarity, length/format, and abstention on eight synthetic out-of-scope prompts.

`reports/preference_seed.csv` contains 40 seeded prompt/chosen pairs sampled from training data; rejected responses are blank until the notebook generates them. The notebook writes `reports/preference_candidates.csv`, where all generated pairs remain marked for manual review. The GRPO update uses the reference-bearing training examples directly with rule-based rewards; it does not learn from the preference CSV. This keeps the requested preference review artifact distinct from the GRPO objective. After the GPU run, the notebook compares pre/post validation loss and perplexity, ROUGE-L, BERTScore, citation accuracy, and source-wise ROUGE-L in `reports/grpo_eval_results.md`.

No GRPO adapter, rejected generations, or post-training metrics are included yet because this workspace has no CUDA GPU. The small synthetic reward experiment can overfit and is not a legal correctness measure. Any answer generated without retrieved evidence must be treated as unverified model knowledge.

## Phase 5: Tool-calling agent

`src/tools.py` defines JSON schemas for case-law search, local glossary lookup, and illustrative deadline arithmetic; `src/agent.py` routes requests to those tools, Phase 3 retrieval, or a direct-answer callback. Each execution returns a trace with tool name, arguments, result, and latency. The agent labels direct model output as unverified and abstains when no answer model or RAG source is available.

For live Indian Kanoon searches, set `INDIANKANOON_TOKEN` in the runtime environment. The API uses an authorization token header; the official docs and terms require attribution. Without a token, the tool searches the bundled Supreme Court summary paraphrases and marks the trace `cached_fallback`; these are not live case-law results and the Court says the summaries are not judgments. See `data/legal_glossary.json` for the local educational glossary. The deadline calculator supports calendar/business days and optional caller-supplied holidays, but it does not apply legal rules or infer court holidays.

Open `notebooks/lawbot_agent.ipynb` for tool and route examples. After running it, the notebook writes `reports/tool_trace_examples.json`. The RAG example requires a FAISS index generated by Phase 3; without one, it fails closed with an abstention. The direct-answer example is a trace-only placeholder until an answer model callback is connected.

The data reflects legacy IPC and CrPC material. The IPC, CrPC, and Evidence Act were replaced by the BNS, BNSS, and Bharatiya Sakshya Adhiniyam, effective 1 July 2024. This Phase 1 dataset does not map or verify those equivalents. Check current statutes and effective dates before relying on any generated answer; LawBot is for educational use and is not legal advice.

## Phase 6: Streamlit app and final report

`app.py` provides a session-based chat interface with starter questions, a glossary and date-tool setting, optional answer details, and source citations when a RAG index is present. Locally, it can load Qwen3-4B-Instruct-2507 and an optional local adapter. On Streamlit Community Cloud, it sends chat prompts to Hugging Face Inference Providers using `HF_TOKEN`, keeping the model weights off the small web host. Set `LAWBOT_API_MODEL` to choose another supported hosted model. Hosted inference usage is billed by the selected provider. Legal model answers without an installed source index are unverified and must be checked against current official sources.

For local inference, install the app dependencies and a PyTorch build suited to your machine, then launch:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-app.txt
streamlit run app.py
```

Install a PyTorch build suitable for your host separately. For RAG, install `requirements-rag.txt` as well, then run `notebooks/rag_lawbot.ipynb` to download the listed official PDFs and build `data/rag_corpus/faiss_index/`. A GPU is recommended for model inference; CPU execution may be slow and requires enough memory. The weights, adapter, PDFs, and generated index are not included in this project. If retrieval assets are absent, the agent abstains on retrieval-routed questions. To use live Indian Kanoon search, set a token in the current PowerShell session before launching:

```powershell
$env:INDIANKANOON_TOKEN = "your-token"
streamlit run app.py
```

Without a token, case-law search uses the bundled Supreme Court summary paraphrases where relevant and labels that fallback in its trace. Do not treat the summaries as full judgments. The app's date calculator is illustrative arithmetic only. Read [`reports/final_report.md`](reports/final_report.md) for scope, evaluation status, limitations, and next steps. Training, retrieval evaluation, and GRPO results remain pending GPU/network-enabled runs.

### Streamlit Community Cloud

The root `requirements.txt` contains only the web app dependencies. After publishing the repository, deploy `app.py` from the `main` branch. In the app's **Manage app → Settings → Secrets**, add `HF_TOKEN = "hf_…"` using a Hugging Face user token with **Inference Providers** permission. Never commit the token. Requests are sent to Hugging Face Inference Providers; usage is billed to the Hugging Face account. If the token is not configured, the app can load but hosted model answers are unavailable.
