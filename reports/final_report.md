# LawBot final report

## Problem and scope

LawBot is an educational Indian-law question-answering prototype built around supplied question/answer data for the Constitution, Indian Penal Code (IPC), and Code of Criminal Procedure (CrPC). The project demonstrates dataset preparation, a planned small-model fine-tune, retrieval-augmented generation (RAG), an optional GRPO experiment, tool routing, and a Streamlit interface. It is not a legal service and has not been validated for legal use.

## Data and preprocessing

The three supplied JSON files contain `question` and `answer` fields. Preprocessing normalized those fields into `instruction` and `output`, removed short and duplicate questions, retained answer text, extracted numeric section/article references, and made a seeded 80/20 train/validation split. Near-duplicate removal is based on character-trigram MinHash and is a heuristic.

| Source | Input records | Clean records | Train | Validation |
|---|---:|---:|---:|---:|
| IPC | 2,267 | 2,171 | 1,736 | 435 |
| CrPC | 8,194 | 7,770 | 6,216 | 1,554 |
| Constitution | 4,082 | 3,878 | 3,102 | 776 |
| **Total** | **14,543** | **13,819** | **11,054** | **2,765** |

Filtering removed 612 records below the minimum length, 21 exact duplicates, and 91 near-duplicate questions; the garbage/truncation filter removed none. There are 3,169 records with at least one extracted reference (22.9%). See [`preprocessing_report.md`](preprocessing_report.md) for source breakdowns, length statistics, and examples.

## Model and training plan

The fine-tuning notebook targets Qwen2.5-1.5B-Instruct with 4-bit QLoRA on a Colab/Kaggle GPU. It compares the base model with the adapter on validation loss/perplexity, ROUGE-L, BERTScore, citation-set accuracy, and source-wise results, and prepares examples for manual review. The optional GRPO notebook applies small, rule-based proxy rewards for reference overlap, answer similarity, formatting/length, and abstention.

**Training has not been run in this workspace.** No adapter, merged model, generated preference candidates, or GRPO checkpoint is included. These metrics must remain unreported until the notebooks are run; the proxy reward functions are not a measure of legal correctness.

| Planned model comparison | Validation loss / perplexity | ROUGE-L F1 | BERTScore F1 | Citation-set accuracy |
|---|---:|---:|---:|---:|
| Base model | Pending GPU run | Pending | Pending | Pending |
| QLoRA adapter | Pending GPU run | Pending | Pending | Pending |
| Optional GRPO adapter | Pending GPU run | Pending | Pending | Pending |

## Retrieval-augmented generation

The Phase 3 notebook downloads three historical primary-law PDFs from [India Code](https://www.indiacode.nic.in/), extracts page text, groups it around article/section boundaries, builds a normalized MiniLM FAISS index, and evaluates Hit@5 and mean reciprocal rank on a reference-bearing sample. The app requests up to the selected top-k passages and displays their source, section/article or page, excerpt, and a prototype cosine similarity score. The generation prompt is restricted to retrieved passages and asks the model to abstain if those passages do not support an answer. A fixed score floor of 0.30 is a prototype heuristic, not a calibrated confidence value.

**The PDFs, embeddings, vector index, and retrieval metrics have not been generated here.** The app handles a missing index by abstaining and showing that retrieval is unavailable. The evaluation report is pending a network-enabled notebook run. See [`rag_eval_results.md`](rag_eval_results.md).

| Retrieval measure | Result |
|---|---:|
| Corpus chunk count | Pending corpus build |
| Hit@5 | Pending notebook run |
| MRR | Pending notebook run |
| Abstention threshold | 0.30 prototype floor; not calibrated |

## Optional reinforcement learning

The GRPO experiment is documented as an optional, short GPU study. It uses synthetic out-of-scope prompts and rule-based proxies; it does not train on the preference seed CSV. The seed file contains chosen examples, while rejected generations remain pending the notebook run and manual review. There is no RL result to report. The experiment could reward surface similarity or reference-number overlap while missing legal meaning, so any measured gains would be exploratory only.

## Tool integration and app

The agent routes requests to a local glossary, optional Indian Kanoon case search (with a cached Supreme Court summary fallback), an illustrative date calculator, RAG, or a direct model callback. Each tool/retrieval call returns a trace with its name, arguments, result, and latency. Direct model answers are labeled unverified. The Streamlit chat app stores turn history for the session, offers top-k/RAG/tools controls, and places citation and trace expanders under each answer.

The cached case records are paraphrases of the [Supreme Court of India’s landmark judgment summaries](https://www.sci.gov.in/landmark-judgment-summaries/); they are not the full judgments. Live case search requires an `INDIANKANOON_TOKEN` and use must follow [Indian Kanoon API documentation](https://api.indiankanoon.org/documentation/) and its terms. The date calculator performs date arithmetic only; it does not determine statutory limitation or court deadlines.

## Evaluation plan and current status

| Component | Planned evaluation | Current result |
|---|---|---|
| Dataset | Schema/filter counts, split distribution, duplicate checks | Counts recorded in preprocessing report; no legal review of all answers |
| Fine-tuned model | Validation loss/perplexity, ROUGE-L, BERTScore, citation accuracy, source-wise scores, manual rubric | Not run; GPU adapter unavailable |
| RAG | Corpus/chunk statistics, Hit@5, MRR, examples, unsupported-query abstention | Not run; PDFs/index unavailable |
| GRPO | Pre/post metrics and qualitative examples | Not run; no checkpoint |
| Tools | Trace examples and manual route review | Implementation/notebook supplied; no live API run reported |

No model quality or legal-accuracy claim is made in this report.

## Limitations, ethics, and risks

- **Outdated legal corpus:** The supplied dataset and retrieval PDFs cover legacy IPC and CrPC material. The IPC, CrPC, and Evidence Act were replaced by the Bharatiya Nyaya Sanhita (BNS), Bharatiya Nagarik Suraksha Sanhita (BNSS), and Bharatiya Sakshya Adhiniyam (BSA), respectively, effective 1 July 2024. The dataset does not map old provisions to current statutes. Refer to the official [BNS](https://www.indiacode.nic.in/indiacode/handle/123456789/20062?col=123456789%2F1362&view_type=search), [BNSS](https://www.indiacode.nic.in/indiacode/handle/123456789/20099?view_type=browse), and [BSA](https://www.indiacode.nic.in/indiacode/handle/123456789/20063?col=123456789%2F1362&view_type=search) texts and subsequent amendments for current law.
- **Dataset quality:** Supplied answers are not comprehensively checked against authoritative sources. A clean schema and a reference match do not establish correctness.
- **Retrieval quality:** OCR/extraction and section parsing can miss or mislabel material; similarity scores can rank irrelevant passages. The threshold and validation sample are preliminary.
- **Generation risks:** The model may hallucinate, omit exceptions, misread a source, or give outdated information. Direct answers are explicitly marked unverified.
- **Tool scope:** Case-search summaries are explanatory paraphrases. Deadline arithmetic has no legal rule engine, holiday calendar, or jurisdiction-specific counting logic.
- **Responsible use:** The interface displays “LawBot is for educational use only and is not legal advice.” Users must verify law and citations against current primary sources and consult a qualified lawyer for personal matters. Do not enter sensitive personal or case data into an unreviewed deployment.
- **Data handling:** The demo keeps message history in Streamlit session state. Deployment operators should assess hosting, logging, retention, and access controls before exposing it to users.

## Future work

1. Build a current-law corpus for BNS, BNSS, and BSA, track amendments, and make legacy-to-current mappings explicit and human-reviewed.
2. Run training and retrieval notebooks on pinned, documented GPU/runtime configurations; archive exact model revisions, logs, and artifacts.
3. Create expert-reviewed, source-grounded evaluation questions, measure retrieval and answer faithfulness separately, and report uncertainty and failure cases.
4. Calibrate abstention on held-out in-scope and out-of-scope queries; audit citation resolution and source passages.
5. Add access controls, safe logging/retention defaults, versioned sources, and clear disclosure for any public deployment.

## Reproducibility and setup

See [`README.md`](../README.md) for local installation and launch instructions. `reports/eval_results.md`, `reports/rag_eval_results.md`, and `reports/grpo_eval_results.md` are pending templates, not measured results. The source PDFs, generated FAISS index, and model weights/adapters are intentionally not included.
