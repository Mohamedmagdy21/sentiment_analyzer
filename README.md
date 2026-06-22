# Brand Mood Ring

Know what your customers are feeling — before they churn.

![Brand Mood Ring UI](ui_hero.png)

Every review, support ticket, and social mention is a signal that most businesses ignore until the quarterly report. This project turns those signals into real-time sentiment data and — more importantly — **tells you when the way people talk about your brand has shifted**, so you can adapt before your engagement metrics do.

---

## Why this exists

Sentiment analysis is not a one-time ML project. It is a business feedback loop.

Language evolves. Internet culture redefines words overnight. "Sick" meant disgusting ten years ago — today it means awesome. A model trained on last year's data will miss half of today's negatives, which means you miss churn signals, which means you lose customers you could have kept.

This project is built around three principles:

### 1. Cost efficient
Open-source **RoBERTa** runs on a 4 GB GPU. All fine-tuning uses **LoRA adapters** — megabytes instead of gigabytes, minutes instead of hours. The base model stays in memory permanently; only the tiny adapter gets swapped.

### 2. Plug-in / plug-out
Change `pretrained_name` in one YAML file and the entire project rewires itself — preprocessing, inference, embedding extraction for drift. Drop an `adapter_config.json` into the models folder and the server picks it up on restart. Delete it and it falls back to the base model. No code changes.

### 3. Monitoring or it didn't happen
Four drift signals run continuously. When any crosses the alert threshold, a card turns red in the dashboard and you know it is time to retrain — not because a calendar told you to, but because the data did.

---

## The UI — what does what

### Classify + Cache

Upload a CSV with a `text` column. Every row runs through a two-model ensemble (one trained on reviews, one on social media). Results stream back with sentiment, confidence, and which model decided.

The UI caches the last upload by file hash. Re-upload the same file and results load instantly — no redundant inference.

### Three chart views — why each exists

The bar chart shows positive / neutral / negative counts across three views:

| View | Resets | Why it exists |
|---|---|---|
| **Inference** | Every new upload | You just ran a batch. See what the model thinks, right now. |
| **Accumulation** | Every quarter (configurable) | This is the quarter so far. Use it mid-quarter to spot trends and adjust campaigns, support staffing, or product messaging — not after the quarter ends. |
| **Memory** | Never (grows) | Last quarter's data lives here. Compare Q3 vs Q4. Did a pricing change shift sentiment? Did the new feature launch actually improve things? |

**Business logic:** Accumulation is your canary. Memory is your history book. Together they turn sentiment from a lagging KPI into a leading one.

### System metrics

Three numbers under the chart:

- **Avg Latency** — milliseconds per classification. If this creeps up, the model might need optimization or the hardware is bottlenecked.
- **Predictions** — total since server start. Usage tracking.
- **Avg Confidence** — mean model confidence. If it drops over time, the model is seeing data it is less sure about — a leading drift indicator before PSI confirms it.

### Model Drift Risk

Four cards at the bottom right. Each shows a PSI (Population Stability Index) value — a statistical measure of how much a distribution has shifted from training to now:

| Card | What it measures | If it turns red |
|---|---|---|
| **Data Drift** | Text length distribution | Customers changed how they write (shorter reviews, longer tickets). May need preprocessing adjustments or a new data sample. |
| **Prediction Drift** | Model confidence distribution | The model is more (or less) confident than it was. Something about the incoming data is unfamiliar. |
| **Target Drift** | Sentiment label distribution | The mix of positive/neutral/negative has shifted. Customers are genuinely changing how they feel. |
| **Semantic Drift** | Embedding distribution (PCA + KMeans → PSI) | The *words people use* are changing. "Sick" started meaning cool. The model is blind to this. This card is your earliest warning. |

PSI is unbounded. The risk color gradient is: **low < 0.1 → medium < 0.2 → high < 0.3 → critical ≥ 0.3**. The **alert threshold is PSI ≥ 0.25** — any card hitting that is your signal to collect fresh data and trigger a LoRA retraining run.

> A red drift card does not mean the model is broken. It means the world moved, and the model hasn't. That is a much cheaper problem to fix.

---

## Architecture

```
Airflow DAGs
  Base Pipeline (schedule) ──► artifacts/models/<name>/
                                    ├── adapter (LoRA)
  PEFT Training (manual)    ──►    └── monitoring/ baselines
                                          ├── data_drift_baseline.json
                                          ├── prediction_drift_baseline.json
                                          ├── target_drift_baseline.json
                                          ├── semantic_pca.pkl + kmeans.pkl
                                          └── semantic_expected.npy
                                            │
FastAPI Server                   ◄──────────┘
  /predict ─► decision_engine ─► predictions.jsonl ─► PSI computation
  /drift/metrics ─► Prometheus gauges ─► Grafana dashboard
  /lifecycle/metrics ─► accumulation + memory windows
```

| DAG | Trigger | What it does |
|---|---|---|
| `base_model_dag` | Schedule | Preprocess → evaluate base model → generate semantic baselines |
| `peft_training` | Manual (from drift alert) | Train LoRA adapter → evaluate → generate all baselines → deploy |

---

## Quick start

```bash
git clone https://github.com/Mohamedmagdy21/sentiment_analyzer.git
cd sentiment_analyzer
python3 -m venv venv_cuda && source venv_cuda/bin/activate
pip install -r requirements.txt

# Preprocess
python3 -m preprocessing.preprocess dataset=twitter
python3 -m preprocessing.preprocess dataset=amazon

# Evaluate base model
python3 -m evaluation.evaluate dataset=twitter model=twitter_roberta evaluator.use_peft=False

# Generate baselines (see README quick-start section for full command)
# Then:
docker compose up -d
```

Open `http://localhost:8000`, upload a CSV, click **Classify CSV**.

---

## Training (when drift says so)

```bash
python3 -m training.train dataset=twitter model=twitter_roberta
python3 -m training.train dataset=amazon model=amazon_roberta
```

Restart the container and the server picks up the LoRA adapter automatically.

---

## Project structure

```
├── configs/model/          # YAML — change pretrained_name to swap models
├── dags/                   # Airflow pipelines
├── inference/              # FastAPI server + drift monitoring
├── training/               # LoRA fine-tuning
├── evaluation/             # Model evaluation (with/without PEFT)
├── preprocessing/          # Text cleaning + train/val/test split
├── scripts/                # Production drift jobs (24h cron)
├── Data/                   # Raw (DVC) + processed CSVs
├── docker-compose.yml
└── prometheus/             # Alert rules
```

---

## License

MIT
