# Leukemia Harmoniser

Ingests messy hospital CBC (complete blood count) CSV files, harmonises them to a
standard schema, runs pre-trained XGBoost / Random Forest / Neural Network models,
and presents the results in a dashboard with a downloadable PDF report.

> **Research / educational prototype. Not a medical device and not for clinical use.**
> Models are trained on synthetic data.

## Pipeline
1. **Upload** CSV (UTF-8 / Latin-1 / CP1252)
2. **Harmonise**: drop non-CBC columns, fuzzy-match column names to 17 canonical
   features, fix unit scales (10⁹/L vs cells/µL), clip outliers to clinical ranges,
   fill missing values with population defaults
3. **Predict** with 3 models; all-agree = High Risk, all-negative = Low Risk,
   otherwise Inconclusive
4. **Dashboard + PDF report**

## Running (Google Colab)
1. Open a Colab notebook and add `NGROK_TOKEN` under **Secrets** (key icon, left sidebar),
   with notebook access enabled. Get a token at https://dashboard.ngrok.com.
2. Open `notebooks/leukemia_harmoniser.ipynb` in Colab (or paste
   `app/colab_cell1_install.py` and `app/colab_cell2_system.py` as two cells)
3. Run cell 1 (installs), then cell 2, and upload the three `.pkl` files from
   `trained ml models/` when prompted
4. Open the printed ngrok URL

## Repository layout
| Path | Contents |
|---|---|
| `notebooks/` | Colab notebook (outputs cleared) |
| `app/` | Same code as plain `.py` Colab cells: FastAPI backend + embedded JS dashboard |
| `trained ml models/` | Pre-trained `.pkl` models (XGBoost, RF, NN) |
| `training related data/` | Synthetic training / prediction datasets |
| `other datasets/`, `new datasets/` | Messy multi-source test exports |
| `leukemia-harmonizer/` | Earlier Streamlit prototype and evaluation figures |
| `explanations.pdf`, `ultra condensed.pdf` | Architecture write-ups |

## License
Released under the [MIT License](LICENSE).

## Security note
`.pkl` files are Python pickles; loading one executes code. Only load models from
a source you trust.
