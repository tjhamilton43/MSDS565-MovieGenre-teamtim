# MSDS 565 — Movie Genre Classification (`teamtim`)

This is my solo Project 3 submission. I use only movie plot synopses as model input and evaluate Action, Comedy, and Horror independently.

## Data
Course-provided **mpst_imdb_movies.csv** joins [MPST movie plot synopses](https://www.kaggle.com/datasets/cryptexcode/mpst-movie-plot-synopses-with-tags) with [IMDB non-commercial datasets](https://developer.imdb.com/non-commercial-datasets/). The course CSV is not committed to GitHub; place it at the repository root to run the notebooks.

## Notebook order
1. `notebooks/P3_MovieGenre-1_EDA_teamtim.ipynb`: explore movie metadata, 100-vote filter, genre rates, text statistics, and vocabulary.
2. `notebooks/P3_MovieGenre-2_Modeling_teamtim.ipynb`: TF-IDF and 384-dimensional **`sentence-transformers/all-MiniLM-L6-v2`** embeddings, 5-fold macro-F1-tuned logistic regression, held-out evaluation, OpenAI **`gpt-4o-mini`** teacher pseudo-labels and student evaluation. Exports fitted model and TF-IDF preprocessing files to `models/`.
3. `notebooks/P3_MovieGenre-3_xAI_teamtim.ipynb`: TF-IDF model coefficients, SHAP mean-impact/beeswarm and individual prediction waterfall plots.

## How to run
Open Jupyter with the `moviegenre` conda kernel. Keep the CSV in the root, and run notebooks in numerical order. Notebook 2 uses an `OPENAI_API_KEY` set in the local environment (never commit your key). It checkpoints each teacher label under `local_artifacts/teacher_checkpoint.csv`, which can be resumed after API rate limits. Notebook 3 requires the files generated in `models/` by Notebook 2.

**Note:** Teacher annotation requires actual API access and usage credits. The notebooks do not fabricate labels or experiment results. A completed submission should have all notebook cells executed on your machine and outputs saved.

## Environment
Run `conda activate moviegenre` and `conda env export --no-builds > environment.yml` from the repository root **after** finishing installations. This is deliberately not a fabricated export from a different machine.

## Non-repository files
Do not commit `mpst_imdb_movies.csv`, local artifacts, `.npy` embedding caches, API keys, or CSV exports. GitHub submission includes `README.md`, `environment.yml`, three executed notebooks, and exported `models/*.pkl` (subject to GitHub's file-size limit).
