# Assignment 3: Jet Generation Challenge

Build the best conditional flow matching model for particle physics jet generation. Top 2 on the leaderboard win 2 months of Claude Max.

## Structure

- **Assignment description**: `assignment-03.pdf`
- **Starter code**: `code/assignment-03-starter.ipynb`
- **Shared utilities**: `code/utils.py`
- **Generation template**: `code/generate.py`
- **Evaluation script**: `code/evaluate.py`
- **Your report**: `report/report.tex`

## Getting Started

### 1. Download the data

The data files are hosted on Google Drive (~460 MB total). Download them into `code/data/`:

```bash
cd code
python download_data.py
```

This creates `code/data/train.npz` and `code/data/val.npz`.

### 2. Open the notebook

**Google Colab (recommended — training is much faster on GPU)**

Students can get free Colab Pro at [colab.research.google.com/signup](https://colab.research.google.com/signup), which provides access to A100, L4, or T4 GPUs — any of these are fine for this assignment.

1. Go to [colab.research.google.com](https://colab.research.google.com/) and upload the notebook: **File → Upload notebook**, then select `code/assignment-03-starter.ipynb`.
2. Enable GPU: **Runtime → Change runtime type → T4 GPU** (or A100/L4 if available), then click Save.
3. Upload the required files using the **Files pane** on the left sidebar (folder icon):
   - `code/utils.py`
   - `code/generate.py`
   - Create a `data/` folder and upload `code/data/train.npz` and `code/data/val.npz`
4. After training, download `model_params.pkl` and `submission.npz` from the Files pane (right-click → Download) and save them to your local `code/` directory before committing.

**Local (VS Code / JupyterLab)**

1. Install dependencies: `pip install -r code/requirements.txt`
2. Open `code/assignment-03-starter.ipynb`

### 3. Build your model

1. Run the baseline end-to-end to understand the pipeline
2. Improve the model (see Section 5 of the notebook for ideas)
3. Update `code/generate.py` with your model architecture
4. Write your report in `report/report.tex`

## Submitting

Make sure these files are in your `code/` directory:
- `submission.npz` — 2,000 generated jets per type
- `generate.py` — self-contained script that reproduces `submission.npz`
- `model_params.pkl` — your trained model weights

```bash
git add .
git commit -m "Complete assignment 3"
git push
```

You can push multiple times — only the final version at the deadline will be graded.

## Evaluation

Your leaderboard score is the composite W1 distance across 5 jet types × 4 observables (mass, η, φ, pT). Lower is better. Evaluate locally against the validation set:

```bash
cd code
python evaluate.py submission.npz --reference data/val.npz
```

Final ranking uses a held-out test set.

## Resources

- [Lecture notes: Flow Matching](https://bu-ds595.github.io/course-materials-spring26/notes/11-flow-matching-script.pdf)
- [Practical Flow Matching notebook](https://bu-ds595.github.io/course-materials-spring26/) (galaxy generation)
- [JetNet paper (Kansal et al., 2021)](https://arxiv.org/abs/2106.11535)
- [JetNet documentation](https://jetnet.readthedocs.io/)
