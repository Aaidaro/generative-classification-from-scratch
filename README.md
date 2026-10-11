# Generative Classification Methods from Scratch

**A machine learning coursework project exploring discriminant analysis and Naive Bayes classifiers in Python.**

This repository contains three notebook-based experiments: (1) Gaussian discriminant analysis on PCA-reduced Iris flowers, (2) Gaussian Naive Bayes for numerical features, and (3) multinomial Naive Bayes for baseball-versus-hockey bag-of-words classification. The core parameter estimates, discriminant functions, decision rules, and evaluation functions are implemented with NumPy; scikit-learn is used for the built-in Iris dataset and PCA.

## Contents

- [Experiments](#experiments)
- [Mathematical background](#mathematical-background)
- [Results](#results)
- [Getting started](#getting-started)
- [Dataset files](#dataset-files)
- [Implementation notes and limitations](#implementation-notes-and-limitations)

## Experiments

### 1. Gaussian Discriminant Analysis: Iris

The notebook loads the 150-sample, three-class Iris dataset and projects its four measured features onto **two principal components** for visualization. It then:

- Estimates class mean vectors and a **shared (pooled) covariance matrix** for linear discriminant analysis (LDA).
- Evaluates linear discriminant scores and plots two-class and three-class decision regions.
- Computes pairwise linear decision boundaries and compares equal priors (`[0.5, 0.5]`) with unequal priors (`[0.1, 0.9]`).
- Estimates **class-specific covariance matrices** for quadratic discriminant analysis (QDA).
- Samples 80% of Iris observations with `np.random.seed(54)` and repeats parameter/region analysis.
- Implements QDA posterior probabilities and a classifier that **rejects** uncertain samples (prediction `-1`).

### 2. Gaussian Naive Bayes: Numerical Features

This experiment reads `guassianBayes.mat`, estimates class frequencies and per-feature Gaussian means and variances, and predicts labels using sums of log likelihoods. It reports overall accuracy and class-specific precision and recall and writes selected metrics to `evaluation.txt`.

### 3. Multinomial Naive Bayes: Sports Text

Precomputed bag-of-words count matrices represent **baseball (label 0)** and **hockey (label 1)** documents. The notebook:

- Estimates per-class word probabilities using maximum likelihood estimation (MLE).
- Scores documents using word-count-weighted log probabilities.
- Re-estimates probabilities with **additive smoothing**, called `beta` in the code.
- Examines classification accuracy over different smoothing values and different numbers of training samples per class.

The notebook creates binary labels using 50 documents per class for each training and test split. Its input CSV files must use the row ordering and dimensions expected by the code.

## Mathematical background

Let $x\in\mathbb{R}^d$ be a feature vector, $k$ a class, $N_k$ its sample count, and $\pi_k=P(y=k)$ its prior probability.

### PCA

The Iris features are transformed into a two-dimensional subspace using the first two principal components:

$$z=W_2^\top(x-\bar{x}),$$

where the columns of $W_2$ are principal directions. `PCA(n_components=2)` performs this step; the notebook does not implement PCA from scratch.

### LDA: shared-covariance Gaussian model

The maximum-likelihood class means and pooled covariance used in the notebook are:

$$\hat\mu_k=\frac{1}{N_k}\sum_{i:y_i=k}x_i,$$

$$\hat\Sigma=\frac{1}{\sum_kN_k}\sum_k\sum_{i:y_i=k}(x_i-\hat\mu_k)(x_i-\hat\mu_k)^\top.$$

Under a shared covariance Gaussian model, the class score (ignoring class-independent constants) is:

$$g_k(x)=x^\top\Sigma^{-1}\mu_k-\frac12\mu_k^\top\Sigma^{-1}\mu_k+\log\pi_k.$$

The predicted class maximizes $g_k(x)$. The boundary between classes $i$ and $j$ is linear:

$$w_{ij}^\top x+w_{0,ij}=0,$$

$$w_{ij}=\Sigma^{-1}(\mu_i-\mu_j),$$

$$w_{0,ij}=-\frac12\left(\mu_i^\top\Sigma^{-1}\mu_i-\mu_j^\top\Sigma^{-1}\mu_j\right)+\log\frac{\pi_i}{\pi_j}.$$

Changing class priors shifts the intercept and hence the decision boundary.

### QDA: class-specific covariance matrices

Instead of one shared covariance matrix, QDA estimates a separate $\Sigma_k$ for each class. Its score is:

$$g_k(x)=-\frac12(x-\mu_k)^\top\Sigma_k^{-1}(x-\mu_k)-\frac12\log|\Sigma_k|+\log\pi_k.$$

The notebook converts these scores to posteriors with a numerically stabilized softmax:

$$P(y=k\mid x)=\frac{e^{g_k(x)}}{\sum_j e^{g_j(x)}}.$$

The rejection rule implemented in the code is:

$$\hat y(x)=\begin{cases}\arg\max_kP(y=k\mid x),&\max_kP(y=k\mid x)\ge 1-\lambda,\\-1,&\text{otherwise}.\end{cases}$$

Here `-1` means the classifier abstains; the notebook plots an example using $\lambda=0.2$.

### Gaussian Naive Bayes

For each class $k$ and feature $j$, the model estimates $\mu_{jk}$ and $\sigma_{jk}^2$ and assumes conditional independence of features given class:

$$P(x\mid y=k)=\prod_{j=1}^d\mathcal{N}(x_j;\mu_{jk},\sigma_{jk}^2).$$

For numerical stability, predictions use log scores:

$$\hat y=\arg\max_k\left[\log\pi_k+\sum_j\left(-\frac12\log(2\pi\sigma_{jk}^2)-\frac{(x_j-\mu_{jk})^2}{2\sigma_{jk}^2}\right)\right].$$

### Multinomial Naive Bayes and additive smoothing

Let $n_{kv}$ be the count of vocabulary term $v$ in documents belonging to class $k$, and let $V$ be vocabulary size. The MLE probability is:

$$\hat\theta_{kv}^{\mathrm{MLE}}=\frac{n_{kv}}{\sum_{u=1}^{V}n_{ku}}.$$

For a bag-of-words document $x=(x_1,\ldots,x_V)$, the implemented comparison score omits the shared multinomial coefficient and (effectively equal) class-prior term:

$$s_k(x)=\sum_{v=1}^{V}x_v\log\theta_{kv}.$$

To reduce zero-probability problems, the implementation also uses additive smoothing:

$$\hat\theta_{kv}=\frac{n_{kv}+\beta}{\sum_{u=1}^{V}n_{ku}+\beta V}.$$

**Terminology note:** the notebook calls this function `train_multinomial_nb_map`. Its formula is standard additive/Lidstone smoothing; it equals a Dirichlet **posterior mean** for concentration $\alpha=\beta$, or a **MAP estimate** for symmetric Dirichlet parameters $\alpha=\beta+1$ (where defined). The parameterization matters when describing it mathematically.

### Evaluation

$$\mathrm{Accuracy}=\frac{\#\{i:\hat y_i=y_i\}}{N},\qquad\mathrm{Precision}=\frac{TP}{TP+FP},\qquad\mathrm{Recall}=\frac{TP}{TP+FN}.$$

The Gaussian Naive Bayes evaluator returns zero for precision or recall when its denominator is zero.

## Results

These values are transcribed from the **saved outputs in the supplied notebook**; they have **not been independently rerun**, because the external coursework data files were not provided.

| Experiment | Saved result |
|---|---:|
| Gaussian Naive Bayes (numerical features) | **83.49% test accuracy** |
| Gaussian NB: class 1 precision / recall | 95.12% / 97.50% |
| Gaussian NB: class 5 precision / recall | 87.50% / 77.78% |
| Sports text NB (unsmoothed MLE) | **86.00% test accuracy** |
| Sports text NB (smoothing $\beta\approx 0.263$–$0.295$ in the tested sweep) | **89.00% test accuracy** |
| Sports text NB (50 training examples/class, $\beta=0.1$) | 87.00% test accuracy |

The saved experiments suggest that moderate smoothing can improve accuracy in this particular test set. The training-set-size experiment varies from 1 to 50 observations per class; results are not strictly monotonic. These are exploratory results on a fixed split, not a cross-validated generalization estimate.

## Getting started

### Requirements

- Python 3.9+ (suggested; original environment/version not recorded)
- NumPy, Matplotlib, SciPy, scikit-learn, Jupyter

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
jupyter notebook ML_annotated.ipynb
```

Run notebook cells **in order**. The Iris portion relies on scikit-learn's bundled dataset and does not need an external download. The Gaussian NB and sports-text sections require additional data files, described below.

## Dataset files

Put the following files in the **same directory as the notebook**, preserving their names exactly:

```text
ml-github-ready/
├── ML_annotated.ipynb
├── README.md
├── requirements.txt
├── .gitignore
├── guassianBayes.mat             # required; not included
├── baseball_train_set.csv       # required; not included
├── baseball_test_set.csv        # required; not included
├── hockey_train_set.csv         # required; not included
└── hockey_test_set.csv          # required; not included
```

The name **`guassianBayes.mat`** is spelled this way in the original code; changing the filename without updating the code will cause an error. MATLAB data is expected to contain `xTrain`, `yTrain`, `xTest`, and `yTest`. The CSVs must contain numeric document-term count matrices with matching vocabulary columns. Neither their provenance nor their redistribution permissions were supplied; only upload them if sharing is permitted.

## Implementation notes and limitations

- **Notebook preservation:** descriptive Markdown headings have been added, and the original executable cell code, saved figures, and saved console outputs have been retained. The notebook has **not been refactored or independently executed**.
- **Shared global variables:** some symbols such as `X`, `y`, and `theta` are reused. Execute from top to bottom or restart the kernel before rerunning.
- **Numerical edge cases:** matrix inversions require invertible covariance matrices; zero Gaussian variance can lead to undefined log likelihoods. These cases are not handled in the original implementation.
- **Plotting details:** the prior comparison routine produces two separate figures rather than true side-by-side subplots; the smoothing sweep includes $\beta=0$ on a logarithmic x-axis, which cannot display zero.
- **Smoothing and priors:** the text classifier uses class log scores without an explicit class-prior term, consistent with equal class sizes in the constructed splits. `safe_log` adds `1e-12` before logarithms; this is a computational safeguard, distinct from probabilistic additive smoothing.
- **Evaluation:** some smoothing parameters were compared on the test set itself; reported best values therefore should not be interpreted as untouched holdout estimates. Use a separate validation set for formal model selection.
- **Dataset availability:** the supplied upload contained only the notebook. Saved non-Iris results are visible, but rerunning those sections requires the missing data files.

## References

- [scikit-learn: Iris dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html)
- [scikit-learn: PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
- [GitHub: Mathematical expressions in Markdown](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/writing-mathematical-expressions)

---

*Originally developed as a machine learning course project; the repository version adds explanatory navigation and reproducibility notes.*
