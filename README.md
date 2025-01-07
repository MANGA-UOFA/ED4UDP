# ED4UDP

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/error-diversity-matters-an-error-resistant/unsupervised-dependency-parsing-on-penn)](https://paperswithcode.com/sota/unsupervised-dependency-parsing-on-penn?p=error-diversity-matters-an-error-resistant)


Official codebase for paper "Error Diversity Matters: An Error-Resistant Ensemble Method for Unsupervised Dependency
Parsing."

## Overview

The repository provides tools to evaluate unsupervised dependency parsing models, aggregate predictions using ensemble techniques, and optimize model selection considering error diversity.

1. Evaluation (`eval.py`): Computes performance metrics such as Corpus F1 and Corpus UAS (Unlabeled Attachment Score).
2. Ensemble Aggregation (`ensemble.py`): Aggregates predictions from multiple models using accuracy-based or unweighted methods.
3. Model Selection (`model_selection.py`): Selects optimal model subsets for ensemble using strategies based on performance and error diversity metrics.

## Install

```
conda create -n ED4UDP python=3.9
conda activate ED4UDP
while read requirement; do pip install $requirement; done < requirements.txt 
```

## Usage

### 1. Evaluation
```
python eval.py --ref path/to/gold_file --pred path/to/prediction_file
```
Outputs CorpusF1 and CorpusUAS metrics.

### 2. Ensemble Aggregation

See `ensemble.py` for the example usage of the `ensemble()` function. Here’s a breakdown of its arguments, their purposes, and how they affect its behavior:

1. `references` (list of lists):
    * A list of prediction files where each file is represented as a list of dependency trees (attachments).
    * Each tree in a list corresponds to the predictions for a particular sentence or data instance.
    * This is the main input to the ensemble function, as it aggregates predictions across all the trees.

2. `agg` (str, default: `acc`):
    * Specifies the aggregation method to use:
    * 'acc': Uses accuracy-based aggregation.
    * 'f1': Uses F1-based aggregation.
  
3. `beta` (float, default: `1`):
    * A parameter for F1-based aggregation (Ignored if `agg == 'acc'`).
    * Adjusts the relative importance of precision and recall in the F1 score.
  
4. `weights` (list of floats, default: `None`):
    * Determines the influence of each model's predictions during aggregation.
    * If `None`, all models are given equal weight.
    * The length of weights must match the number of models in references.

5. `parallel` (bool, default: `True`):
    * Controls whether the ensemble computation is performed in parallel using multiprocessing.
  
6. `return_times` (bool, default: `False`):
    * Determines whether the function returns the computation time for each instance:
         * `True`: Returns a tuple (aggregated_predictions, times) where times is a list of computation times for each instance.
         * `False`: Returns only the aggregated predictions.
    * Cannot be `True` if `parallel=True`, as individual processing times are not tracked in parallel mode.
  
7. `progress_bar` (bool, default: `True`):
    * Controls whether a progress bar is displayed during computation.

### 3. Model Selection

First, configure parameters in `model_selection.py`.
Then,
```
python model_selection.py
```

<a href="https://TheShayegh.github.io/"><img src="https://TheShayegh.github.io/img/favicon.png" style="background-color:red;"/></a>
