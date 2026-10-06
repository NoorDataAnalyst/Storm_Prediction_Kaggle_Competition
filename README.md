# StormCost: Severe Weather Property Damage Prediction

A machine learning solution for predicting property damage caused by severe weather events using textual weather narratives and structured event metadata.

This repository contains the data processing, feature engineering, text embedding, cross-validation, two-stage modeling, evaluation, and submission pipeline developed for the StormCost Kaggle competition.

---

## 1. Problem Overview

Severe weather events can result in substantial property damage. Predicting the financial impact of these events is challenging because the target variable has two important characteristics:

* A large proportion of events have zero property damage.
* Non-zero damage values are highly skewed, with a small number of events producing extremely large losses.

The objective of the competition is to predict the total property damage, in USD, for each unique event.

### Task

| Component         | Description                                      |
| ----------------- | ------------------------------------------------ |
| Task              | Property damage prediction                       |
| Target            | `predicted_damage`                               |
| Target Type       | Continuous                                       |
| Input             | Weather narratives and structured event metadata |
| Main Challenge    | Zero-inflated and highly skewed target           |
| Validation        | 5-Fold GroupKFold                                |
| Grouping Variable | `episode_group`                                  |

---

## 2. Solution Overview

The solution uses a two-stage modeling approach to separately model:

1. The probability that an event causes property damage.
2. The magnitude of the damage when damage occurs.

Textual weather narratives are converted into dense semantic embeddings using a pretrained Sentence Transformer model. These embeddings are then combined with structured numerical and categorical features.

The overall pipeline is:

```text
Weather Narratives + Event Metadata
                 |
                 v
       Feature Engineering
                 |
                 +----------------------+
                 |                      |
                 v                      v
       Text Embeddings          Tabular Features
                 |                      |
                 +----------+-----------+
                            |
                            v
                    Combined Features
                            |
              +-------------+-------------+
              |                           |
              v                           v
       Classification               Regression
           Head                         Head
              |                           |
              v                           v
     P(Damage > 0)             Log Damage Magnitude
              |                           |
              +-------------+-------------+
                            |
                            v
                 Final Damage Estimate
```

---

## 3. Text Representation

Weather narratives contain information about event characteristics and severity that may not be available in structured metadata.

The solution uses the pretrained Sentence Transformer:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Each narrative is transformed into a dense semantic embedding.

These embeddings provide the model with information contained in the textual descriptions, including:

* Event characteristics
* Severity indicators
* Physical damage descriptions
* Locations and areas affected
* Infrastructure impacts
* Weather-related conditions

The resulting embeddings are combined with structured event features before model training.

---

## 4. Feature Engineering

The structured dataset contains numerical and categorical information associated with each weather event.

The feature engineering pipeline processes variables such as:

* Event type
* Geographic information
* Event duration
* Temporal characteristics
* Other numerical and categorical event metadata

Categorical variables are encoded appropriately, while numerical variables are prepared for model training.

The processed tabular features are concatenated with the text embeddings to create the final modeling matrix.

---

## 5. Two-Stage Modeling

### Stage 1: Damage Occurrence Classification

The first model determines whether an event is likely to have non-zero property damage.

The target is defined as:

```text
Damage > 0
```

This converts the first part of the problem into a binary classification task.

The model estimates:

$$
P(\text{Damage} > 0)
$$

This allows the pipeline to explicitly model the large number of events where no property damage occurs.

---

### Stage 2: Damage Magnitude Regression

The second model predicts the magnitude of property damage for events where damage occurs.

Because the damage distribution is highly right-skewed, the regression target is transformed using:

$$
y_{log} = \log(1 + \text{Damage})
$$

The transformation reduces the influence of extreme values and provides a more stable target for regression.

The predicted value is subsequently converted back to the original damage scale.

---

### Final Prediction

The outputs of the two stages are combined to estimate expected property damage:

**Predicted Damage = P(Damage > 0) × Predicted Conditional Damage**

This formulation allows the final prediction to account for both the probability of damage and its expected magnitude.
This formulation allows the final prediction to account for both the probability of damage and its expected magnitude.

---

## 6. Cross-Validation Strategy

The solution uses **5-Fold GroupKFold Cross-Validation** based on the `episode_group` variable.

### Grouping

Events belonging to the same broader weather episode are kept within the same fold.

This prevents related observations from appearing in both the training and validation sets.

### Benefits

* Reduces potential data leakage.
* Provides a more realistic estimate of generalization.
* Evaluates performance across distinct weather episodes.
* Maintains consistent validation splits between the classification and regression stages.

Both model stages use the same fold assignments so that their out-of-fold predictions remain aligned.

---

## 7. Evaluation

The two stages can be evaluated independently.

### Classification Metrics

The damage occurrence classifier can be evaluated using:

* ROC-AUC
* Precision
* Recall
* F1-score

### Regression Metrics

The damage magnitude model can be evaluated using:

* RMSE
* MAE

The final competition performance is determined by the official competition evaluation metric.

Because the target contains both many zero values and a small number of extreme damage values, evaluation can be particularly sensitive to high-damage observations.

---

## 8. Some Images

<img width="469" height="129" alt="Screenshot 2026-10-06 223548" src="https://github.com/user-attachments/assets/77734a1c-8fab-44fb-9276-692bc18b4e5b" />



<img width="469" height="127" alt="Screenshot 2026-10-06 223530" src="https://github.com/user-attachments/assets/9911738f-3a99-415e-abba-7eb2568c6836" />



<img width="391" height="179" alt="Screenshot 2026-10-06 223457" src="https://github.com/user-attachments/assets/9c0231a4-3a17-43ed-8e62-144ea34be175" />

---

## 9. Submission Format

The final submission follows the required competition schema:

| `sample_id`  | `predicted_damage` |
| ------------ | -----------------: |
| `ENV_000102` |             `0.00` |
| `ENV_000103` |         `14500.50` |
| `ENV_000104` |           `250.00` |

Each row corresponds to an event and contains the predicted property damage in USD.

---

## 10. Getting Started

### Clone the Repository

```bash
git clone https://github.com/your-username/stormcost-damage-prediction.git
cd stormcost-damage-prediction
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Generate Features and Embeddings

```bash
python src/data_processing.py
```

### Train the Models

The training pipeline uses five grouped cross-validation folds:

```bash
python src/train.py --folds 5
```

### Generate Submission

The resulting predictions are saved in:

```text
submissions/stormcost_final_submission.csv
```

---

## 11. Technology Stack

### Programming Language

* Python

### Machine Learning

* Scikit-learn
* LightGBM
* XGBoost
* NumPy
* Pandas

### Natural Language Processing

* Sentence Transformers
* Hugging Face Transformers
* PyTorch
* `all-MiniLM-L6-v2`

### Data Visualization

* Matplotlib
* Seaborn

### Development Environment

* Jupyter Notebook
* Google Colab
* Git
* GitHub

---

## 12. Technical Summary

| Component           | Approach                                 |
| ------------------- | ---------------------------------------- |
| Text Representation | Sentence Transformer embeddings          |
| Embedding Model     | `all-MiniLM-L6-v2`                       |
| Structured Features | Numerical and categorical event metadata |
| Damage Occurrence   | Binary classification                    |
| Damage Magnitude    | Regression using `log1p(damage)`         |
| Validation          | 5-Fold GroupKFold                        |
| Group Variable      | `episode_group`                          |
| Final Prediction    | Probability × Conditional Damage         |
| Output              | Property damage prediction in USD        |

---

## 13. Limitations

Several challenges remain in predicting severe weather property damage:

* Extreme damage values are concentrated in a small number of observations.
* High-damage events can have a disproportionate effect on error metrics.
* Test years may differ from the training period, resulting in distribution shift.
* Frozen sentence embeddings may not capture all domain-specific information contained in weather narratives.
* Grouped cross-validation reduces leakage but cannot completely reproduce unseen future conditions.
* Exact prediction of extreme financial losses remains difficult because of their inherently high variance.

---

## 14. Future Improvements

Potential areas for further development include:

* Fine-tuning a transformer model specifically for weather narratives.
* Testing larger Sentence Transformer models.
* Incorporating additional temporal and geographic features.
* Developing interaction features between text embeddings and structured variables.
* Comparing the two-stage approach with alternative zero-inflated and hurdle models.
* Optimizing the classification component and probability calibration.
* Ensembling multiple gradient boosting models.
* Testing time-based validation to better simulate future distribution shifts.
* Developing specialized approaches for extreme-damage events.

---

## 15. Conclusion

StormCost combines natural language processing, structured feature engineering, grouped cross-validation, and two-stage modeling to address the challenges of severe weather property damage prediction.

The approach separates the problem into two components:

```text
Probability of Damage
          +
Magnitude of Damage
          =
Expected Property Damage
```

This provides a structured framework for modeling a target that contains a large number of zero-damage events alongside highly variable non-zero damage values.

---

## 16. Author

**Noor ul Ain**


---

## 17. License

This project is intended for educational, research, and competition purposes.

Please refer to the original Kaggle competition rules and dataset license before redistributing competition data or derived datasets.
