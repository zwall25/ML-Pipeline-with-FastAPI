# Model Card

This model card documents the RandomForestClassifier model developed for the ML Pipeline with FastAPI project. It describes the model’s purpose, data sources, evaluation, performance, and limitations to ensure clarity and responsible usage.

Reference: https://arxiv.org/pdf/1810.03993.pdf

---

## Model Details

- **Model Type:** RandomForestClassifier  
- **Framework:** scikit-learn  
- **Purpose:** Classifies census records into income categories  
- **Integration:** Exposed through a FastAPI application for real-time inference  
- **Recommended Reproducibility Setting:** `random_state=42`  

This model serves as part of an educational machine learning workflow to demonstrate training, evaluation, and deployment concepts.

---

## Intended Use

The model is intended for **educational and demonstration purposes**, specifically to:

- Illustrate an end-to-end ML pipeline  
- Demonstrate training, model persistence, and inference  
- Show how FastAPI can be used to deploy an ML model  
- Support CI/CD examples through GitHub Actions  

### Primary Intended Users
- Data science students  
- Educators  
- Developers learning ML deployment  

### Primary Intended Uses
- Learning exercises  
- Demonstration of model architecture  
- Evaluation of pipeline functionality  

### Not Intended For
- Production deployments  
- Any high-stakes or regulated decision-making  
- Employment, housing, medical, lending, or other sensitive use cases  

---

## Training Data

- **Dataset:** `census.csv` (derived from the UCI Adult dataset)  
- **Content:** Demographic attributes and income labels  
- **Training Sample Size:** ~80% of the dataset after train/test split  
- **Features Used:**  
  - `workclass`  
  - `education`  
  - `marital-status`  
  - `occupation`  
  - `relationship`  
  - `race`  
  - `sex`  
  - `native-country`  

Categorical features were processed via one-hot encoding. The dataset reflects demographic patterns and biases present in the original UCI Adult dataset.

---

## Evaluation Data

- **Dataset:** Remaining ~20% of `census.csv`  
- **Evaluation Method:** Standard train/test split  
- **Evaluation Size:** ~20% of the dataset  
- **Preprocessing:** Identical transformations as applied during training  

---

## Metrics

The model was evaluated using standard classification metrics on the held-out test set.

| Metric    | Value    |
|-----------|----------|
| Precision | **0.7419** |
| Recall    | **0.6384** |
| F1 Score  | **0.6863** |

---

## Ethical Considerations

- The model may reflect biases present in the original census data, including those related to race, gender, and nationality.  
- Predictions should **not** be used for real-world decision-making without further bias and fairness analysis.  
- Sensitive attributes are present in the data and may influence predictions.  
- The dataset reflects historical societal patterns that may carry inherent inequities.  
- The model does not implement bias mitigation or fairness constraints.  

---

## Caveats and Recommendations

- The model is for demonstration purposes only and is **not suitable for production use**.  
- Further tuning, validation, and fairness analysis are recommended before any real-world deployment.  
- Users should be aware of potential biases and limitations in both the dataset and the model.  
- Model performance may degrade if used on data with different distributions than the training data.  
- Additional feature engineering or hyperparameter tuning could significantly improve performance.  

---
