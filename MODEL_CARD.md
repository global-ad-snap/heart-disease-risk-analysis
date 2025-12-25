# Model Card — Heart Disease Risk Prediction
## Model Name
Heart Disease Risk Prediction Models (Random Forest, LightGBM, XGBoost)

## Model Purpose
To predict the presence of heart disease using structured patient clinical data, supporting research-oriented decision-support prototyping and machine learning benchmarking.

## Training Data
- Dataset: raw_merged_heart_dataset.csv
- Data Type: Structured clinical and demographic variables
- Sample Size: 1,744 patients (after cleaning)
- Target Variable: Binary heart disease indicator (1 = Yes, 0 = No)
- Class Balance: 862 positive cases, 882 negative cases
- Limitations: Retrospective dataset, limited geographic and demographic diversity

## Model Architecture
- Primary Model: Random Forest Classifier
- Secondary Model: LightGBM Classifier
- Additional Models Evaluated: Logistic Regression, KNN, SVM, Naive Bayes, XGBoost
- Frameworks: scikit-learn, XGBoost, LightGBM
- Feature Processing: Missing value imputation, feature scaling (StandardScaler), categorical encoding

## Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1 Score
- ROC–AUC
- Validation: Internal test split only (80/20) 

## Performance Summary
The Random Forest classifier achieved the strongest overall performance with an F1 Score of 0.959 and ROC–AUC of 0.995, indicating strong discriminatory performance on the internal test set.
Tree-based ensemble models consistently outperformed linear and distance-based methods.

## Interpretability
- Feature importance and SHAP analysis were applied
- Most influential predictors:
- Maximum heart rate (thalachh)
- Chest pain type (cp)
- ST depression (oldpeak)
- Age
- Interpretability outputs align with established clinical knowledge

## Limitations
- No external or prospective validation
- Dataset represents a limited population
- Model performance may not generalize across healthcare systems
- Binary classification does not capture disease severity

## Ethical Considerations 
- Risk of misuse without clinical oversight
- Potential bias from historical healthcare data
- Predictions should not replace professional medical judgment

## Intended Use Statement 
These models are intended for research and educational purposes, including portfolio demonstration, and are not approved for clinical diagnosis, treatment decisions, or operational deployment without further validation.