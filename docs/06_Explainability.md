# Explainability

## Approach
A SHAP-lite strategy was adopted to ensure:
- Fast inference
- Business interpretability
- Deployment safety

## Features
- Top 3 contributing factors per prediction
- Plain-English explanations
- Optional feature importance plots

## Rationale
Full SHAP computations were avoided at runtime to maintain stability.
