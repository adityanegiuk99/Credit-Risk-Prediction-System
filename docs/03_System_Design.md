# System Design

## Architecture Overview

1. User enters applicant details via Streamlit UI
2. Inputs are passed to a unified sklearn Pipeline
3. Preprocessing and encoding occur inside the pipeline
4. XGBoost model predicts default probability
5. Business logic converts probability into a risk score
6. Results and explanations are displayed

## Design Principles
- No runtime dataset dependency
- No retraining during inference
- Explainability-first
- Streamlit Cloud compatibility
