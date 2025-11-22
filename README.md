Explainable Loan Approval – XGBoost + SHAP + RAG + Streamlit
This project is a compact, end-to-end loan approval engine I built to demonstrate how explainable AI can support business decisions without turning into a black box.
The workflow combines:
XGBoost classifier for approval prediction
SHAP for transparent feature explanations
A simple RAG layer over a lending policy document (no hallucinations)
Streamlit UI for an interactive demo
Everything is local, fast, and easy to audit — designed to show how AI models can justify their decisions clearly and safely.

🔍 What the app does
Takes borrower inputs
Predicts approve / reject with a trained XGBoost model
Generates SHAP values showing the top feature contributors
Retrieves relevant rules from lending_policies.md to justify the decision
Produces a clean, policy-grounded explanation
This mirrors how financial teams review loan decisions in the real world.

📂 Project structure
app/
    main.py            # Streamlit interface
    rag_pipeline.py    # Retrieval + justification logic

data/
    sample_loan_data.csv    # synthetic demo dataset

models/
    xgb_model_demo.pkl      # trained model
    feature_info.pkl        # feature names metadata

policy_docs/
    lending_policies.md     # simplified lending rules

notebooks/
    02_model_training.ipynb # training + model export

▶️ Running the app
pip install -r requirements.txt
streamlit run app/main.py
You’ll get a front-end where you can adjust inputs and see the full reasoning behind each decision.

🧪 Model notes
Model: XGBClassifier
Metrics: small synthetic dataset — demo-level accuracy
Explainability: SHAP (global + local)
RAG: simple keyword retrieval over policy text, no external APIs
This is a demonstration of the workflow, not production scoring performance.

🚀 Future improvements
Add fairness checks
Improve the RAG retrieval scoring
Add LIME for comparison
Package the app in Docker and deploy
Connect to a real credit risk dataset

Why this project matters
Most loan models can predict, but they cannot explain.
This repo shows a practical way to build:
explainable ML
policy-aligned reasoning
auditable decision support
all in one clean workflow.
requirements.txt
README.md
