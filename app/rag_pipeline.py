import pathlib

def load_policies(policy_path: str) -> str:
    """Load the policy text file."""
    path = pathlib.Path(policy_path)
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def simple_keyword_retrieval(policies_text: str, keywords: list[str], top_n: int = 3) -> list[str]:
    """
    Very simple keyword-based retrieval.
    Splits the policy file into paragraphs, scores them by keyword matches,
    returns the top paragraphs.
    """
    if not policies_text:
        return []

    paragraphs = [p.strip() for p in policies_text.split("\n\n") if p.strip()]
    scored = []

    for p in paragraphs:
        score = sum(1 for kw in keywords if kw.lower() in p.lower())
        if score > 0:
            scored.append((score, p))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored[:top_n]]


def build_text_justification(pred_label: int, shap_dict: dict, retrieved_paragraphs: list[str]) -> str:
    """
    Construct a clean explanation combining:
    - model decision
    - SHAP-driven key features
    - retrieved policy rules
    """
    outcome = "approved" if pred_label == 1 else "rejected"

    # Sort features by absolute contribution
    sorted_feats = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    top_feats = [f"{name} ({value:+.2f})" for name, value in sorted_feats[:3]]

    lines = []
    lines.append(f"The application was **{outcome}**.\n")
    lines.append("Top contributing factors (SHAP): " + ", ".join(top_feats) + ".\n")

    if retrieved_paragraphs:
        lines.append("Relevant policy excerpts:")
        for p in retrieved_paragraphs:
            lines.append(f"- {p.strip()}")

    return "\n".join(lines)
