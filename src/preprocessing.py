"""
Preprocessing partagé entre les notebooks d'entraînement et l'API.

Règle d'or : toute transformation appliquée au texte à l'entraînement
doit être appliquée à l'identique à l'inférence. D'où ce module unique.
"""

import nltk
import pandas as pd
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

TEXT_COL = "Document"
LABEL_COL = "Topic_group"
SEED = 42

# Formules email/politesse identifiées dans l'EDA (absentes de la liste NLTK)
DOMAIN_STOPWORDS = {
    "please", "thanks", "thank", "regards", "hi", "hello", "dear",
    "kind", "best", "sent", "re", "pm", "am",
}


def get_stopwords() -> set:
    """Stopwords NLTK anglais + stopwords spécifiques au domaine."""
    try:
        base = set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        base = set(stopwords.words("english"))
    return base.union(DOMAIN_STOPWORDS)


def clean_for_classical(text: str, stop_words: set) -> str:
    """Version pour TF-IDF / embeddings classiques : stopwords retirés."""
    return " ".join(w for w in str(text).split() if w not in stop_words)


def clean_for_transformer(text: str) -> str:
    """Version pour DistilBERT : texte conservé tel quel (déjà lowercased
    et sans ponctuation en amont dans le dataset Kaggle)."""
    return str(text)


def add_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute text_classical et text_transformer, sans modifier df d'origine."""
    df = df.copy()
    stop_words = get_stopwords()
    df["text_classical"] = df[TEXT_COL].apply(lambda t: clean_for_classical(t, stop_words))
    df["text_transformer"] = df[TEXT_COL].apply(clean_for_transformer)
    return df


def make_splits(df: pd.DataFrame, seed: int = SEED):
    """Split stratifié 70/15/15. Même seed = mêmes splits à chaque exécution."""
    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df[LABEL_COL], random_state=seed
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df[LABEL_COL], random_state=seed
    )
    return train_df, val_df, test_df
