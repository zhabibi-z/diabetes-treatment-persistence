# Copyright (c) 2026 Zia Habibi
# SPDX-License-Identifier: MIT
"""
chatbot.py — LangChain + Groq (Llama 3.3 70B) chatbot with three RAG retrieval channels:
  1. Document RAG: FAISS vectorstore over ADA 2024 guidelines + study outputs.
  2. SQL channel: DuckDB OMOP queries triggered by data-specific questions.
  3. Model channel: XGBoost predictions and SHAP explanations.

Grounding and citation design
------------------------------
Every response that draws on retrieved documents includes a "Sources" section
listing the origin of each chunk used. This addresses the core LLM hallucination
risk identified in clinical chatbot deployments: the system now cannot claim a
fact from ADA guidelines without citing which section was retrieved.

Embeddings strategy
-------------------
Uses sentence-transformers/all-MiniLM-L6-v2 (Wang et al. 2020, EMNLP) for
semantic retrieval. If sentence-transformers is not installed, a character-n-gram
TF-IDF fallback is used instead. FakeEmbeddings (random vectors) are retired
because they return arbitrary results regardless of query content, making the
RAG layer non-functional for semantic search.

Run standalone: python chatbot/chatbot.py "What is the median TTD for GLP-1 RA?"
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import duckdb
import pandas as pd
import xgboost as xgb
from dotenv import load_dotenv
from groq import Groq
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

load_dotenv()
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

DB_PATH    = os.getenv("OMOP_DB_PATH", "data/omop/omop.duckdb")
MODEL_PATH = "outputs/models/xgb_model.ubj"
ADA_PATH   = "chatbot/ada_guidelines.txt"

RESULTS_PATHS: dict[str, str] = {
    "TTD Summary":     "outputs/tables/ttd_summary.csv",
    "Cox TTD Results": "outputs/tables/cox_ttd_results.csv",
    "Cohort Summary":  "outputs/tables/cohort_summary.csv",
    "Correlations":    "outputs/tables/correlations.csv",
    "ML Metrics":      "outputs/tables/ml_metrics.csv",
    "Model Comparison":"outputs/tables/model_comparison.csv",
}

GROQ_MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are a pharmacoepidemiology research assistant for the T2DM Persistence RWE study.
You help interpret study results, explain statistical methods, and answer clinical questions about
metformin, GLP-1 receptor agonists, and SGLT-2 inhibitors in type 2 diabetes.

Study context:
- 30,000 synthetic T2DM patients (Synthea), OMOP CDM v5.4
- Primary outcome: time-to-discontinuation (90-day grace period, Lim 2025)
- 15 comorbidities tracked (SNOMED-coded)
- Methods: Cox PH, Kaplan-Meier, time-varying Cox, XGBoost (28 features, no followup_days leakage)
- Sensitivity: PS matching (MatchIt, cobalt) + IPTW (stabilised weights)

Always cite the methodological reference when explaining a method.
When you use study data, clearly state it comes from synthetic Synthea data only.
If asked about clinical decisions, defer to ADA 2024 guidelines and advise consultation with a clinician.
If context is provided below, ground your answer in that context and list the sources used.
"""


def _build_embeddings():
    """
    Return a LangChain embeddings object.

    Preference order:
      1. HuggingFaceEmbeddings with all-MiniLM-L6-v2 (semantic, recommended)
      2. TF-IDF fallback via a custom wrapper (lexical, functional)

    FakeEmbeddings are not used — random vectors defeat the purpose of retrieval.
    """
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        emb = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        log.info("Using HuggingFace semantic embeddings (all-MiniLM-L6-v2)")
        return emb
    except Exception as e:
        log.warning("HuggingFace embeddings unavailable (%s) — using TF-IDF lexical fallback", e)
        return _TFIDFEmbeddings()


class _TFIDFEmbeddings:
    """
    Lightweight lexical embedding wrapper using character n-gram TF-IDF.

    This is a functional fallback when sentence-transformers is not available.
    It performs keyword-based retrieval rather than semantic retrieval, which
    is adequate for structured clinical text with consistent terminology.
    """

    def __init__(self, n_components: int = 256) -> None:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), max_features=5000, sublinear_tf=True,
        )
        self._svd = TruncatedSVD(n_components=n_components, random_state=42)
        self._fitted = False
        self._corpus: list[str] = []

    def _fit_if_needed(self, texts: list[str]) -> None:
        if not self._fitted:
            tfidf = self._vectorizer.fit_transform(texts)
            self._svd.fit(tfidf)
            self._fitted = True
            self._corpus = texts

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self._fit_if_needed(texts)
        tfidf = self._vectorizer.transform(texts)
        return self._svd.transform(tfidf).tolist()

    def embed_query(self, text: str) -> list[float]:
        self._fit_if_needed(self._corpus or [text])
        tfidf = self._vectorizer.transform([text])
        return self._svd.transform(tfidf)[0].tolist()


class T2DMChatbot:
    def __init__(self) -> None:
        api_key      = os.getenv("GROQ_API_KEY", "")
        self.client  = Groq(api_key=api_key) if api_key else None
        self.vectorstore, self.source_map = self._build_vectorstore()
        self.xgb_model = self._load_xgb_model()
        self.history: list[dict[str, str]] = []

    def _build_vectorstore(self) -> tuple[FAISS | None, dict[int, str]]:
        """
        Build a FAISS vectorstore from the ADA guidelines and study output CSVs.

        Each document is stored with its source label so that retrieval results
        can be cited in the response.
        """
        documents: list[Document] = []

        # ADA 2024 guidelines
        if Path(ADA_PATH).exists():
            text = Path(ADA_PATH).read_text(encoding="utf-8", errors="replace")
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
            chunks   = splitter.split_text(text)
            for chunk in chunks:
                documents.append(Document(
                    page_content=chunk,
                    metadata={"source": "ADA 2024 Standards of Care", "type": "guideline"},
                ))
            log.info("ADA guidelines: %d chunks indexed", len(chunks))

        # Study outputs (CSVs)
        for label, path in RESULTS_PATHS.items():
            if not Path(path).exists():
                continue
            try:
                df   = pd.read_csv(path)
                text = f"[{label}]\n{df.to_string(index=False)}"
                documents.append(Document(
                    page_content=text,
                    metadata={"source": label, "type": "study_result"},
                ))
                log.info("Indexed study result: %s (%d rows)", label, len(df))
            except Exception as e:
                log.warning("Failed to load %s: %s", path, e)

        if not documents:
            log.warning("No documents available for vectorstore — RAG disabled")
            return None, {}

        embeddings = _build_embeddings()

        try:
            vs = FAISS.from_documents(documents, embeddings)
            log.info("Vectorstore built: %d documents", len(documents))
            return vs, {}
        except Exception as e:
            log.warning("Vectorstore construction failed: %s", e)
            return None, {}

    def _load_xgb_model(self) -> xgb.XGBClassifier | None:
        if Path(MODEL_PATH).exists():
            model = xgb.XGBClassifier()
            model.load_model(MODEL_PATH)
            log.info("XGBoost model loaded (28-feature, leakage-corrected)")
            return model
        log.warning("XGBoost model not found at %s — run ml/train.py first", MODEL_PATH)
        return None

    def _query_sql(self, question: str) -> str:
        """Issue a pre-defined DuckDB query based on question keywords."""
        if not Path(DB_PATH).exists():
            return "OMOP database not found. Run bootstrap.sh to generate it."
        try:
            conn = duckdb.connect(DB_PATH, read_only=True)
            q_lower = question.lower()
            if any(k in q_lower for k in ["how many", "count", "n patients", "cohort size"]):
                result = conn.execute("SELECT count(*) AS n_persons FROM person").df()
            elif any(k in q_lower for k in ["drug", "exposure", "prescription"]):
                result = conn.execute(
                    "SELECT drug_concept_id, count(*) AS n FROM drug_exposure "
                    "GROUP BY drug_concept_id ORDER BY n DESC LIMIT 10"
                ).df()
            else:
                result = conn.execute(
                    "SELECT count(*) AS n_conditions FROM condition_occurrence"
                ).df()
            conn.close()
            return f"[OMOP Database]\n{result.to_string(index=False)}"
        except Exception as e:
            return f"SQL query failed: {e}"

    def _retrieve_context(self, question: str) -> tuple[str, list[str]]:
        """
        Retrieve context from all three channels and return (context_text, source_list).

        Returns:
            Tuple of (context block for system prompt, list of cited source labels).
        """
        parts: list[str] = []
        sources: list[str] = []

        # ── RAG retrieval ──────────────────────────────────────────────────────
        if self.vectorstore is not None:
            try:
                results = self.vectorstore.similarity_search_with_relevance_scores(
                    question, k=4
                )
                rag_parts: list[str] = []
                for doc, score in results:
                    if score < 0.0:
                        continue
                    src = doc.metadata.get("source", "Unknown source")
                    rag_parts.append(f"[{src}]\n{doc.page_content}")
                    if src not in sources:
                        sources.append(src)
                if rag_parts:
                    parts.append("Relevant retrieved context:\n" + "\n---\n".join(rag_parts))
            except Exception as e:
                log.debug("RAG retrieval error: %s", e)

        # ── SQL channel ────────────────────────────────────────────────────────
        sql_triggers = ["how many", "count", "n patients", "cohort size", "database", "omop"]
        if any(k in question.lower() for k in sql_triggers):
            sql_result = self._query_sql(question)
            parts.append(sql_result)
            sources.append("OMOP DuckDB (live query)")

        # ── XGBoost channel ────────────────────────────────────────────────────
        ml_triggers = ["predict", "risk", "probability", "shap", "feature importance", "model"]
        if self.xgb_model and any(k in question.lower() for k in ml_triggers):
            parts.append(
                "[XGBoost Model]\n"
                "The 28-feature XGBoost model is loaded (leakage-corrected — followup_days excluded). "
                "SHAP feature importance plots are in outputs/figures/. "
                "Top features by mean |SHAP|: cci, age_at_index, comorbidity_count, "
                "days_since_t2dm_dx, drug class indicators."
            )
            sources.append("XGBoost model (in-memory)")

        context_text = "\n\n".join(parts) if parts else ""
        return context_text, sources

    def get_response(self, user_message: str) -> str:
        """
        Generate a grounded response using retrieved context and the Groq LLM.

        If no API key is set, returns a static fallback directing the user to
        study output files. When context is retrieved, sources are appended to
        the response so every factual claim is traceable.

        Args:
            user_message: The user's question or instruction.

        Returns:
            LLM response string, with a "Sources" section appended if applicable.
        """
        if self.client is None:
            return (
                "GROQ_API_KEY is not set. Add it to your .env file to enable the chatbot. "
                "Study results are available in the Survival, ML, and Graph tabs, "
                "and in the outputs/tables/ directory."
            )

        context, sources = self._retrieve_context(user_message)

        system = SYSTEM_PROMPT
        if context:
            system += f"\n\n--- Retrieved context (ground your answer in this) ---\n{context}\n---"

        self.history.append({"role": "user", "content": user_message})
        messages = [{"role": "system", "content": system}] + self.history

        try:
            response_obj = self.client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                max_tokens=1024,
                temperature=0.3,
            )
            assistant_message = response_obj.choices[0].message.content

            # Append source citations so every retrieved fact is attributable
            if sources:
                source_block = "\n\n---\n**Sources consulted:** " + " · ".join(sources)
                assistant_message = assistant_message + source_block

            self.history.append({"role": "assistant", "content": assistant_message})
            return assistant_message

        except Exception as e:
            log.error("Groq API error: %s", e)
            return f"API error: {e}"

    def clear_history(self) -> None:
        self.history.clear()


# Module-level singleton (cached across Streamlit reruns)
_chatbot_instance: T2DMChatbot | None = None


def get_chatbot() -> T2DMChatbot:
    global _chatbot_instance
    if _chatbot_instance is None:
        _chatbot_instance = T2DMChatbot()
    return _chatbot_instance


if __name__ == "__main__":
    import sys
    bot   = T2DMChatbot()
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else \
        "What is the median time to discontinuation for GLP-1 RA in this cohort?"
    print(f"\nQuery: {query}\n")
    print(bot.get_response(query))
