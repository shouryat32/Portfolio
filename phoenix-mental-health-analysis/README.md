# 🏥 Automated Survey Analysis System for Mental Health Training

[![Project Status](https://img.shields.io/badge/status-production-success)](https://github.com/yourusername/phoenix-dashboard)
[![Python](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/)
[![AI Powered](https://img.shields.io/badge/AI-Gemini%20%2B%20BERT-purple)](https://github.com/yourusername/phoenix-dashboard)

> AI-powered automated survey analysis system using the Kirkpatrick Model to transform mental health training feedback into actionable insights — reducing analysis time by 95%.

**Organization:** Phoenix Australia (Mental Health Non-Profit, University of Melbourne)  
**Duration:** March 2024 - October 2024  
**Role:** ML Engineer & Data Scientist  
**Team:** 5 Graduate Students

> **Note:** This project was developed under NDA. This portfolio showcases the technical implementation, architecture, and code without disclosing proprietary survey content, specific visualizations, or client-sensitive data.

---

## 📋 Quick Links

[Executive Summary](#executive-summary) • [Problem](#problem-statement) • [Solution](#solution-architecture) • [AI/ML](#aiml-implementation) • [Code Examples](#code-implementation) • [Results](#results--impact) • [Tech Stack](#technology-stack)

---

## 🎯 Executive Summary

Built an end-to-end automated survey analysis system processing feedback from **190+ mental health training courses**. Integrates Google Gemini LLM, SBERT embeddings, RoBERTa sentiment analysis, and statistical testing to generate publication-quality PDF reports.

### Impact at a Glance

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Analysis Time | 2-3 hours | 5-10 min | **95% ↓** |
| Classification | Manual | <10ms | **3000x faster** |
| Accuracy | Variable | 87-93% | **Consistent** |
| Cost per Analysis | $80-120 | <$5 | **95% ↓** |
| Annual Capacity | 20-30 courses | 190+ courses | **10x ↑** |

**Key Technologies:** Google Gemini API, SBERT, RoBERTa, scikit-learn, Dash, LaTeX/PyLaTeX, SQLite

![System Architecture](figures/figure1_system_architecture.png)
*Figure 1: End-to-end pipeline from survey ingestion to PDF generation*

---

## 🔍 Problem Statement

Phoenix Australia collects extensive feedback through Qualtrics surveys but faced critical bottlenecks:

**Manual Process:**
- ⏰ 2-3 hours per survey for analysis
- 📝 Inconsistent question classification across analysts  
- 😓 Repetitive work for 190+ courses
- 📉 Limited statistical rigor

**Business Impact:**
- Delayed course improvement insights
- High operational costs
- Unable to scale evaluation
- Lack of professional stakeholder reports

**Requirements:** Automate classification using Kirkpatrick Model (4 levels), handle diverse survey formats, maintain high accuracy, generate board-ready reports.

---

## 🏗️ Solution Architecture

### System Overview

![Data Flow](figures/figure2_data_flow_pipeline.png)
*Figure 2: Data pipeline from Qualtrics to PDF reports*

**4-Layer Architecture:**
1. **UI Layer:** Dash interactive dashboard
2. **Application Layer:** Data processing, visualization, report generation
3. **AI/ML Layer:** Classification, sentiment analysis, statistics
4. **Data Layer:** Qualtrics API, SQLite cache, file storage

### Kirkpatrick Four-Level Model

![Kirkpatrick Model](figures/figure3_kirkpatrick_model.png)
*Figure 3: Evaluation framework guiding question classification*

- **Level 1 (Reaction):** Satisfaction, engagement
- **Level 2 (Learning):** Knowledge/skills gained
- **Level 3 (Behavior):** Workplace application
- **Level 4 (Results):** Organizational impact

---

## 🤖 AI/ML Implementation

### 1. Hierarchical Question Classification

**Challenge:** Classify questions with 59 training examples across semantically similar classes.

![Classification Pipeline](figures/figure4_classification_pipeline.png)
*Figure 4: 3-tier fallback ensuring 100% availability*

**Architecture:**
```
Question → Cache Lookup (10ms) 
         ↓ miss
       → Gemini API (93% acc, 2-5s, $0.01)
         ↓ failure
       → SBERT-KNN (87% acc, 30ms, free)
         ↓
       → Store in DB
```

**Performance:**

| Method | Accuracy | Speed | Cost |
|--------|----------|-------|------|
| **Gemini LLM** | 93% | 2-5s | $0.01 |
| **SBERT-KNN** | 87% | 30ms | Free |
| Semantic-only | 72% | 20ms | Free |

**Cache Performance:** 70-90% hit rate after initial processing → 95% cost reduction

### 2. SBERT-KNN Hybrid Innovation

**Novel Approach:** Combine semantic similarity (60%) + k-nearest neighbors (40%)

![Hybrid Model](figures/figure5_hybrid_model.png)
*Figure 5: Dual-component hybrid architecture*

![KNN Tuning](figures/figure6_knn_optimization.png)
*Figure 6: K=7 achieves optimal 87.6% accuracy*

**Results:**

![Confusion Matrix](figures/figure7_confusion_matrix.png)
*Figure 7: Balanced performance across all classes*

| Class | Precision | Recall | F1 | Samples |
|-------|-----------|--------|----|----|
| Level 1 | 89% | 82% | 0.854 | 50 |
| Level 2 | 79% | 92% | 0.852 | 50 |
| Level 3 | 93% | 82% | 0.875 | 17 |
| Level 4 | 100% | 92% | 0.960 | 13 |
| None | 92% | 90% | 0.911 | 40 |

---

## 💻 Code Implementation

### 1. Intelligent Classification with Hierarchical Fallback

```python
import hashlib
import sqlite3
from typing import Dict, Optional, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import google.generativeai as genai

class IntelligentQuestionClassifier:
    """
    3-tier hierarchical classifier with automatic fallback:
    1. Cache lookup (SQLite) - <10ms
    2. Gemini LLM - 93% accuracy, 2-5s
    3. SBERT-KNN Hybrid - 87% accuracy, 30ms
    """
    
    def __init__(self, db_path: str, gemini_api_key: str):
        # Database for caching
        self.db_conn = sqlite3.connect(db_path)
        self._init_database()
        
        # Gemini API setup
        genai.configure(api_key=gemini_api_key)
        self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        
        # SBERT for embeddings (fallback)
        self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # KNN classifier (trained on historical data)
        self.knn_model = None
        self._load_knn_model()
        
        # Category descriptions for semantic matching
        self.category_descriptions = {
            'Level 1 - Reaction': 'satisfaction engagement enjoyment training experience feelings',
            'Level 2 - Learning': 'knowledge gained skills acquired learning outcomes understanding',
            'Level 3 - Behavior': 'application workplace implementation job behavior changes',
            'Level 4 - Results': 'business impact organizational outcomes ROI performance improvement',
            'None': 'demographic information administrative data'
        }
        self.category_embeddings = self._compute_category_embeddings()
    
    def _init_database(self):
        """Initialize SQLite cache with hash-based indexing"""
        cursor = self.db_conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS classifications (
                question_hash TEXT PRIMARY KEY,
                question_text TEXT,
                kirkpatrick_level TEXT,
                confidence REAL,
                method TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_hash ON classifications(question_hash)')
        self.db_conn.commit()
    
    def _compute_hash(self, text: str) -> str:
        """Content-addressable hashing with normalization"""
        normalized = ' '.join(text.lower().strip().split())
        return hashlib.sha256(normalized.encode()).hexdigest()
    
    def classify_question(self, question_text: str) -> Dict:
        """
        Main classification method with intelligent fallback
        
        Returns:
            {
                'level': str,
                'confidence': float,
                'method': str ('cache', 'gemini', 'sbert_knn')
            }
        """
        # Tier 1: Cache lookup (exact match)
        question_hash = self._compute_hash(question_text)
        cached = self._get_from_cache(question_hash)
        
        if cached and cached['confidence'] > 0.85:
            return {
                'level': cached['level'],
                'confidence': cached['confidence'],
                'method': 'cache'
            }
        
        # Tier 2: Semantic similarity cache (near matches)
        similar = self._find_similar_cached(question_text, threshold=0.95)
        if similar:
            return {
                'level': similar['level'],
                'confidence': similar['confidence'],
                'method': 'semantic_cache'
            }
        
        # Tier 3: Gemini LLM (primary classifier)
        try:
            result = self._classify_with_gemini(question_text)
            self._store_in_cache(question_hash, question_text, result, method='gemini')
            return result
        
        except Exception as e:
            print(f"Gemini API failed: {e}. Using fallback classifier...")
            
            # Tier 4: SBERT-KNN fallback
            result = self._classify_with_sbert_knn(question_text)
            self._store_in_cache(question_hash, question_text, result, method='sbert_knn')
            return result
    
    def _get_from_cache(self, question_hash: str) -> Optional[Dict]:
        """Retrieve exact match from cache"""
        cursor = self.db_conn.cursor()
        cursor.execute(
            'SELECT kirkpatrick_level, confidence FROM classifications WHERE question_hash = ?',
            (question_hash,)
        )
        row = cursor.fetchone()
        return {'level': row[0], 'confidence': row[1]} if row else None
    
    def _find_similar_cached(self, question_text: str, threshold: float = 0.95) -> Optional[Dict]:
        """Find semantically similar questions in cache using cosine similarity"""
        embedding = self.sbert_model.encode(question_text)
        
        cursor = self.db_conn.cursor()
        cursor.execute('SELECT question_text, kirkpatrick_level, confidence FROM classifications')
        
        max_similarity = 0
        best_match = None
        
        for row in cursor.fetchall():
            cached_embedding = self.sbert_model.encode(row[0])
            similarity = np.dot(embedding, cached_embedding) / (
                np.linalg.norm(embedding) * np.linalg.norm(cached_embedding)
            )
            
            if similarity > max_similarity and similarity >= threshold:
                max_similarity = similarity
                best_match = {'level': row[1], 'confidence': row[2]}
        
        return best_match
    
    def _classify_with_gemini(self, question_text: str) -> Dict:
        """Classify using Google Gemini LLM with engineered prompt"""
        prompt = f"""Classify this survey question into one Kirkpatrick evaluation level.

Levels:
- Level 1 (Reaction): Satisfaction, engagement, course relevance
- Level 2 (Learning): Knowledge gained, skills acquired
- Level 3 (Behavior): Workplace application, behavior change
- Level 4 (Results): Organizational impact, business outcomes
- None: Demographics, administrative data

Question: "{question_text}"

Respond with ONLY the level (e.g., "Level 1 - Reaction" or "None") and confidence 0-1."""

        response = self.gemini_model.generate_content(prompt)
        
        # Parse response
        text = response.text.strip()
        
        # Extract level and confidence
        if 'Level 1' in text or 'Reaction' in text:
            level = 'Level 1 - Reaction'
        elif 'Level 2' in text or 'Learning' in text:
            level = 'Level 2 - Learning'
        elif 'Level 3' in text or 'Behavior' in text:
            level = 'Level 3 - Behavior'
        elif 'Level 4' in text or 'Results' in text:
            level = 'Level 4 - Results'
        else:
            level = 'None'
        
        # Extract confidence (default to 0.9 for LLM)
        confidence = 0.9
        
        return {'level': level, 'confidence': confidence, 'method': 'gemini'}
    
    def _compute_category_embeddings(self) -> np.ndarray:
        """Pre-compute embeddings for category descriptions"""
        descriptions = list(self.category_descriptions.values())
        return self.sbert_model.encode(descriptions)
    
    def _classify_with_sbert_knn(self, question_text: str) -> Dict:
        """
        Hybrid SBERT-KNN classifier (fallback):
        - 60% semantic similarity to category descriptions
        - 40% KNN from historical classifications
        """
        # Generate question embedding
        embedding = self.sbert_model.encode(question_text).reshape(1, -1)
        
        # Component 1: Semantic similarity (60% weight)
        semantic_scores = np.zeros(len(self.category_descriptions))
        for i, cat_embedding in enumerate(self.category_embeddings):
            similarity = np.dot(embedding[0], cat_embedding) / (
                np.linalg.norm(embedding[0]) * np.linalg.norm(cat_embedding)
            )
            semantic_scores[i] = similarity
        
        # Normalize to probabilities
        semantic_scores = np.exp(semantic_scores) / np.sum(np.exp(semantic_scores))
        
        # Component 2: KNN (40% weight)
        if self.knn_model is not None:
            knn_probs = self.knn_model.predict_proba(embedding)[0]
        else:
            knn_probs = np.zeros(len(self.category_descriptions))
        
        # Weighted combination
        final_scores = 0.6 * semantic_scores + 0.4 * knn_probs
        
        # Get prediction
        classes = list(self.category_descriptions.keys())
        predicted_idx = np.argmax(final_scores)
        predicted_class = classes[predicted_idx]
        confidence = final_scores[predicted_idx]
        
        return {
            'level': predicted_class,
            'confidence': float(confidence),
            'method': 'sbert_knn'
        }
    
    def _load_knn_model(self):
        """Load or train KNN model from cached classifications"""
        cursor = self.db_conn.cursor()
        cursor.execute('SELECT question_text, kirkpatrick_level FROM classifications')
        rows = cursor.fetchall()
        
        if len(rows) < 10:  # Not enough data
            return
        
        # Generate embeddings for all cached questions
        texts = [row[0] for row in rows]
        labels = [row[1] for row in rows]
        
        embeddings = self.sbert_model.encode(texts)
        
        # Train KNN with optimal hyperparameters
        self.knn_model = KNeighborsClassifier(
            n_neighbors=min(7, len(rows) // 2),  # K=7 optimal
            weights='distance',  # Inverse distance weighting
            metric='cosine'
        )
        self.knn_model.fit(embeddings, labels)
    
    def _store_in_cache(self, question_hash: str, question_text: str, 
                        result: Dict, method: str):
        """Store classification in cache for future reuse"""
        cursor = self.db_conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO classifications 
            (question_hash, question_text, kirkpatrick_level, confidence, method)
            VALUES (?, ?, ?, ?, ?)
        ''', (question_hash, question_text, result['level'], result['confidence'], method))
        self.db_conn.commit()
```

### 2. Advanced Sentiment Analysis Pipeline

```python
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
import torch

class EnhancedFeedbackAnalyzer:
    """
    Multi-stage NLP pipeline for qualitative feedback:
    1. Sentiment classification (RoBERTa)
    2. Theme extraction (TF-IDF + Clustering)
    3. Insights generation (automated)
    """
    
    # Global model cache for 10-50x speedup
    _models = {}
    
    @classmethod
    def get_model(cls, model_name: str, loader_func):
        """Singleton pattern for model caching"""
        if model_name not in cls._models:
            print(f"Loading {model_name}... (first time only)")
            cls._models[model_name] = loader_func()
        return cls._models[model_name]
    
    def __init__(self):
        # RoBERTa for sentiment (state-of-the-art)
        self.sentiment_model = self.get_model(
            'roberta-sentiment',
            lambda: pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment",
                device=0 if torch.cuda.is_available() else -1
            )
        )
        
        # SBERT for semantic embeddings
        self.embedding_model = self.get_model(
            'sbert',
            lambda: SentenceTransformer('all-MiniLM-L6-v2')
        )
        
        # TF-IDF for keyword extraction
        self.vectorizer = TfidfVectorizer(
            max_features=100,
            stop_words='english',
            ngram_range=(1, 3),
            min_df=2
        )
    
    def analyze_feedback_batch(self, responses: list) -> dict:
        """
        Batch analysis for efficiency (100 responses in ~3 seconds)
        
        Args:
            responses: List of text responses
            
        Returns:
            {
                'sentiments': List of sentiment labels,
                'themes': Dict of themes with keywords,
                'insights': Dict with strengths/concerns,
                'distribution': Sentiment percentages
            }
        """
        if not responses:
            return {'error': 'No responses provided'}
        
        # 1. Sentiment classification (batch processing)
        print(f"Analyzing sentiment for {len(responses)} responses...")
        sentiments = self.sentiment_model(responses, batch_size=32)
        
        # 2. Generate embeddings for clustering
        embeddings = self.embedding_model.encode(
            responses,
            batch_size=32,
            show_progress_bar=False
        )
        
        # 3. Extract themes via clustering
        themes = self._extract_themes(embeddings, responses, sentiments)
        
        # 4. Generate actionable insights
        insights = self._generate_insights(responses, sentiments, themes)
        
        # 5. Calculate distribution
        sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
        for s in sentiments:
            label = s['label'].lower()
            if 'pos' in label:
                sentiment_counts['positive'] += 1
            elif 'neg' in label:
                sentiment_counts['negative'] += 1
            else:
                sentiment_counts['neutral'] += 1
        
        total = len(sentiments)
        distribution = {
            k: round(v / total * 100, 1) for k, v in sentiment_counts.items()
        }
        
        return {
            'sentiments': sentiments,
            'themes': themes,
            'insights': insights,
            'distribution': distribution
        }
    
    def _extract_themes(self, embeddings, texts, sentiments):
        """
        Hierarchical clustering to identify thematic categories
        
        Uses DBSCAN for automatic cluster detection:
        - eps=0.3: Maximum distance between neighbors
        - min_samples=3: Minimum cluster size
        - metric='cosine': Semantic similarity
        """
        clusterer = DBSCAN(eps=0.3, min_samples=3, metric='cosine')
        clusters = clusterer.fit_predict(embeddings)
        
        themes = {}
        for cluster_id in set(clusters):
            if cluster_id == -1:  # Noise cluster
                continue
            
            # Get texts in this cluster
            cluster_indices = [i for i, c in enumerate(clusters) if c == cluster_id]
            cluster_texts = [texts[i] for i in cluster_indices]
            
            # Extract keywords using TF-IDF
            try:
                tfidf_matrix = self.vectorizer.fit_transform(cluster_texts)
                feature_names = self.vectorizer.get_feature_names_out()
                
                # Get top 5 keywords
                scores = tfidf_matrix.sum(axis=0).A1
                top_indices = scores.argsort()[-5:][::-1]
                keywords = [feature_names[i] for i in top_indices]
                
                themes[f"Theme {cluster_id + 1}"] = {
                    'keywords': keywords,
                    'count': len(cluster_texts),
                    'sample_quotes': cluster_texts[:3]  # Representative samples
                }
            except:
                pass
        
        return themes
    
    def _generate_insights(self, responses, sentiments, themes):
        """
        Automated insight generation:
        - Top strengths (positive themes)
        - Main concerns (negative themes)
        - Representative quotes
        """
        # Separate by sentiment
        positive_responses = []
        negative_responses = []
        
        for i, sent in enumerate(sentiments):
            if 'pos' in sent['label'].lower():
                positive_responses.append(responses[i])
            elif 'neg' in sent['label'].lower():
                negative_responses.append(responses[i])
        
        # Generate insights
        insights = {
            'top_strengths': positive_responses[:5] if positive_responses else [],
            'main_concerns': negative_responses[:5] if negative_responses else [],
            'positive_count': len(positive_responses),
            'negative_count': len(negative_responses),
            'theme_summary': list(themes.keys())
        }
        
        return insights
```

### 3. Statistical Analysis with Proper Safeguards

```python
from scipy import stats
import numpy as np
from statsmodels.stats.power import TTestPower

def analyze_learning_outcomes(pre_scores: np.ndarray, post_scores: np.ndarray) -> dict:
    """
    Rigorous statistical analysis of pre/post training effectiveness
    
    Performs:
    - Paired t-test for significance
    - Cohen's d for effect size
    - 95% confidence intervals
    - Power analysis (Type II error protection)
    - Sample size warnings
    
    Args:
        pre_scores: Pre-training scores (numpy array)
        post_scores: Post-training scores (numpy array)
        
    Returns:
        dict with statistical results and interpretation
    """
    n = len(pre_scores)
    differences = post_scores - pre_scores
    
    # Paired t-test
    t_statistic, p_value = stats.ttest_rel(pre_scores, post_scores)
    
    # Effect size (Cohen's d for paired samples)
    cohens_d = np.mean(differences) / np.std(differences, ddof=1)
    
    # 95% Confidence Interval
    ci_95 = stats.t.interval(
        confidence=0.95,
        df=n-1,
        loc=np.mean(differences),
        scale=stats.sem(differences)
    )
    
    # Statistical power analysis
    power_analysis = TTestPower()
    power = power_analysis.solve_power(
        effect_size=abs(cohens_d),
        nobs=n,
        alpha=0.05,
        alternative='two-sided'
    )
    
    # Significance interpretation
    if p_value < 0.001:
        significance = "***"
        interpretation = "Highly significant"
    elif p_value < 0.01:
        significance = "**"
        interpretation = "Very significant"
    elif p_value < 0.05:
        significance = "*"
        interpretation = "Significant"
    else:
        significance = "ns"
        interpretation = "Not significant"
    
    # Effect size interpretation
    abs_d = abs(cohens_d)
    if abs_d > 0.8:
        effect_interpretation = "Large effect"
    elif abs_d > 0.5:
        effect_interpretation = "Medium effect"
    elif abs_d > 0.2:
        effect_interpretation = "Small effect"
    else:
        effect_interpretation = "Negligible effect"
    
    # Quality warnings
    warnings = []
    if n < 30:
        warnings.append(f"Small sample size (n={n}): interpret with caution")
    if power < 0.8:
        warnings.append(f"Low statistical power ({power:.2f}): may miss real effects")
    if p_value > 0.05 and abs_d > 0.5:
        warnings.append("Non-significant but medium-large effect: consider Type II error")
    
    return {
        'sample_size': n,
        'pre_mean': float(np.mean(pre_scores)),
        'post_mean': float(np.mean(post_scores)),
        'mean_change': float(np.mean(differences)),
        't_statistic': float(t_statistic),
        'p_value': float(p_value),
        'cohens_d': float(cohens_d),
        'confidence_interval': (float(ci_95[0]), float(ci_95[1])),
        'significance': significance,
        'interpretation': interpretation,
        'effect_size': effect_interpretation,
        'statistical_power': float(power),
        'warnings': warnings,
        'conclusion': (
            f"Training {'significantly' if p_value < 0.05 else 'did not significantly'} "
            f"improve scores (M_change={np.mean(differences):.2f}, "
            f"t({n-1})={t_statistic:.2f}, p={p_value:.4f}, d={cohens_d:.2f})"
        )
    }


def calculate_nps(recommendation_scores: list) -> dict:
    """
    Calculate Net Promoter Score (industry-standard metric)
    
    NPS = % Promoters (9-10) - % Detractors (0-6)
    Range: -100 (all detractors) to +100 (all promoters)
    
    Benchmarks:
    - >70: Excellent (World-Class)
    - 30-70: Great (Industry Leader)
    - 0-30: Good (Above Average)
    - <0: Needs Improvement
    """
    n = len(recommendation_scores)
    
    # Categorize respondents
    promoters = sum(1 for r in recommendation_scores if r >= 9)
    passives = sum(1 for r in recommendation_scores if 7 <= r <= 8)
    detractors = sum(1 for r in recommendation_scores if r <= 6)
    
    # Calculate percentages
    promoters_pct = (promoters / n) * 100
    passives_pct = (passives / n) * 100
    detractors_pct = (detractors / n) * 100
    
    # NPS score
    nps_score = promoters_pct - detractors_pct
    
    # Interpretation
    if nps_score > 70:
        category = "Excellent (World-Class)"
    elif nps_score > 30:
        category = "Great (Industry Leader)"
    elif nps_score > 0:
        category = "Good (Above Average)"
    else:
        category = "Needs Improvement"
    
    # Statistical test (is mean significantly above neutral 5.0?)
    t_stat, p_value = stats.ttest_1samp(recommendation_scores, 5.0)
    
    return {
        'nps_score': round(nps_score, 1),
        'promoters_pct': round(promoters_pct, 1),
        'promoters_count': promoters,
        'passives_pct': round(passives_pct, 1),
        'passives_count': passives,
        'detractors_pct': round(detractors_pct, 1),
        'detractors_count': detractors,
        'category': category,
        'sample_size': n,
        'mean_score': float(np.mean(recommendation_scores)),
        'significantly_positive': p_value < 0.05 and np.mean(recommendation_scores) > 5.0
    }
```

### 4. LaTeX PDF Report Generation

```python
from pylatex import Document, Section, Figure, Table, NoEscape
from pylatex.utils import bold
import matplotlib.pyplot as plt

class PDFReportGenerator:
    """
    Professional LaTeX-based PDF report generation
    
    Features:
    - Custom Phoenix Australia branding
    - High-resolution figures (300 DPI)
    - Statistical tables
    - 15-20 page comprehensive reports
    - 30-60 second generation time
    - 99%+ compilation success rate
    """
    
    def __init__(self, branding_config: dict):
        self.brand_colors = branding_config.get('colors', {})
        self.logo_path = branding_config.get('logo', '')
    
    def generate_report(self, analysis_data: dict, output_path: str):
        """Generate complete evaluation report"""
        
        # Initialize LaTeX document
        doc = Document(
            documentclass='article',
            document_options=['11pt', 'a4paper']
        )
        
        # Add packages
        doc.preamble.append(NoEscape(r'\usepackage{graphicx}'))
        doc.preamble.append(NoEscape(r'\usepackage{booktabs}'))
        doc.preamble.append(NoEscape(r'\usepackage{hyperref}'))
        doc.preamble.append(NoEscape(r'\usepackage{geometry}'))
        doc.preamble.append(NoEscape(r'\geometry{margin=1in}'))
        
        # Cover page
        self._add_cover_page(doc, analysis_data['metadata'])
        
        # Executive summary
        self._add_executive_summary(doc, analysis_data['summary'])
        
        # Level 2 analysis (example)
        self._add_level2_section(doc, analysis_data['level2'])
        
        # Generate PDF
        doc.generate_pdf(output_path, clean_tex=False, compiler='pdflatex')
        print(f"Report generated: {output_path}.pdf")
    
    def _add_level2_section(self, doc, level2_data):
        """Add Level 2 (Learning) analysis section"""
        with doc.create(Section('Level 2: Learning Outcomes')):
            doc.append('Statistical analysis of knowledge acquisition.')
            doc.append(NoEscape(r'\vspace{0.3cm}'))
            
            # Create and embed figure
            fig_path = self._create_comparison_chart(level2_data)
            with doc.create(Figure(position='h!')) as fig:
                fig.add_image(fig_path, width=NoEscape(r'0.75\textwidth'))
                fig.add_caption('Pre-training vs Post-training comparison')
            
            # Statistical results table
            with doc.create(Table(position='h!')) as table:
                table.add_caption('Statistical Test Results')
                table.append(NoEscape(r'\begin{tabular}{ll}'))
                table.append(NoEscape(r'\toprule'))
                table.append(NoEscape(r'Metric & Value \\'))
                table.append(NoEscape(r'\midrule'))
                table.append(NoEscape(
                    f"Sample Size & {level2_data['sample_size']} \\\\"
                ))
                table.append(NoEscape(
                    f"t-statistic & {level2_data['t_statistic']:.2f} \\\\"
                ))
                table.append(NoEscape(
                    f"p-value & {level2_data['p_value']:.4f} {level2_data['significance']} \\\\"
                ))
                table.append(NoEscape(
                    f"Cohen's d & {level2_data['cohens_d']:.2f} ({level2_data['effect_size']}) \\\\"
                ))
                table.append(NoEscape(r'\bottomrule'))
                table.append(NoEscape(r'\end{tabular}'))
            
            # Interpretation
            doc.append(NoEscape(r'\subsection*{Interpretation}'))
            doc.append(self._escape_latex(level2_data['conclusion']))
    
    def _escape_latex(self, text: str) -> NoEscape:
        """
        Escape special LaTeX characters (prevents 99% of compilation failures)
        
        Characters: & % $ # _ { } ~ ^ \
        """
        if not text:
            return NoEscape("")
        
        replacements = {
            '&': r'\&',
            '%': r'\%',
            '$': r'\$',
            '#': r'\#',
            '_': r'\_',
            '{': r'\{',
            '}': r'\}',
            '~': r'\textasciitilde{}',
            '^': r'\textasciicircum{}',
            '\\': r'\textbackslash{}'
        }
        
        for char, replacement in replacements.items():
            text = text.replace(char, replacement)
        
        return NoEscape(text)
    
    def _create_comparison_chart(self, data: dict) -> str:
        """Generate high-resolution comparison chart"""
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        
        categories = ['Pre-Training', 'Post-Training']
        means = [data['pre_mean'], data['post_mean']]
        
        ax.bar(categories, means, color=['#3498db', '#2ecc71'])
        ax.set_ylabel('Mean Score')
        ax.set_title('Knowledge Gain Analysis')
        ax.set_ylim(0, 5)
        
        output_path = 'temp_chart.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
```

---

## 📊 Results & Impact

### Quantitative Metrics

**Operational Efficiency:**
- 📊 **190+ courses** now evaluated (vs 20-30 before)
- ⚡ **1,000+ questions** classified and cached
- 💰 **$10K+ annual savings** in analyst time
- 🔄 **99%+ uptime** in production (6 months)

**Model Performance:**
```
Test Set (n=170):
├─ SBERT-KNN: 87% accuracy, 0.872 F1
├─ Gemini LLM: 93% accuracy
├─ Cache hit rate: 70-90%
└─ Classification speed: <10ms (cached), 30ms (new)
```

### Qualitative Impact

**For Staff:** Push-button automation, consistent methodology, unlimited capacity  
**For Stakeholders:** Professional reports, statistical rigor, data-driven decisions  
**Strategic Value:** First automated system, exportable framework, foundation for ML optimization

---

## 🛠️ Technology Stack

**AI/ML:** Google Gemini 1.5, SBERT (all-MiniLM-L6-v2), RoBERTa, scikit-learn  
**Backend:** Python 3.12, Pandas, NumPy, SQLite, pyreadstat  
**Frontend:** Dash, Plotly, Bootstrap Components  
**Reporting:** PyLaTeX, Matplotlib, Seaborn  
**Integration:** Qualtrics API (OAuth), Docker  
**Development:** Git, Jupyter, pytest

---

## 🎓 Skills Demonstrated

**Machine Learning:** LLM integration & prompt engineering, transformer models (BERT, RoBERTa, SBERT), hybrid architecture design, hyperparameter optimization, class imbalance handling

**Engineering:** Multi-tier caching architecture, API rate limiting & cost optimization, database design (SQLite), system architecture & design patterns, Docker containerization

**Data Science:** Statistical hypothesis testing, effect size calculations, NLP (sentiment analysis, theme extraction), data visualization (Plotly, Matplotlib), ETL pipeline development

**Domain:** Kirkpatrick evaluation model, survey design & validation, educational assessment, stakeholder communication

---

## 📚 Project Structure

```
Phoenix-Dashboard/ (35K+ lines)
├── simple_dashboard.py (3K lines)
├── kirkpatrick_classifier.py (400 lines)
├── bert_knn_hybrid_improved.py (600 lines)
├── enhanced_feedback_analyzer.py (2K lines)
├── pdf_report_generator.py (1.2K lines)
├── qualtrics_connection.py (800 lines)
├── level2_analyzer.py (600 lines)
├── classification_database.py (300 lines)
└── [25+ additional modules]
```

---

## 📞 Contact

**Email:** shouryat32@gmail.com  
**LinkedIn:** [Your LinkedIn Profile]  
**GitHub:** [Your GitHub Repository]

**Available:** Technical documentation, architecture diagrams, code walkthrough

---

## 📊 Figure References

**Required Figures for GitHub `/figures` directory:**

1. `figure1_system_architecture.png` - 4-layer system architecture diagram
2. `figure2_data_flow_pipeline.png` - End-to-end data flow from Qualtrics to reports
3. `figure3_kirkpatrick_model.png` - Kirkpatrick evaluation hierarchy pyramid
4. `figure4_classification_pipeline.png` - 3-tier fallback classification flowchart
5. `figure5_hybrid_model.png` - SBERT-KNN dual-component architecture diagram
6. `figure6_knn_optimization.png` - K-value hyperparameter tuning performance graph
7. `figure7_confusion_matrix.png` - Classification performance confusion matrix

> **Note:** Dashboard UI screenshots, survey-specific visualizations, and client data are excluded per NDA requirements. Only technical architecture diagrams and model performance metrics are shown.

---

*Transforming 2-3 hours of manual work into 5-10 minutes of automated analysis — 95% time reduction, 87-93% accuracy, 190+ courses supported, $10K+ annual savings.*

**Key Innovation:** Novel SBERT-KNN hybrid classifier achieving 87% accuracy with zero API costs, combined with intelligent 3-tier caching reducing operational expenses by 95%.
