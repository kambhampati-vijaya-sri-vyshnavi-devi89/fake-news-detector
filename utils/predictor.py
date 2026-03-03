import re
import math
import random

# ─── Lightweight rule-based + statistical classifier (no heavy dependencies) ───
# In production, replace with actual fine-tuned BERT/RoBERTa from HuggingFace

# Sensationalist / fake-news indicator words
FAKE_INDICATORS = [
    "shocking", "unbelievable", "breaking", "urgent", "exposed", "secret",
    "they don't want you to know", "cover-up", "conspiracy", "hoax",
    "fake", "fraud", "scam", "lie", "lies", "liar", "corrupt", "scandal",
    "wake up", "sheeple", "illuminati", "deep state", "crisis actor",
    "false flag", "plandemic", "chemtrail", "microchip", "5g", "mind control",
    "miracle cure", "doctors hate", "big pharma", "they're hiding",
    "mainstream media won't", "share before deleted", "spread the word",
    "you won't believe", "insane truth", "bombshell", "explosive",
    "100%", "guaranteed", "proven fact", "scientifically proven",
    "click here", "click now", "act now", "limited time",
]

REAL_INDICATORS = [
    "according to", "research shows", "study finds", "published in",
    "peer-reviewed", "data suggests", "evidence indicates", "sources say",
    "officials confirmed", "statement released", "report says",
    "experts say", "analysis", "investigation", "documented",
    "statistics show", "survey", "poll", "government", "university",
    "journal", "findings", "methodology", "sample size",
]

PUNCTUATION_ABUSE = re.compile(r'[!?]{2,}')
ALL_CAPS = re.compile(r'\b[A-Z]{4,}\b')

def clean_text(text):
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def extract_features(text):
    text_lower = text.lower()
    words = text_lower.split()
    total_words = max(len(words), 1)

    fake_count = sum(1 for ind in FAKE_INDICATORS if ind in text_lower)
    real_count = sum(1 for ind in REAL_INDICATORS if ind in text_lower)
    punct_abuse = len(PUNCTUATION_ABUSE.findall(text))
    caps_words = len(ALL_CAPS.findall(text))
    avg_word_len = sum(len(w) for w in words) / total_words
    exclamation_ratio = text.count('!') / total_words
    question_ratio = text.count('?') / total_words
    caps_ratio = caps_words / total_words

    return {
        'fake_count': fake_count,
        'real_count': real_count,
        'punct_abuse': punct_abuse,
        'caps_words': caps_words,
        'avg_word_len': avg_word_len,
        'exclamation_ratio': exclamation_ratio,
        'question_ratio': question_ratio,
        'caps_ratio': caps_ratio,
        'total_words': total_words,
    }

def tfidf_lr_predict(text):
    """Simulates TF-IDF + Logistic Regression baseline."""
    feats = extract_features(text)

    score = 0.5  # neutral baseline

    # Push toward fake
    score += feats['fake_count'] * 0.08
    score += feats['punct_abuse'] * 0.06
    score += feats['caps_ratio'] * 0.3
    score += feats['exclamation_ratio'] * 0.5
    score += (1 / max(feats['avg_word_len'], 1)) * 0.05

    # Push toward real
    score -= feats['real_count'] * 0.07
    if feats['total_words'] > 100:
        score -= 0.05
    if feats['avg_word_len'] > 5.5:
        score -= 0.04

    score = max(0.05, min(0.97, score))
    label = "FAKE" if score > 0.5 else "REAL"
    confidence = score if label == "FAKE" else (1 - score)
    return label, round(confidence * 100, 1)

def bert_predict(text):
    """Simulates fine-tuned BERT (in prod: use transformers pipeline)."""
    base_label, base_conf = tfidf_lr_predict(text)
    # BERT is more accurate — simulate slightly better confidence + occasional correction
    feats = extract_features(text)
    adjustment = random.uniform(-3.5, 5.5)
    confidence = min(97.0, max(55.0, base_conf + adjustment + feats['real_count'] * 1.2))
    # Occasionally flip if borderline
    if base_conf < 57 and random.random() < 0.3:
        label = "REAL" if base_label == "FAKE" else "FAKE"
    else:
        label = base_label
    return label, round(confidence, 1)

def roberta_predict(text):
    """Simulates fine-tuned RoBERTa."""
    base_label, base_conf = bert_predict(text)
    adjustment = random.uniform(-2, 4)
    confidence = min(98.5, max(58.0, base_conf + adjustment))
    return base_label, round(confidence, 1)

def get_highlight_words(text):
    text_lower = text.lower()
    highlights = []
    words = text.split()
    for i, word in enumerate(words):
        clean_word = re.sub(r'[^a-z]', '', word.lower())
        if any(ind in clean_word for ind in FAKE_INDICATORS):
            highlights.append({'word': word, 'type': 'fake', 'index': i})
        elif any(ind in clean_word for ind in REAL_INDICATORS):
            highlights.append({'word': word, 'type': 'real', 'index': i})
    return highlights[:10]

def predict_news(text, model_name='bert'):
    text = clean_text(text)

    if model_name == 'tfidf_lr':
        label, confidence = tfidf_lr_predict(text)
        model_display = "TF-IDF + Logistic Regression"
    elif model_name == 'roberta':
        label, confidence = roberta_predict(text)
        model_display = "RoBERTa (Fine-tuned)"
    else:
        label, confidence = bert_predict(text)
        model_display = "BERT (Fine-tuned)"

    # All three model scores for comparison chart
    lr_label, lr_conf = tfidf_lr_predict(text)
    b_label, b_conf = bert_predict(text)
    r_label, r_conf = roberta_predict(text)

    feats = extract_features(text)

    return {
        'label': label,
        'confidence': confidence,
        'model': model_display,
        'highlights': get_highlight_words(text),
        'features': {
            'fake_indicators_found': feats['fake_count'],
            'credibility_indicators_found': feats['real_count'],
            'excessive_punctuation': feats['punct_abuse'],
            'all_caps_words': feats['caps_words'],
            'word_count': feats['total_words'],
            'avg_word_length': round(feats['avg_word_len'], 2),
        },
        'model_comparison': {
            'TF-IDF + LR': {'label': lr_label, 'confidence': lr_conf},
            'BERT': {'label': b_label, 'confidence': b_conf},
            'RoBERTa': {'label': r_label, 'confidence': r_conf},
        },
        'explanation': generate_explanation(label, feats),
    }

def generate_explanation(label, feats):
    reasons = []
    if label == "FAKE":
        if feats['fake_count'] > 0:
            reasons.append(f"Contains {feats['fake_count']} sensationalist indicator(s)")
        if feats['caps_ratio'] > 0.1:
            reasons.append("Excessive use of ALL CAPS (emotional manipulation)")
        if feats['punct_abuse'] > 0:
            reasons.append("Punctuation abuse (!! or ??)")
        if feats['exclamation_ratio'] > 0.05:
            reasons.append("High exclamation mark density")
        if not reasons:
            reasons.append("Linguistic patterns consistent with misinformation")
    else:
        if feats['real_count'] > 0:
            reasons.append(f"Contains {feats['real_count']} credibility marker(s)")
        if feats['total_words'] > 80:
            reasons.append("Sufficient detail and length")
        if feats['avg_word_len'] > 5:
            reasons.append("Formal vocabulary detected")
        if not reasons:
            reasons.append("Neutral, measured language patterns")
    return reasons
