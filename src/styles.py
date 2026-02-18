CUSTOM_CSS = """
<style>
/* Main header */
.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1.5rem 2rem;
    border-radius: 10px;
    margin-bottom: 1.5rem;
    color: white;
    text-align: center;
}
.main-header h1 {
    margin: 0;
    font-size: 2rem;
    font-weight: 700;
}
.main-header p {
    margin: 0.3rem 0 0 0;
    opacity: 0.9;
    font-size: 0.95rem;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #e2e8f0;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label {
    color: #cbd5e1;
}

/* Status badges */
.status-badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
}
.status-connected {
    background: #065f46;
    color: #6ee7b7;
    border: 1px solid #6ee7b7;
}
.status-disconnected {
    background: #7f1d1d;
    color: #fca5a5;
    border: 1px solid #fca5a5;
}

/* Source cards */
.source-card {
    border-radius: 8px;
    padding: 0.75rem 1rem;
    margin: 0.5rem 0;
    border-left: 4px solid;
}
.source-pdf {
    background: #1e3a5f20;
    border-left-color: #3b82f6;
}
.source-txt {
    background: #14532d20;
    border-left-color: #22c55e;
}
.source-web {
    background: #7c2d1220;
    border-left-color: #f97316;
}
.source-docx {
    background: #1e1a5f20;
    border-left-color: #8b5cf6;
}
.source-csv {
    background: #5f1e3a20;
    border-left-color: #ec4899;
}
.source-card .source-title {
    font-weight: 600;
    font-size: 0.9rem;
    margin-bottom: 0.3rem;
}
.source-card .source-content {
    font-size: 0.8rem;
    opacity: 0.8;
}

/* Metrics panel */
.metrics-panel {
    display: flex;
    gap: 0.75rem;
    margin: 0.75rem 0;
    flex-wrap: wrap;
}
.metric-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 0.5rem 1rem;
    text-align: center;
    flex: 1;
    min-width: 120px;
}
.metric-card .metric-value {
    font-size: 1.3rem;
    font-weight: 700;
    color: #60a5fa;
}
.metric-card .metric-label {
    font-size: 0.75rem;
    color: #94a3b8;
    margin-top: 0.15rem;
}

/* Footer */
.app-footer {
    text-align: center;
    padding: 1.5rem 0;
    margin-top: 2rem;
    border-top: 1px solid #334155;
    color: #64748b;
    font-size: 0.85rem;
}

/* Chat improvements */
.stChatMessage {
    border-radius: 12px !important;
}

/* Evaluation table */
.eval-table {
    width: 100%;
    border-collapse: collapse;
    margin: 1rem 0;
}
.eval-table th {
    background: #1e293b;
    color: #e2e8f0;
    padding: 0.75rem;
    text-align: left;
    border-bottom: 2px solid #3b82f6;
}
.eval-table td {
    padding: 0.6rem 0.75rem;
    border-bottom: 1px solid #334155;
}

/* Score colors */
.score-high { color: #22c55e; font-weight: 700; }
.score-medium { color: #eab308; font-weight: 700; }
.score-low { color: #ef4444; font-weight: 700; }
</style>
"""


def get_source_card_html(source, index):
    """Generate HTML for a styled source card."""
    source_type = source.get("type", "unknown")
    css_class = f"source-{source_type}" if source_type in ("pdf", "txt", "web", "docx", "csv") else "source-txt"

    name = source.get("name", "Unknown")
    page_info = f" (Page {source['page'] + 1})" if "page" in source else ""
    content = source.get("content", "")

    return f"""
    <div class="source-card {css_class}">
        <div class="source-title">Source {index}: {name}{page_info} [{source_type.upper()}]</div>
        <div class="source-content">{content}</div>
    </div>
    """


def get_metrics_html(metrics):
    """Generate HTML for metrics display panel."""
    relevance_score = metrics.get("relevance", {}).get("avg_score", 0)
    if relevance_score >= 0.7:
        score_class = "score-high"
    elif relevance_score >= 0.4:
        score_class = "score-medium"
    else:
        score_class = "score-low"

    return f"""
    <div class="metrics-panel">
        <div class="metric-card">
            <div class="metric-value">{metrics.get('response_time', 0)}s</div>
            <div class="metric-label">Response Time</div>
        </div>
        <div class="metric-card">
            <div class="metric-value {score_class}">{relevance_score:.1%}</div>
            <div class="metric-label">Relevance Score</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{metrics.get('chunks_used', 0)}</div>
            <div class="metric-label">Chunks Used</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{metrics.get('answer_words', 0)}</div>
            <div class="metric-label">Words</div>
        </div>
    </div>
    """
