import html
import io
import pickle
import textwrap
from datetime import datetime

import numpy as np
import streamlit as st
import tensorflow as tf

from text_sources import (
    MAX_CSV_BYTES,
    MAX_CSV_ROWS,
    read_texts_from_csv,
    render_batch_results_csv,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="CineSentiment AI",
    page_icon="🎬",
    layout="wide",
)


# ============================================================
# CUSTOM HTML RENDER HELPER
# ============================================================

def render_html(content):
    """
    Render custom HTML without Streamlit treating
    indented HTML as a code block.
    """
    cleaned_content = "\n".join(line.strip() for line in content.splitlines())
    st.markdown(
        cleaned_content,
        unsafe_allow_html=True,
    )


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>

/* ==========================================================
   GLOBAL
   ========================================================== */

.stApp {
    background:
        radial-gradient(
            circle at 5% 10%,
            rgba(124, 58, 237, 0.18),
            transparent 28%
        ),
        radial-gradient(
            circle at 95% 15%,
            rgba(37, 99, 235, 0.13),
            transparent 28%
        ),
        #080b14;

    color: #f8fafc;
}

.main .block-container {
    max-width: 1250px;
    padding-top: 1.5rem;
    padding-bottom: 4rem;
}


/* ==========================================================
   REMOVE STREAMLIT DEFAULT ELEMENTS
   ========================================================== */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    background: transparent !important;
}


/* ==========================================================
   TOP NAVIGATION
   ========================================================== */

.top-nav {
    width: 100%;

    display: flex;
    align-items: center;
    justify-content: space-between;

    padding: 0.8rem 0 1.2rem 0;

    border-bottom: 1px solid rgba(255,255,255,0.08);

    margin-bottom: 1rem;
}

.brand {
    display: flex;
    align-items: center;
    gap: 10px;

    font-size: 1.35rem;
    font-weight: 800;

    color: #f8fafc;
}

.brand-icon {
    font-size: 1.8rem;
}

.brand-subtitle {
    color: #64748b;
    font-size: 0.8rem;
}


/* ==========================================================
   TOP NAVIGATION RADIO
   ========================================================== */

div[role="radiogroup"] {
    display: flex;
    justify-content: center;

    gap: 8px;

    margin-bottom: 1.5rem;
}

div[role="radiogroup"] label {
    background: rgba(15, 23, 42, 0.75);

    border: 1px solid rgba(255,255,255,0.08);

    border-radius: 12px;

    padding: 7px 18px;

    color: #cbd5e1;

    transition:
        transform 0.2s ease,
        border-color 0.2s ease;
}

div[role="radiogroup"] label:hover {
    border-color: rgba(139,92,246,0.55);

    transform: translateY(-1px);
}


/* ==========================================================
   HIDE RADIO CIRCLE
   ========================================================== */

div[role="radiogroup"] label > div:first-child {
    display: none;
}


/* ==========================================================
   HEADINGS
   ========================================================== */

h1 {
    font-size: 3rem !important;
    font-weight: 800 !important;
    letter-spacing: -1px;
}

h2 {
    font-weight: 750 !important;
}

h3 {
    font-weight: 700 !important;
}


/* ==========================================================
   HERO
   ========================================================== */

.hero {
    padding: 3rem;

    border-radius: 28px;

    margin-bottom: 2.2rem;

    background:
        linear-gradient(
            135deg,
            rgba(124, 58, 237, 0.28),
            rgba(37, 99, 235, 0.16)
        );

    border: 1px solid rgba(255,255,255,0.10);

    box-shadow:
        0 25px 60px rgba(0,0,0,0.30);
}

.hero-badge {
    display: inline-block;

    padding: 7px 14px;

    border-radius: 999px;

    background: rgba(124,58,237,0.20);

    border: 1px solid rgba(167,139,250,0.30);

    color: #c4b5fd;

    font-size: 0.82rem;

    font-weight: 700;

    margin-bottom: 1rem;
}

.hero-title {
    font-size: 3.2rem;

    line-height: 1.1;

    font-weight: 850;

    background:
        linear-gradient(
            90deg,
            #ffffff,
            #c4b5fd,
            #93c5fd
        );

    -webkit-background-clip: text;

    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    color: #cbd5e1;

    font-size: 1.1rem;

    line-height: 1.7;

    max-width: 850px;

    margin-top: 1rem;
}


/* ==========================================================
   SECTION HEADERS
   ========================================================== */

.section-label {
    color: #a78bfa;

    text-transform: uppercase;

    letter-spacing: 2px;

    font-size: 0.75rem;

    font-weight: 800;

    margin-bottom: 0.4rem;
}

.section-title {
    font-size: 1.8rem;

    font-weight: 800;

    margin-bottom: 1.2rem;
}


/* ==========================================================
   FEATURE CARDS
   ========================================================== */

.feature-card {
    padding: 1.5rem;

    border-radius: 20px;

    background:
        rgba(15, 23, 42, 0.72);

    border:
        1px solid rgba(255,255,255,0.08);

    min-height: 160px;

    box-shadow:
        0 10px 35px rgba(0,0,0,0.20);

    overflow: hidden;
}

.feature-icon {
    font-size: 2rem;

    margin-bottom: 0.8rem;
}

.feature-title {
    font-weight: 750;

    font-size: 1.05rem;

    margin-bottom: 0.4rem;
}

.feature-text {
    color: #94a3b8;

    font-size: 0.9rem;

    line-height: 1.5;
}


/* ==========================================================
   INFO BOX
   ========================================================== */

.info-box {
    padding: 1.2rem 1.4rem;

    border-radius: 16px;

    background:
        rgba(30,41,59,0.65);

    border:
        1px solid rgba(255,255,255,0.08);

    color: #cbd5e1;

    line-height: 1.6;
}


/* ==========================================================
   TEXT AREA
   ========================================================== */

textarea {
    background:
        #0f172a !important;

    color:
        #f8fafc !important;

    border:
        1px solid rgba(255,255,255,0.10) !important;

    border-radius:
        16px !important;
}

textarea:focus {
    border-color:
        #8b5cf6 !important;

    box-shadow:
        0 0 0 1px #8b5cf6 !important;
}


/* ==========================================================
   BUTTONS
   ========================================================== */

.stButton > button {
    border-radius: 12px;

    font-weight: 700;

    border:
        1px solid rgba(255,255,255,0.10);

    transition:
        transform 0.2s ease,
        box-shadow 0.2s ease;
}

.stButton > button:hover {
    transform:
        translateY(-2px);

    box-shadow:
        0 8px 25px rgba(124,58,237,0.25);
}


/* ==========================================================
   RESULT CARD
   ========================================================== */

.result-card {
    padding: 2rem;

    border-radius: 24px;

    margin-top: 1.5rem;

    background:
        linear-gradient(
            135deg,
            rgba(15,23,42,0.95),
            rgba(30,41,59,0.85)
        );

    border:
        1px solid rgba(255,255,255,0.10);

    box-shadow:
        0 20px 50px rgba(0,0,0,0.25);
}

.positive-result {
    border-left:
        5px solid #22c55e;
}

.negative-result {
    border-left:
        5px solid #ef4444;
}

.result-label {
    color: #94a3b8;

    font-size: 0.8rem;

    text-transform: uppercase;

    letter-spacing: 1.5px;

    font-weight: 700;
}

.result-sentiment {
    font-size: 2.5rem;

    font-weight: 850;

    margin: 0.4rem 0;
}

.positive-text {
    color: #4ade80;
}

.negative-text {
    color: #f87171;
}

.confidence-text {
    color: #cbd5e1;

    font-size: 1rem;
}


/* ==========================================================
   HISTORY
   ========================================================== */

.history-card {
    padding: 1.3rem;

    margin-bottom: 1rem;

    background:
        rgba(15,23,42,0.75);

    border-radius: 16px;

    border:
        1px solid rgba(255,255,255,0.07);
}

.history-review {
    color: #e2e8f0;

    font-size: 0.95rem;

    line-height: 1.6;

    margin-top: 0.5rem;
}

.history-meta {
    color: #94a3b8;

    font-size: 0.8rem;
}


/* ==========================================================
   FILE UPLOADER
   ========================================================== */

[data-testid="stFileUploader"] {
    background:
        rgba(15,23,42,0.60);

    border-radius:
        18px;

    padding:
        0.5rem;
}


/* ==========================================================
   DATAFRAME
   ========================================================== */

[data-testid="stDataFrame"] {
    border-radius: 14px;
    overflow: hidden;
}


/* ==========================================================
   FOOTER
   ========================================================== */

.footer {
    text-align: center;

    color: #64748b;

    font-size: 0.8rem;

    padding-top: 3rem;
}


/* ==========================================================
   DIVIDERS
   ========================================================== */

hr {
    border-color:
        rgba(255,255,255,0.08) !important;
}


/* ==========================================================
   METRICS
   ========================================================== */

[data-testid="stMetric"] {
    background:
        rgba(15,23,42,0.65);

    border:
        1px solid rgba(255,255,255,0.08);

    padding:
        1rem;

    border-radius:
        16px;
}


/* ==========================================================
   RESPONSIVE
   ========================================================== */

@media (max-width: 768px) {

    .hero {
        padding: 2rem;
    }

    .hero-title {
        font-size: 2.3rem;
    }

    .hero-subtitle {
        font-size: 1rem;
    }

    .top-nav {
        flex-direction: column;

        align-items: flex-start;

        gap: 5px;
    }

}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# MODEL UTILITIES
# ============================================================

def pad_sequences(
    sequences,
    maxlen=200,
    padding="post",
    truncating="post",
    value=0,
):
    padded = []

    for seq in sequences:

        if len(seq) > maxlen:

            if truncating == "pre":
                seq = seq[-maxlen:]
            else:
                seq = seq[:maxlen]

        elif len(seq) < maxlen:

            pad_len = maxlen - len(seq)

            pad_array = [value] * pad_len

            if padding == "post":
                seq = list(seq) + pad_array
            else:
                seq = pad_array + list(seq)

        padded.append(seq)

    return np.array(
        padded,
        dtype=np.int32,
    )


def predict_sentiment(
    review,
    tokenizer,
    model,
):
    """
    Predict sentiment using the existing
    LSTM model and tokenizer.
    """

    cleaned_review = review.lower().strip()

    sequence = tokenizer.texts_to_sequences(
        [cleaned_review]
    )

    padded_sequence = pad_sequences(
        sequence,
        maxlen=200,
        padding="pre",
        truncating="pre",
    )

    prediction = model.predict(
        padded_sequence,
        verbose=0,
    )

    score = float(
        prediction[0][0]
    )

    sentiment = (
        "Positive"
        if score > 0.5
        else "Negative"
    )

    confidence = (
        score
        if score > 0.5
        else 1.0 - score
    )

    return sentiment, confidence


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model_and_tokenizer():

    try:

        with open(
            "tokenizer.pkl",
            "rb",
        ) as handle:

            tokenizer = pickle.load(
                handle
            )

        model = tf.keras.models.load_model(
            "model.h5"
        )

        return tokenizer, model

    except Exception as e:

        st.error(
            f"Error loading model or tokenizer: {str(e)}"
        )

        return None, None


# ============================================================
# RESULT COMPONENT
# ============================================================

def show_result(
    sentiment,
    confidence,
    title="Analysis Result",
):

    is_positive = (
        sentiment == "Positive"
    )

    card_class = (
        "positive-result"
        if is_positive
        else "negative-result"
    )

    text_class = (
        "positive-text"
        if is_positive
        else "negative-text"
    )

    emoji = (
        "🎬"
        if is_positive
        else "🍿"
    )

    render_html(
        f"""
        <div class="result-card {card_class}">

            <div class="result-label">
                {title}
            </div>

            <div class="result-sentiment {text_class}">
                {emoji} {sentiment}
            </div>

            <div class="confidence-text">
                Model confidence:
                <strong>{confidence:.2%}</strong>
            </div>

        </div>
        """
    )

    st.progress(
        min(
            max(
                confidence,
                0.0,
            ),
            1.0,
        ),
        text=f"Confidence: {confidence:.2%}",
    )


# ============================================================
# BATCH CSV ANALYSIS
# ============================================================

def render_batch_analysis(
    tokenizer,
    model,
):

    render_html(
        """
        <div class="section-label">
            Batch Analysis
        </div>

        <div class="section-title">
            📂 Analyze Multiple Reviews
        </div>
        """
    )

    render_html(
        """
        <div class="info-box">

            Upload a CSV containing a review column.

            Supported column names include
            <b>Tweet Text</b>,
            <b>review</b>,
            <b>text</b>,
            and <b>comment</b>.

            <br><br>

            Maximum file size:
            <b>5 MiB</b>

            &nbsp;|&nbsp;

            Maximum analyzed rows:
            <b>50</b>

        </div>
        """
    )

    st.write("")

    uploaded_csv = st.file_uploader(
        "Upload your CSV file",
        type=["csv"],
        key="review_csv_upload",
        label_visibility="collapsed",
    )

    if uploaded_csv is None:
        return

    if uploaded_csv.size > MAX_CSV_BYTES:

        st.error(
            "CSV file must be 5 MiB or smaller."
        )

        return

    try:

        batch = read_texts_from_csv(
            uploaded_csv.getvalue()
        )

    except ValueError as error:

        st.error(
            str(error)
        )

        return

    if batch.truncated:

        st.warning(
            f"Only the first {MAX_CSV_ROWS} "
            "non-empty rows are analyzed."
        )

    results = []

    with st.spinner(
        "Analyzing your reviews..."
    ):

        for review in batch.texts:

            sentiment, confidence = (
                predict_sentiment(
                    review,
                    tokenizer,
                    model,
                )
            )

            results.append(
                {
                    "review": review,
                    "sentiment": sentiment,
                    "confidence": f"{confidence:.2%}",
                }
            )

    positive_count = sum(
        result["sentiment"] == "Positive"
        for result in results
    )

    negative_count = (
        len(results)
        - positive_count
    )

    st.success(
        f"Successfully analyzed "
        f"{len(results)} reviews."
    )

    # --------------------------------------------------------
    # SUMMARY METRICS
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "📝 Reviews",
            len(results),
        )

    with col2:

        st.metric(
            "😊 Positive",
            positive_count,
        )

    with col3:

        st.metric(
            "😞 Negative",
            negative_count,
        )

    st.write("")

    # --------------------------------------------------------
    # RESULTS TABLE
    # --------------------------------------------------------

    st.dataframe(
        results,
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        label="⬇️ Download CSV Results",
        data=render_batch_results_csv(
            results
        ),
        file_name="sentiment_results.csv",
        mime="text/csv",
        key="download_csv_results_button",
        use_container_width=True,
    )


# ============================================================
# HOME PAGE
# ============================================================

def home_page():

    # --------------------------------------------------------
    # HERO
    # --------------------------------------------------------

    render_html(
        """
        <div class="hero">

            <div class="hero-badge">
                ✨ DEEP LEARNING • LSTM • NLP
            </div>

            <div class="hero-title">
                CineSentiment AI
            </div>

            <div class="hero-subtitle">
                Understand the emotion behind movie reviews.
                Our deep learning model analyzes your text
                and predicts whether the sentiment is
                <b>Positive</b> or <b>Negative</b> —
                together with a confidence score.
            </div>

        </div>
        """
    )

    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    render_html(
        """
        <div class="section-label">
            What You Can Do
        </div>

        <div class="section-title">
            Intelligent Review Analysis
        </div>
        """
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        render_html(
            """
            <div class="feature-card">

                <div class="feature-icon">
                    🎭
                </div>

                <div class="feature-title">
                    Sentiment Prediction
                </div>

                <div class="feature-text">
                    Enter any movie review and
                    instantly classify it as
                    positive or negative.
                </div>

            </div>
            """
        )

    with col2:

        render_html(
            """
            <div class="feature-card">

                <div class="feature-icon">
                    🧠
                </div>

                <div class="feature-title">
                    LSTM Deep Learning
                </div>

                <div class="feature-text">
                    Powered by a trained LSTM
                    neural network for natural
                    language sentiment analysis.
                </div>

            </div>
            """
        )

    with col3:

        render_html(
            """
            <div class="feature-card">

                <div class="feature-icon">
                    📊
                </div>

                <div class="feature-title">
                    Batch Analysis
                </div>

                <div class="feature-text">
                    Upload a CSV and analyze
                    multiple movie reviews
                    automatically.
                </div>

            </div>
            """
        )

    st.write("")
    st.write("")

    # --------------------------------------------------------
    # WORKFLOW
    # --------------------------------------------------------

    render_html(
        """
        <div class="section-label">
            Workflow
        </div>

        <div class="section-title">
            How It Works
        </div>
        """
    )

    col1, col2, col3, col4 = st.columns(4)

    steps = [
        (
            "01",
            "✍️",
            "Enter Review",
            "Type or paste a movie review.",
        ),
        (
            "02",
            "🔤",
            "Tokenization",
            "Text is converted into numerical sequences.",
        ),
        (
            "03",
            "🧠",
            "LSTM Model",
            "The trained model analyzes the sequence.",
        ),
        (
            "04",
            "🎯",
            "Prediction",
            "Sentiment and confidence are displayed.",
        ),
    ]

    columns = [
        col1,
        col2,
        col3,
        col4,
    ]

    for column, step in zip(
        columns,
        steps,
    ):

        number, icon, title, description = step

        with column:

            render_html(
                f"""
                <div class="feature-card">

                    <div style="
                        color:#a78bfa;
                        font-size:0.75rem;
                        font-weight:800;
                    ">
                        {number}
                    </div>

                    <div class="feature-icon">
                        {icon}
                    </div>

                    <div class="feature-title">
                        {title}
                    </div>

                    <div class="feature-text">
                        {description}
                    </div>

                </div>
                """
            )

    # --------------------------------------------------------
    # FOOTER
    # --------------------------------------------------------

    render_html(
        """
        <div class="footer">
            CineSentiment AI • Deep Learning Sentiment Analysis
        </div>
        """
    )


# ============================================================
# PREDICTION PAGE
# ============================================================

def prediction_page():

    # --------------------------------------------------------
    # SESSION STATE
    # --------------------------------------------------------

    if "analysis_history" not in st.session_state:

        st.session_state.analysis_history = []

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    tokenizer, model = (
        load_model_and_tokenizer()
    )

    if tokenizer is None or model is None:

        st.error(
            "Failed to load model or tokenizer. "
            "Check that model.h5 and tokenizer.pkl "
            "are present."
        )

        return

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    render_html(
        """
        <div class="section-label">
            AI Prediction
        </div>

        <div class="section-title">
            🎬 Movie Review Sentiment
        </div>
        """
    )

    render_html(
        """
        <div class="info-box">

            Enter a movie review below and let the
            LSTM model determine its sentiment.

        </div>
        """
    )

    st.write("")

    # ========================================================
    # SINGLE REVIEW
    # ========================================================

    st.markdown(
        "### ✍️ Write Your Review"
    )

    with st.form(
        key="review_form"
    ):

        review_text = st.text_area(
            "Movie review",
            placeholder=(
                "Example: The acting was incredible "
                "and the story kept me hooked from "
                "beginning to end!"
            ),
            height=180,
            label_visibility="collapsed",
        )

        submitted = st.form_submit_button(
            "✨ Analyze Sentiment",
            use_container_width=True,
            type="primary",
        )

    if submitted:

        if not review_text.strip():

            st.warning(
                "Please enter a movie review first."
            )

        else:

            with st.spinner(
                "The AI is analyzing your review..."
            ):

                sentiment, confidence = (
                    predict_sentiment(
                        review_text,
                        tokenizer,
                        model,
                    )
                )

            show_result(
                sentiment,
                confidence,
            )

            current_time = (
                datetime.now().strftime(
                    "%I:%M %p IST, %A, %B %d, %Y"
                )
            )

            st.session_state.analysis_history.append(
                {
                    "review": review_text,
                    "sentiment": sentiment,
                    "confidence": f"{confidence:.2%}",
                    "timestamp": current_time,
                }
            )

    # ========================================================
    # EXAMPLE REVIEWS
    # ========================================================

    st.divider()

    render_html(
        """
        <div class="section-label">
            Quick Try
        </div>

        <div class="section-title">
            🍿 Try Example Reviews
        </div>
        """
    )

    example_reviews = [
        (
            "🌟",
            "Highly Positive",
            (
                "This movie was absolutely fantastic! "
                "Great acting and storyline."
            ),
        ),
        (
            "💀",
            "Highly Negative",
            (
                "Terrible waste of time, poor direction "
                "and boring plot."
            ),
        ),
        (
            "🎞️",
            "Mixed",
            (
                "It was okay, not great but watchable."
            ),
        ),
    ]

    example_cols = st.columns(3)

    for i, example_data in enumerate(
        example_reviews
    ):

        icon, label, example = example_data

        with example_cols[i]:

            render_html(
                f"""
                <div class="feature-card">

                    <div class="feature-icon">
                        {icon}
                    </div>

                    <div class="feature-title">
                        {label}
                    </div>

                    <div class="feature-text">
                        "{example}"
                    </div>

                </div>
                """
            )

            if st.button(
                "Analyze Example",
                key=f"ex_btn_{i}",
                use_container_width=True,
            ):

                sentiment, confidence = (
                    predict_sentiment(
                        example,
                        tokenizer,
                        model,
                    )
                )

                show_result(
                    sentiment,
                    confidence,
                    title="Example Analysis",
                )

                st.caption(
                    f"Review: {example}"
                )

                current_time = (
                    datetime.now().strftime(
                        "%I:%M %p IST, %A, %B %d, %Y"
                    )
                )

                st.session_state.analysis_history.append(
                    {
                        "review": example,
                        "sentiment": sentiment,
                        "confidence": f"{confidence:.2%}",
                        "timestamp": current_time,
                    }
                )

    # ========================================================
    # BATCH ANALYSIS
    # ========================================================

    st.divider()

    render_batch_analysis(
        tokenizer,
        model,
    )

    # ========================================================
    # HISTORY
    # ========================================================

    st.divider()

    render_html(
        """
        <div class="section-label">
            Session
        </div>

        <div class="section-title">
            🕘 Analysis History
        </div>
        """
    )

    history = (
        st.session_state.analysis_history
    )

    if history:

        for entry in reversed(history):

            sentiment_class = (
                "positive-text"
                if entry["sentiment"] == "Positive"
                else "negative-text"
            )

            safe_review = html.escape(
                entry["review"]
            )

            render_html(
                f"""
                <div class="history-card">

                    <div class="history-meta">
                        {entry["timestamp"]}
                    </div>

                    <div class="history-review">
                        {safe_review}
                    </div>

                    <br>

                    <b class="{sentiment_class}">
                        {entry["sentiment"]}
                    </b>

                    &nbsp;&nbsp;

                    <span class="history-meta">
                        Confidence: {entry["confidence"]}
                    </span>

                </div>
                """
            )

    else:

        render_html(
            """
            <div class="info-box">

                No reviews analyzed yet.

                Your analysis history will appear here.

            </div>
            """
        )

    # ========================================================
    # HISTORY EXPORT
    # ========================================================

    def format_history_as_markdown():

        if not history:

            return (
                "# Analysis History\n\n"
                "No analysis history available."
            )

        markdown_content = (
            "# CineSentiment AI - Analysis History\n\n"
        )

        for entry in history:

            markdown_content += (
                f"## Review "
                f"(Analyzed on {entry['timestamp']})\n"
            )

            markdown_content += (
                f"{entry['review']}\n\n"
            )

            markdown_content += (
                "## Result\n"
            )

            markdown_content += (
                f"- **Sentiment**: "
                f"{entry['sentiment']}\n"
            )

            markdown_content += (
                f"- **Confidence**: "
                f"{entry['confidence']}\n\n"
            )

            markdown_content += (
                "---\n\n"
            )

        return markdown_content

    st.write("")

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "🗑️ Clear History",
            key="clear_history_button",
            use_container_width=True,
        ):

            st.session_state.analysis_history = []

            st.rerun()

    with col2:

        markdown_export = format_history_as_markdown()

        st.download_button(
            label="📥 Export History (Markdown)",
            data=markdown_export,
            file_name="analysis_history.md",
            mime="text/markdown",
            key="download_history_button",
            use_container_width=True,
        )


# ============================================================
# MAIN ROUTING
# ============================================================

def main():

    page = st.radio(
        "Navigation",
        options=["Home", "Analyze"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if page == "Home":
        home_page()
    else:
        prediction_page()


if __name__ == "__main__":
    main()