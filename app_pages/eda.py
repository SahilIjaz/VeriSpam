"""Exploratory analysis: distributions, most common words, missing/duplicate breakdowns."""

import altair as alt
import streamlit as st

from src.eda import (
    class_distribution,
    duplicate_summary,
    missing_value_breakdown,
    most_common_words,
    most_common_words_by_class,
    text_stats,
)
from src.state import require_columns

text_col, label_col = require_columns()
df = st.session_state.dataset

tab_dist, tab_length, tab_words, tab_quality = st.tabs(
    ["Class distribution", "Text length", "Common words", "Data quality"]
)

with tab_dist:
    dist = class_distribution(df, label_col)
    st.bar_chart(dist)
    st.dataframe(dist.rename("count").to_frame(), width="stretch")

with tab_length:
    stats = text_stats(df, text_col)
    metric_col = st.segmented_control(
        "Metric", options=["char_count", "word_count", "avg_word_length"], default="char_count"
    )
    metric_col = metric_col or "char_count"
    chart = (
        alt.Chart(stats)
        .mark_bar()
        .encode(x=alt.X(f"{metric_col}:Q", bin=alt.Bin(maxbins=40), title=metric_col), y=alt.Y("count()", title="messages"))
    )
    st.altair_chart(chart)
    st.dataframe(stats.describe().transpose(), width="stretch")

with tab_words:
    top_n = st.slider("Number of words", min_value=5, max_value=40, value=20)
    st.markdown("**Most common words (overall)**")
    overall = most_common_words(df[text_col], top_n=top_n)
    st.bar_chart(overall.set_index("word"))

    by_class = most_common_words_by_class(df, text_col, label_col, top_n=top_n)
    class_cols = st.columns(len(by_class)) if by_class else []
    for col, (cls, words_df) in zip(class_cols, by_class.items()):
        with col:
            st.markdown(f"**Most common words: {cls}**")
            st.bar_chart(words_df.set_index("word"))

with tab_quality:
    st.markdown("**Missing values by column**")
    st.dataframe(missing_value_breakdown(df), width="stretch", hide_index=True)

    n_duplicates, dup_preview = duplicate_summary(df)
    st.markdown(f"**Duplicate rows:** {n_duplicates:,}")
    if n_duplicates:
        st.dataframe(dup_preview, width="stretch")

st.divider()
st.page_link("app_pages/preprocessing.py", label="Continue to Preprocessing", icon=":material/auto_fix_high:")
