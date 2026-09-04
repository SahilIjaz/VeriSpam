"""Dataset upload: file intake, file/quality summary, and text/label column selection."""

import streamlit as st

from src.column_detection import guess_text_label_columns
from src.config import ALLOWED_EXTENSIONS, MAX_FILE_SIZE_MB
from src.eda import dataset_overview
from src.loader import load_uploaded_file
from src.state import set_dataset

st.caption("CSV or Excel, up to " + f"{MAX_FILE_SIZE_MB} MB. The file stays in memory for this session only — nothing is written to disk.")

uploaded_file = st.file_uploader(
    "Upload dataset",
    type=[ext.lstrip(".") for ext in ALLOWED_EXTENSIONS],
    help="A spam/ham dataset with a message-text column and a label column (e.g. spam/ham).",
)

if uploaded_file is not None:
    signature = (uploaded_file.name, uploaded_file.size)
    if st.session_state.get("_upload_signature") != signature:
        load_result = load_uploaded_file(uploaded_file)
        if not load_result.ok:
            st.error(load_result.error)
            st.stop()
        if load_result.size_mb > MAX_FILE_SIZE_MB:
            st.error(
                f"File is {load_result.size_mb:.1f} MB, which exceeds the {MAX_FILE_SIZE_MB} MB limit."
            )
            st.stop()
        set_dataset(load_result.dataframe, load_result)
        st.session_state._upload_signature = signature
        st.session_state.text_col, st.session_state.label_col = guess_text_label_columns(load_result.dataframe)

dataset = st.session_state.get("dataset")
load_result = st.session_state.get("dataset_meta")

if dataset is None:
    st.info("No dataset loaded yet. Upload a file above, or load the bundled sample from Home.", icon=":material/upload_file:")
    st.stop()

overview = dataset_overview(dataset)
n_duplicates = int(dataset.duplicated().sum())
n_missing = int(dataset.isna().sum().sum())

st.subheader("File summary")
with st.container(horizontal=True):
    st.metric("File name", load_result.filename, border=True)
    st.metric("File size", f"{load_result.size_kb:.1f} KB", border=True)
    st.metric("Rows", f"{overview['n_rows']:,}", border=True)
    st.metric("Columns", overview["n_cols"], border=True)
    st.metric("Memory usage", f"{overview['memory_usage_mb']:.2f} MB", border=True)

with st.container(horizontal=True):
    st.metric("Missing cells", f"{n_missing:,}", border=True)
    st.metric("Duplicate rows", f"{n_duplicates:,}", border=True)
    st.metric("Data types", len(overview["dtype_counts"]), border=True, help=str(overview["dtype_counts"]))

st.subheader("Column selection")
st.caption("Auto-detected where possible — override if the guess is wrong.")
columns = list(dataset.columns)
col1, col2 = st.columns(2)
with col1:
    text_default = columns.index(st.session_state.text_col) if st.session_state.text_col in columns else 0
    st.session_state.text_col = st.selectbox(
        "Text column", options=columns, index=text_default, help="The column containing the message/email text."
    )
with col2:
    label_default = columns.index(st.session_state.label_col) if st.session_state.label_col in columns else 0
    st.session_state.label_col = st.selectbox(
        "Label column", options=columns, index=label_default, help="The column containing the spam/ham label."
    )

if st.session_state.text_col == st.session_state.label_col:
    st.warning("Text column and label column should be different.", icon=":material/warning:")

with st.expander("Preview data", expanded=False):
    st.dataframe(dataset.head(50), width="stretch")

with st.expander("Column details", expanded=False):
    st.dataframe(
        dataset.dtypes.astype(str).rename("dtype").to_frame().assign(
            missing=dataset.isna().sum(), unique=dataset.nunique(dropna=True)
        ),
        width="stretch",
    )

st.divider()
st.page_link("app_pages/validation_page.py", label="Continue to Data Validation", icon=":material/fact_check:")
