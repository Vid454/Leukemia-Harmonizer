import streamlit as st
import pandas as pd

from harmonizer import harmonize_cbc
from train_model import train_leukemia_model
from predict import load_trained_model, predict_leukemia


st.set_page_config(page_title="Leukemia Harmonizer", layout="wide")
st.title("🔬 Leukemia Harmonizer – CBC → ML Pipeline")

tab1, tab2, tab3 = st.tabs(["1️⃣ Harmonize", "2️⃣ Train Model", "3️⃣ Predict"])


# TAB 1 – HARMONIZE
with tab1:
    st.header("Step 1: Harmonize CBC Datasets")

    uploaded_files = st.file_uploader(
        "Upload one or more messy CBC CSV files",
        type=["csv"],
        accept_multiple_files=True
    )

    if uploaded_files:
        st.write(f"Uploaded files: {[f.name for f in uploaded_files]}")

        dataframes = []
        for f in uploaded_files:
            df = pd.read_csv(f)
            df["source_file"] = f.name
            dataframes.append(df)

        raw_df = pd.concat(dataframes, ignore_index=True)
        st.subheader("Raw merged data (head)")
        st.dataframe(raw_df.head())

        if st.button("Run Harmonization"):
            clean_df, report_text = harmonize_cbc(raw_df)

            st.subheader("Harmonized CBC data (head)")
            st.dataframe(clean_df.head())

            st.subheader("Quality report (summary)")
            st.text(report_text)

            csv_bytes = clean_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download harmonized_CBC.csv",
                data=csv_bytes,
                file_name="harmonized_CBC.csv",
                mime="text/csv"
            )


# TAB 2 – TRAIN MODEL
with tab2:
    st.header("Step 2: Train Leukemia Classification Model")

    clean_file = st.file_uploader(
        "Upload harmonized CBC CSV (output of Step 1)",
        type=["csv"],
        key="clean_upload"
    )

    if clean_file:
        clean_df = pd.read_csv(clean_file)
        st.subheader("Harmonized data (head)")
        st.dataframe(clean_df.head())

        target_col = st.selectbox(
            "Select target label column (AML/ALL/CLL/CML/Normal)",
            options=[c for c in clean_df.columns if c.lower() in ["label", "diagnosis", "class"]],
            index=0 if any(c.lower() in ["label", "diagnosis", "class"] for c in clean_df.columns) else 0
        )

        if st.button("Train XGBoost Model"):
            with st.spinner("Training model..."):
                model, metrics = train_leukemia_model(clean_df, target_col)

            st.subheader("Training metrics")
            st.json(metrics)

            st.success("Model trained (in memory). For full project, save to disk with joblib.")


# TAB 3 – PREDICT
with tab3:
    st.header("Step 3: Predict Leukemia on New CBC Data")

    st.write("This tab demonstrates using a previously trained model to predict AML/ALL/CLL/CML/Normal.")

    model_placeholder = st.text_input(
        "For now, imagine we loaded leukemia_model.pkl here (file load disabled in this demo).",
        value="leukemia_model.pkl"
    )

    new_cbc_file = st.file_uploader(
        "Upload new CBC CSV for prediction",
        type=["csv"],
        key="predict_upload"
    )

    if new_cbc_file:
        new_df_raw = pd.read_csv(new_cbc_file)
        st.subheader("New CBC data (raw head)")
        st.dataframe(new_df_raw.head())

        if st.button("Run Prediction (demo)"):
            # In a real system we would:
            # 1. re-use harmonize_cbc on new_df_raw
            # 2. load the saved model file
            # 3. run predict_leukemia
            pred_df = new_df_raw.copy()
            pred_df["predicted_class"] = "AML"   # dummy placeholder
            pred_df["AML_prob"] = 0.92           # dummy placeholder
            pred_df["Normal_prob"] = 0.03        # dummy placeholder

            st.subheader("Prediction results (demo)")
            st.dataframe(pred_df.head())
            st.info("Tomorrow you can explain this as a placeholder for the real ML predictions.")
