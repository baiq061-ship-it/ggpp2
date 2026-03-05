# ==============================
# Streamlit必须第一条
# ==============================
import streamlit as st

st.set_page_config(
    page_title="Clinical Prediction System",
    layout="wide",
    page_icon="🏥"
)

# ==============================
# 导入库
# ==============================
import os
import joblib
import pandas as pd
import shap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit.components.v1 as components


# ==============================
# 预测变量
# ==============================
FEATURES = [
    "sz1","sz2","sz3","sz4","sz5",
    "lxsz",
    "cjl0","cjl1","cjl2","cjl3","cjl4","cjl5",
    "lb1","lb2","lb3","lb4","lb5","lb6",
    "lb7","lb8","lb9"
]


# ==============================
# SHAP显示函数
# ==============================
def st_shap(plot, height=200):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)


# ==============================
# 加载模型（缓存）
# ==============================
@st.cache_resource
def load_model():
    return joblib.load("best_model.pkl")


model = load_model()


# ==============================
# 标题
# ==============================
st.title("🏥 Clinical Diagnosis Prediction System")


# ==============================
# 页面Tab
# ==============================
tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])


# ==============================
# 单例预测
# ==============================
with tab1:

    st.subheader("Single Case Prediction")

    cols = st.columns(3)
    input_data = {}

    for i, feature in enumerate(FEATURES):
        with cols[i % 3]:
            input_data[feature] = st.number_input(feature, value=0.0)

    if st.button("Predict"):

        input_df = pd.DataFrame([input_data])

        prob = model.predict_proba(input_df)[0][1]

        st.divider()

        c1, c2 = st.columns([1,2])

        with c1:

            st.metric("Risk Probability", f"{prob:.2%}")

            if prob > 0.5:
                st.error("High Risk")
            else:
                st.success("Low Risk")

        with c2:

            st.subheader("SHAP Explanation")

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_df)

            if isinstance(shap_values, list):
                shap_values = shap_values[1]
                base_value = explainer.expected_value[1]
            else:
                base_value = explainer.expected_value

            fig = plt.figure(figsize=(10,5))

            explanation = shap.Explanation(
                values=shap_values[0],
                base_values=base_value,
                data=input_df.iloc[0],
                feature_names=input_df.columns
            )

            shap.plots.waterfall(explanation, show=False)

            st.pyplot(fig)

            force_plot = shap.force_plot(
                base_value,
                shap_values[0],
                input_df.iloc[0],
                matplotlib=False
            )

            st_shap(force_plot)


# ==============================
# 批量预测
# ==============================
with tab2:

    st.subheader("Batch Prediction")

    file = st.file_uploader("Upload CSV / Excel", type=["csv","xlsx"])

    if file:

        if file.name.endswith("csv"):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)

        st.write("Data Preview")
        st.dataframe(df.head())

        if st.button("Run Prediction"):

            probs = model.predict_proba(df)[:,1]

            df["Prediction_Probability"] = probs

            st.success("Prediction Completed")

            st.dataframe(df)

            csv = df.to_csv(index=False).encode("utf-8")

            st.download_button(
                "Download Result",
                csv,
                "prediction_results.csv",
                "text/csv"
            )
