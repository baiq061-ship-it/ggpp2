# ==============================
# 必须第一条 Streamlit 命令
# ==============================
import streamlit as st

st.set_page_config(
    page_title="Clinical Prediction System",
    layout="wide",
    page_icon="🏥"
)

# ==============================
# 其它库导入
# ==============================
import os
import joblib
import pandas as pd
import shap
import numpy as np
import streamlit.components.v1 as components
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ==============================
# >>> 新的预测因子
# ==============================
FEATURES = [
    "sz1","sz2","sz3","sz4","sz5",
    "lxsz",
    "cjl0","cjl1","cjl2","cjl3","cjl4","cjl5",
    "lb1","lb2","lb3","lb4","lb5","lb6",
    "lb7","lb8","lb9"
]


# ==============================
# SHAP交互图函数
# ==============================
def st_shap(plot, height=180):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height, scrolling=True)


# ==============================
# 模型路径
# ==============================
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH_1 = os.path.join(BASE_DIR, "best_model.pkl")
MODEL_PATH_2 = os.path.join(BASE_DIR, "model.pkl")


# ==============================
# 模型加载（缓存）
# ==============================
@st.cache_resource
def load_model():

    if os.path.exists(MODEL_PATH_1):
        return joblib.load(MODEL_PATH_1)
    elif os.path.exists(MODEL_PATH_2):
        return joblib.load(MODEL_PATH_2)
    else:
        raise FileNotFoundError("未找到 best_model.pkl 或 model.pkl")


model = load_model()


# ==============================
# 页面标题
# ==============================
st.title("🏥 Diagnosis Prediction System")


# ==============================
# Pipeline处理（兼容scaler/pipeline）
# ==============================
def process_pipeline(model, df):

    shap_model = model
    X = df.copy()

    if hasattr(model, "named_steps"):
        steps = list(model.named_steps.keys())

        shap_model = model.named_steps[steps[-1]]

        if len(steps) > 1:
            preprocessor = model[:-1]
            X = preprocessor.transform(df)

    return shap_model, X


# ==============================
# 页面Tabs
# ==============================
tab1, tab2 = st.tabs(["📝 Single Prediction", "📂 Batch Prediction"])


# =================================================
# 单例预测
# =================================================
with tab1:

    st.subheader("Single Case Prediction")

    input_data = {}
    cols = st.columns(3)

    for i, feature in enumerate(FEATURES):
        with cols[i % 3]:
            input_data[feature] = st.number_input(feature, value=0.0)

    if st.button("Predict"):

        input_df = pd.DataFrame([input_data])

        try:
            prob = model.predict_proba(input_df)[0][1]

            st.divider()
            c1, c2 = st.columns([1, 2])

            # ===== 预测结果 =====
            with c1:
                st.metric("Risk Probability", f"{prob:.2%}")

                if prob > 0.5:
                    st.error("High Risk")
                else:
                    st.success("Low Risk")

            # ===== SHAP解释 =====
            with c2:

                st.subheader("SHAP Model Explanation")

                shap_model, X_for_shap = process_pipeline(model, input_df)

                explainer = shap.TreeExplainer(shap_model)
                shap_values = explainer.shap_values(X_for_shap)

                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
                    base_value = explainer.expected_value[1]
                else:
                    base_value = explainer.expected_value

                if isinstance(base_value, np.ndarray):
                    base_value = base_value[0]

                # Waterfall图
                fig = plt.figure(figsize=(10, 5))

                explanation = shap.Explanation(
                    values=shap_values[0],
                    base_values=base_value,
                    data=input_df.iloc[0],
                    feature_names=input_df.columns
                )

                shap.plots.waterfall(explanation, show=False)
                st.pyplot(fig)
                plt.close(fig)

                # Force Plot
                force_plot = shap.force_plot(
                    base_value,
                    shap_values[0],
                    input_df.iloc[0],
                    matplotlib=False
                )

                st_shap(force_plot)

        except Exception as e:
            st.error("Prediction failed")
            st.text(str(e))


# =================================================
# 批量预测
# =================================================
with tab2:

    st.subheader("Batch Prediction")

    uploaded_file = st.file_uploader(
        "Upload Excel or CSV",
        type=["xlsx", "csv"]
    )

    if uploaded_file:

        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.write("Data Preview:")
        st.dataframe(df.head())

        if st.button("Run Batch Prediction"):

            try:
                probs = model.predict_proba(df)[:, 1]
                df["Prediction_Probability"] = probs

                st.success("Prediction Completed")
                st.dataframe(df)

                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Result CSV",
                    csv,
                    "prediction_results.csv",
                    "text/csv"
                )

            except Exception as e:
                st.error("Batch prediction failed")
                st.text(str(e))