import streamlit as st
import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="HR Attrition Intelligence", layout="wide")

@st.cache_data
def load_csv(path_or_buffer):
    return pd.read_csv(path_or_buffer)

def robust_impute(df, target_col):
    df = df.copy()
    if target_col in df.columns:
        df = df.dropna(subset=[target_col])
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols and c != target_col]
    for col in numeric_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mean())
    for col in categorical_cols:
        if df[col].isna().any():
            mode_val = df[col].mode(dropna=True)
            df[col] = df[col].fillna(mode_val.iloc[0] if len(mode_val) else "Unknown")
    return df, numeric_cols, categorical_cols

def basic_clean_encode(df, target_col="Attrition"):
    df, numeric_cols, categorical_cols = robust_impute(df, target_col)
    label_le = LabelEncoder()
    y = label_le.fit_transform(df[target_col])
    oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1)
    if categorical_cols:
        X_cats = pd.DataFrame(oe.fit_transform(df[categorical_cols]), columns=categorical_cols, index=df.index).astype(int)
    else:
        X_cats = pd.DataFrame(index=df.index)
    X_nums = df[numeric_cols].copy()
    X = pd.concat([X_nums, X_cats], axis=1)
    metadata = {"numeric_cols": numeric_cols, "categorical_cols": categorical_cols, "label_classes": list(label_le.classes_)}
    return df, X, y, oe, label_le, metadata

def transform_new_data(new_df, oe, label_le, metadata, target_col="Attrition", is_training_like=False):
    df = new_df.copy()
    known_cols = metadata["numeric_cols"] + metadata["categorical_cols"]
    for c in known_cols:
        if c not in df.columns:
            df[c] = np.nan
    df = df[known_cols + ([target_col] if (target_col in df.columns) else [])]
    # impute
    for col in metadata["numeric_cols"]:
        df[col] = df[col].fillna(df[col].mean())
    for col in metadata["categorical_cols"]:
        mode_val = df[col].mode(dropna=True)
        df[col] = df[col].fillna(mode_val.iloc[0] if len(mode_val) else "Unknown")
    if metadata["categorical_cols"]:
        X_cats = pd.DataFrame(oe.transform(df[metadata["categorical_cols"]]), columns=metadata["categorical_cols"], index=df.index).astype(int)
    else:
        X_cats = pd.DataFrame(index=df.index)
    X_nums = df[metadata["numeric_cols"]].copy()
    X = pd.concat([X_nums, X_cats], axis=1)
    y = None
    if target_col in df.columns and is_training_like:
        y = label_le.transform(df[target_col])
    return df, X, y

def train_models(X, y, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=random_state),
        "Gradient Boosting": GradientBoostingClassifier(random_state=random_state)
    }
    for m in models.values():
        m.fit(X_train, y_train)
    return models, X_train, X_test, y_train, y_test

def get_proba_or_score(model, X):
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.shape[1] == 2:
            return proba[:, 1]
        return None
    if hasattr(model, "decision_function"):
        s = model.decision_function(X)
        if s.ndim == 1:
            return s
        return None
    return None

def metrics_summary(models, X_train, X_test, y_train, y_test, binary=True):
    average = "binary" if binary else "macro"
    rows, scores_test, preds_train, preds_test = [], {}, {}, {}
    for name, model in models.items():
        yt_hat_tr = model.predict(X_train); preds_train[name] = yt_hat_tr
        yt_hat_te = model.predict(X_test);  preds_test[name]  = yt_hat_te
        s_test = get_proba_or_score(model, X_test); scores_test[name] = s_test
        acc_tr = accuracy_score(y_train, yt_hat_tr)
        acc_te = accuracy_score(y_test,  yt_hat_te)
        prec  = precision_score(y_test, yt_hat_te, average=average, zero_division=0)
        rec   = recall_score(y_test,    yt_hat_te, average=average, zero_division=0)
        f1    = f1_score(y_test,        yt_hat_te, average=average, zero_division=0)
        if binary and s_test is not None:
            auc_val = roc_auc_score(y_test, s_test)
        else:
            auc_val = roc_auc_score(y_test, model.predict_proba(X_test), multi_class="ovr", average="macro") if hasattr(model, "predict_proba") else np.nan
        rows.append([name, acc_tr, acc_te, prec, rec, f1, auc_val])
    dfm = pd.DataFrame(rows, columns=["Algorithm","Train Accuracy","Test Accuracy","Precision","Recall","F1","AUC"]).sort_values("Test Accuracy", ascending=False)
    return dfm, preds_train, preds_test, scores_test

def plot_cm(cm, class_names, title):
    fig = go.Figure(data=go.Heatmap(z=cm, x=class_names, y=class_names, text=cm, texttemplate="%{text}", hovertemplate="True=%{y}<br>Pred=%{x}<br>Count=%{z}<extra></extra>"))
    fig.update_layout(title=title, xaxis_title="Predicted", yaxis_title="Actual")
    return fig

def roc_overlay(scores_test, y_test):
    fig = go.Figure()
    for name, scores in scores_test.items():
        if scores is None: continue
        fpr, tpr, _ = roc_curve(y_test, scores)
        auc_val = roc_auc_score(y_test, scores)
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} (AUC={auc_val:.3f})"))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Chance", line=dict(dash="dash")))
    fig.update_layout(title="ROC Curves (Test Set)", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    return fig

def feature_importance_fig(model, feature_names, title, top_n=15):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        idx = np.argsort(importances)[::-1][:top_n]
        fig = px.bar(x=[feature_names[i] for i in idx], y=importances[idx], title=title)
        fig.update_layout(xaxis_title="Feature", yaxis_title="Importance")
        return fig
    return go.Figure()

st.title("📊 HR Attrition Intelligence Dashboard")

with st.expander("Data Source", expanded=True):
    st.markdown("Upload your **EA.csv** (must include `Attrition`) or place it next to `app.py`. Nulls are auto-handled (mean/mode).")
    uploaded = st.file_uploader("Upload EA.csv", type=["csv"])
    default_path = "EA.csv"
    df_raw = None
    if uploaded is not None:
        df_raw = load_csv(uploaded)
    elif os.path.exists(default_path):
        df_raw = load_csv(default_path)
        st.info("Loaded local EA.csv from working directory.")
    else:
        st.warning("Please upload EA.csv to proceed.")

if df_raw is None:
    st.stop()

df_clean, X, y, oe, label_le, metadata = basic_clean_encode(df_raw, target_col="Attrition")
class_names = list(label_le.classes_)
binary = len(np.unique(y)) == 2

jobrole_values = df_clean["JobRole"].dropna().unique().tolist() if "JobRole" in df_clean.columns else []
selected_roles = st.multiselect("Filter by JobRole (applies to charts)", jobrole_values, default=jobrole_values)
satisfaction_cols = [c for c in df_clean.columns if "Satisfaction" in c]
sat_col = st.selectbox("Pick a Satisfaction column for slider filter", options=satisfaction_cols if satisfaction_cols else ["(none)"])
sat_min, sat_max = 1, 5
if sat_col != "(none)":
    try:
        sat_min = int(np.nanmin(df_clean[sat_col])); sat_max = int(np.nanmax(df_clean[sat_col]))
        if sat_min == sat_max: sat_min = max(0, sat_min-1)
    except Exception: pass
sat_threshold = st.slider("Minimum selected satisfaction value", min_value=int(sat_min), max_value=int(sat_max), value=int(sat_min), step=1)

df_view = df_clean.copy()
if selected_roles and "JobRole" in df_view.columns:
    df_view = df_view[df_view["JobRole"].isin(selected_roles)]
if sat_col != "(none)":
    df_view = df_view[df_view[sat_col] >= sat_threshold]

tab1, tab2, tab3 = st.tabs(["📈 Insights", "🤖 Models & Metrics", "🧮 Predict on New Data"])

with tab1:
    st.subheader("Actionable Insights")
    if "JobRole" in df_view.columns:
        tmp = df_view.groupby(["JobRole","Attrition"]).size().reset_index(name="count")
        total = tmp.groupby("JobRole")["count"].transform("sum")
        tmp["rate"] = tmp["count"] / total
        fig1 = px.bar(tmp, x="JobRole", y="rate", color="Attrition", barmode="group", title="Attrition Rate by JobRole")
        fig1.update_layout(xaxis_title="JobRole", yaxis_title="Attrition Rate")
        st.plotly_chart(fig1, use_container_width=True)

    if "MonthlyIncome" in df_view.columns and "Attrition" in df_view.columns:
        fig2 = px.box(df_view, x="Attrition", y="MonthlyIncome", color="Attrition", title="Monthly Income Distribution by Attrition")
        st.plotly_chart(fig2, use_container_width=True)

    if "Overtime" in df_view.columns and "Attrition" in df_view.columns:
        tmp = df_view.groupby(["Overtime","Attrition"]).size().reset_index(name="count")
        total = tmp.groupby("Overtime")["count"].transform("sum")
        tmp["rate"] = tmp["count"] / total
        fig3 = px.bar(tmp, x="Overtime", y="rate", color="Attrition", barmode="stack", title="Attrition vs Overtime (Rate)")
        st.plotly_chart(fig3, use_container_width=True)

    if "YearsAtCompany" in df_view.columns and "Attrition" in df_view.columns:
        bins = pd.cut(df_view["YearsAtCompany"], bins=[-1,1,3,5,10,40], labels=["0-1","1-3","3-5","5-10","10+"])
        tmp = df_view.groupby([bins,"Attrition"]).size().reset_index(name="count")
        total = tmp.groupby(0)["count"].transform("sum")
        tmp["rate"] = tmp["count"] / total
        focus = tmp[tmp["Attrition"]=="Yes"] if "Yes" in class_names else tmp
        fig4 = px.line(focus, x=0, y="rate", markers=True, title="Attrition Rate across Tenure Buckets (Yes)")
        fig4.update_layout(xaxis_title="YearsAtCompany Bucket", yaxis_title="Attrition Rate (Yes)")
        st.plotly_chart(fig4, use_container_width=True)

    df_corr = df_view.copy()
    df_corr["_AttritionEnc"] = LabelEncoder().fit_transform(df_corr["Attrition"])
    num_cols = df_corr.select_dtypes(include=[np.number]).columns.tolist()
    if "_AttritionEnc" in num_cols: num_cols = [c for c in num_cols if c != "_AttritionEnc"]
    corr = df_corr[num_cols + ["_AttritionEnc"]].corr()["_AttritionEnc"].drop("_AttritionEnc").sort_values(ascending=False).head(15)
    fig5 = px.bar(x=corr.index, y=corr.values, title="Top 15 Numeric Correlations with Attrition (encoded)")
    fig5.update_layout(xaxis_title="Feature", yaxis_title="Correlation with Attrition (encoded)")
    st.plotly_chart(fig5, use_container_width=True)

with tab2:
    st.subheader("Train Models & Evaluate")
    run_btn = st.button("Run Decision Tree, Random Forest, Gradient Boosting (cv=5, stratified)")
    if run_btn:
        models, X_train, X_test, y_train, y_test = train_models(X, y)
        metrics_df, preds_train, preds_test, scores_test = metrics_summary(models, X_train, X_test, y_train, y_test, binary=binary)
        st.dataframe(metrics_df.style.format({"Train Accuracy": "{:.4f}","Test Accuracy": "{:.4f}","Precision": "{:.4f}","Recall": "{:.4f}","F1": "{:.4f}","AUC": "{:.4f}"}), use_container_width=True)
        for name, model in models.items():
            cm_tr = confusion_matrix(y_train, preds_train[name], labels=np.arange(len(class_names)))
            cm_te = confusion_matrix(y_test,  preds_test[name],  labels=np.arange(len(class_names)))
            st.plotly_chart(plot_cm(cm_tr, class_names, f"{name} - Train Confusion Matrix"), use_container_width=True)
            st.plotly_chart(plot_cm(cm_te, class_names, f"{name} - Test Confusion Matrix"),  use_container_width=True)
        if binary:
            st.plotly_chart(roc_overlay(scores_test, y_test), use_container_width=True)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_rows = []
        for name, model in models.items():
            scoring = {'acc': 'accuracy', 'auc': 'roc_auc'} if binary else {'acc':'accuracy', 'auc':'roc_auc_ovr'}
            cv_res = cross_validate(model, X, y, cv=cv, scoring=scoring, return_train_score=False)
            cv_rows.append({"Algorithm": name, "CV Acc (mean±std)": f"{cv_res['test_acc'].mean():.4f} ± {cv_res['test_acc'].std():.4f}", "CV AUC (mean±std)": f"{cv_res['test_auc'].mean():.4f} ± {cv_res['test_auc'].std():.4f}"})
        st.dataframe(pd.DataFrame(cv_rows), use_container_width=True)
        for name, model in models.items():
            st.plotly_chart(feature_importance_fig(model, X.columns.tolist(), f"{name} - Top 15 Feature Importances"), use_container_width=True)

with tab3:
    st.subheader("Upload new data to predict Attrition")
    st.markdown("Upload a CSV with similar columns (no need to contain `Attrition`). Nulls will be auto-imputed (mean/mode).")
    up = st.file_uploader("Upload new HR dataset", type=["csv"], key="pred_upload")
    chosen_model = st.selectbox("Choose model for prediction", ["Random Forest","Gradient Boosting","Decision Tree"])
    pred_btn = st.button("Predict Attrition")
    if pred_btn:
        models, X_train, X_test, y_train, y_test = train_models(X, y)
        model = models[chosen_model]
        if up is None:
            st.warning("Please upload a CSV to predict.")
        else:
            new_df = pd.read_csv(up)
            _, X_new, _ = transform_new_data(new_df, oe, label_le, metadata, target_col="Attrition", is_training_like=False)
            preds = model.predict(X_new)
            pred_labels = [label_le.classes_[i] for i in preds]
            out = new_df.copy()
            out["Predicted_Attrition"] = pred_labels
            st.success(f"Predictions done using {chosen_model}. Preview below:")
            st.dataframe(out.head(50), use_container_width=True)
            csv_bytes = out.to_csv(index=False).encode("utf-8")
            st.download_button("Download predictions as CSV", data=csv_bytes, file_name="attrition_predictions.csv", mime="text/csv")
