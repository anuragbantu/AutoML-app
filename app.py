import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
from flaml import AutoML
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LogisticRegression

st.title("ML Modeling App")

# 1. Upload CSV
data = None
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Data Preview:", data.head())

if data is not None:
    # 2. Select features and target
    columns = data.columns.tolist()
    target = st.selectbox("Select target variable", columns)
    features = st.multiselect("Select independent variables", [col for col in columns if col != target])

    # New: Preprocessing option for missing values
    st.markdown("**Preprocessing: Handle missing values**")
    missing_option = st.radio(
        "How do you want to handle missing values?",
        ("Drop rows with missing values", "Fill missing values (mean/mode)")
    )

    # Encoding option for categorical variables
    st.markdown("**Encoding: Handle categorical variables**")
    encoding_option = st.radio(
        "How do you want to encode categorical variables?",
        ("One-hot encoding", "Label encoding")
    )

    if features and target:
        X = data[features]
        y = data[target]

        # Apply missing value handling
        if missing_option == "Drop rows with missing values":
            df = pd.concat([X, y], axis=1).dropna()
            X = df[features]
            y = df[target]
        else:
            df = pd.concat([X, y], axis=1)
            for col in features:
                if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                    df[col] = df[col].fillna(df[col].mean())
                else:
                    df[col] = df[col].fillna(df[col].mode()[0])
            # Target variable
            if df[target].dtype in [np.float64, np.float32, np.int64, np.int32]:
                df[target] = df[target].fillna(df[target].mean())
            else:
                df[target] = df[target].fillna(df[target].mode()[0])
            X = df[features]
            y = df[target]

        # Encode categorical variables
        cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        if cat_cols:
            if encoding_option == "One-hot encoding":
                X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
            else:
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                for col in cat_cols:
                    X[col] = le.fit_transform(X[col])

        # 3. Model selection
        problem_type = "Regression"
        test_size = 0.2
        random_state = 42

        # User selects modeling approach
        st.markdown("**Modeling Approach**")
        modeling_option = st.radio(
            "How do you want to model?",
            ("Standard Regression Models", "AutoML (FLAML)")
        )

        if modeling_option == "Standard Regression Models":
            model_name = st.selectbox("Select regression model", ("Linear Regression", "Random Forest Regressor", "Logistic Regression", "XGBoost Regressor"))

        if st.button("Run Model"):
            # 80/20 split, no random state
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            if modeling_option == "Standard Regression Models":
                if model_name == "Linear Regression":
                    reg_model = LinearRegression()
                elif model_name == "Random Forest Regressor":
                    reg_model = RandomForestRegressor()
                elif model_name == "Logistic Regression":
                    reg_model = LogisticRegression(max_iter=1000)
                else:
                    reg_model = XGBRegressor(verbosity=0)
                reg_model.fit(X_train, y_train)
                y_pred = reg_model.predict(X_test)
                best_model = reg_model
            else:
                automl = AutoML()
                automl_settings = {
                    "time_budget": 10,  # seconds
                    "task": "regression",
                    "log_file_name": "automl_regression.log"
                }
                automl.fit(X_train=X_train, y_train=y_train, **automl_settings)
                y_pred = automl.predict(X_test)
                best_model = automl.model

            st.subheader("Error Metrics")
            st.write(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
            st.write(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
            st.write(f"R²: {r2_score(y_test, y_pred):.4f}")

            st.subheader("Replication Plot")
            min_val = min(y_test.min(), y_pred.min())
            max_val = max(y_test.max(), y_pred.max())
            scatter = go.Scatter(x=y_test, y=y_pred, mode='markers', name='Predicted vs Actual')
            ideal = go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines', name='Ideal (45° line)', line=dict(color='red', dash='dash'))
            layout = go.Layout(
                xaxis=dict(title='Actual'),
                yaxis=dict(title='Predicted'),
                title='Actual vs. Predicted',
                showlegend=True
            )
            fig = go.Figure(data=[scatter, ideal], layout=layout)
            st.plotly_chart(fig, use_container_width=True)

            # Feature importance
            st.subheader("Feature Importance")
            feature_names = X.columns if hasattr(X, 'columns') else features
            importances = None
            if hasattr(best_model, "feature_importances_"):
                importances = best_model.feature_importances_
            elif hasattr(best_model, "coef_"):
                coefs = best_model.coef_
                importances = coefs if coefs.ndim == 1 else coefs[0]
            if importances is not None:
                imp_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
                imp_df = imp_df.sort_values("Importance", ascending=False)
                st.bar_chart(imp_df.set_index("Feature"))
            else:
                st.write("Feature importance not available for this model.")

            # SHAP values plot (matplotlib beeswarm/violin)
            st.subheader("SHAP Values Plot (Beeswarm/Violin, Matplotlib)")
            try:
                import shap
                # Try TreeExplainer for supported models
                explainer = None
                shap_values = None
                import numpy as np
                try:
                    explainer = shap.TreeExplainer(best_model)
                    shap_values = explainer.shap_values(X_test)
                except Exception:
                    st.info("TreeExplainer not supported for this model. Using KernelExplainer on a sample (may be slow).")
                    sample_X = X_test.sample(n=min(100, len(X_test)), random_state=42)
                    explainer = shap.KernelExplainer(lambda data: best_model.predict(np.array(data)), np.array(sample_X))
                    shap_values = explainer.shap_values(np.array(sample_X))
                    X_test = sample_X
                # Defensive: ensure shape matches and avoid feature_names_in_ issues
                if isinstance(shap_values, list):
                    shap_values = np.array(shap_values)
                    if shap_values.ndim == 3:
                        shap_values = shap_values[0]
                if shap_values.shape[1] == X_test.shape[1]:
                    # Matplotlib beeswarm/violin plot with numpy array and explicit feature names
                    feature_names = list(X_test.columns) if hasattr(X_test, 'columns') else [f'Feature {i}' for i in range(X_test.shape[1])]
                    X_plot = X_test.values if hasattr(X_test, 'values') else np.array(X_test)
                    fig, ax = plt.subplots(figsize=(8, min(0.5 * X_test.shape[1], 12)))
                    try:
                        shap.summary_plot(shap_values, X_plot, feature_names=feature_names, plot_type="violin", show=False, color_bar=True)
                    except Exception as e_violin:
                        st.warning(f"Violin plot failed: {e_violin}. Falling back to dot plot.")
                        shap.summary_plot(shap_values, X_plot, feature_names=feature_names, plot_type="dot", show=False, color_bar=True)
                    st.pyplot(fig)
                else:
                    st.warning(f"Could not display SHAP plot: SHAP values shape {shap_values.shape} does not match feature count {X_test.shape[1]}")
            except Exception as e:
                st.warning(f"Could not display SHAP plot: {e}")

            if modeling_option == "AutoML (FLAML)":
                st.subheader("Best Model Found")
                st.write(type(best_model).__name__)
                st.json(automl.best_config)

            if modeling_option == "AutoML (FLAML)":
                st.subheader("Top 5 FLAML Trials (by validation loss)")

                try:
                    # Safely access internal training history
                    history_list = getattr(automl._state, "history", [])
                    if history_list:
                        history_df = pd.DataFrame(history_list)
                        if 'metric_target' in history_df.columns:
                            top5_df = history_df.sort_values('metric_target').head(5)
                            top5_df = top5_df[['learner', 'metric_target', 'train_time']]
                            top5_df.columns = ['Model', 'Validation Loss', 'Train Time (s)']
                            st.dataframe(top5_df.reset_index(drop=True))
                        else:
                            st.warning("Column 'metric_target' not found in training history.")
                    else:
                        st.warning("FLAML training history is empty.")
                except Exception as e:
                    st.error(f"Error loading FLAML training history: {e}")
