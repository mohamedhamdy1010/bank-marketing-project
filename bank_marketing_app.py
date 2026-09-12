import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Bank Marketing Prediction",
    page_icon="🏦",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: bold;
        text-align: center;
    }

    .subtitle {
        font-size: 20px;
        text-align: center;
        color: gray;
        margin-bottom: 30px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    # The file is actually a CSV with ; separator
    df = pd.read_csv(
        "bank.xls",
        sep=";"
    )

    # Remove extra spaces from column names
    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    return df


try:

    df = load_data()

except Exception as e:

    st.error(
        f"❌ Error while reading bank.xls:\n\n{e}"
    )

    st.stop()


# ============================================================
# DISPLAY DATASET INFO
# ============================================================

# Remove completely empty columns
df = df.dropna(
    axis=1,
    how="all"
)


# ============================================================
# CHECK TARGET
# ============================================================

if "y" not in df.columns:

    st.error(
        "❌ Target column 'y' was not found."
    )

    st.write(
        "Columns found:"
    )

    st.write(
        df.columns.tolist()
    )

    st.stop()


# ============================================================
# CONVERT NUMERIC COLUMNS
# ============================================================

def convert_numeric_columns(data):

    data = data.copy()

    for column in data.columns:

        if column == "y":
            continue

        # Try converting the column to numeric
        converted = pd.to_numeric(
            data[column],
            errors="coerce"
        )

        # Calculate how much of the column is numeric
        numeric_ratio = converted.notna().mean()

        # If most values are numeric, keep it numeric
        if numeric_ratio >= 0.90:

            data[column] = converted

    return data


df = convert_numeric_columns(df)


# ============================================================
# PREPROCESSING
# ============================================================

@st.cache_data
def preprocess_data(data):

    data = data.copy()

    # -----------------------------
    # Target
    # -----------------------------

    data["y"] = (
        data["y"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "yes": 1,
                "no": 0
            }
        )
    )

    # Remove invalid target rows
    data = data.dropna(
        subset=["y"]
    )

    # -----------------------------
    # Features and target
    # -----------------------------

    X = data.drop(
        "y",
        axis=1
    )

    y = data["y"].astype(int)

    # -----------------------------
    # Handle missing values
    # -----------------------------

    for column in X.columns:

        if pd.api.types.is_numeric_dtype(
            X[column]
        ):

            X[column] = X[column].fillna(
                X[column].median()
            )

        else:

            X[column] = X[column].fillna(
                X[column].mode()[0]
            )

    # -----------------------------
    # Encode categorical columns
    # -----------------------------

    encoders = {}

    for column in X.columns:

        if not pd.api.types.is_numeric_dtype(
            X[column]
        ):

            encoder = LabelEncoder()

            X[column] = encoder.fit_transform(
                X[column].astype(str)
            )

            encoders[column] = encoder

    return X, y, encoders


X, y, encoders = preprocess_data(df)


# ============================================================
# TRAIN MODEL
# ============================================================

@st.cache_resource
def train_model(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    y_pred = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    return (
        model,
        X_train,
        X_test,
        y_train,
        y_test,
        y_pred,
        accuracy,
        precision,
        recall
    )


(
    model,
    X_train,
    X_test,
    y_train,
    y_test,
    y_pred,
    accuracy,
    precision,
    recall
) = train_model(
    X,
    y
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🏦 Bank Marketing")

st.sidebar.write(
    "Machine Learning Classification"
)

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📊 Data Analysis",
        "🤖 Model Performance",
        "🎯 Prediction"
    ]
)


# ============================================================
# HOME
# ============================================================

if page == "🏠 Home":

    st.markdown(
        '<div class="main-title">'
        '🏦 Bank Marketing Prediction'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Machine Learning Classification System'
        '</div>',
        unsafe_allow_html=True
    )

    st.divider()

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "👥 Customers",
            len(df)
        )

    with col2:

        st.metric(
            "📋 Features",
            df.shape[1] - 1
        )

    with col3:

        subscribed = (
            df["y"]
            .astype(str)
            .str.strip()
            .str.lower()
            .eq("yes")
            .sum()
        )

        st.metric(
            "✅ Subscribed",
            subscribed
        )

    with col4:

        st.metric(
            "🎯 Accuracy",
            f"{accuracy:.2%}"
        )

    st.divider()

    st.header(
        "📌 Problem Definition"
    )

    st.write(
        """
        The objective of this project is to predict whether a bank
        customer will subscribe to a term deposit after a marketing
        campaign.
        """
    )

    st.header(
        "💡 Machine Learning Solution"
    )

    st.write(
        """
        A Random Forest Classification model is used to learn patterns
        from customer information and predict the subscription decision.
        """
    )

    st.header(
        "📊 Dataset Preview"
    )

    st.dataframe(
        df.head(10),
        use_container_width=True
    )


# ============================================================
# DATA ANALYSIS
# ============================================================

elif page == "📊 Data Analysis":

    st.header(
        "📊 Exploratory Data Analysis"
    )

    # -----------------------------
    # Information
    # -----------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Rows",
            df.shape[0]
        )

    with col2:

        st.metric(
            "Columns",
            df.shape[1]
        )

    with col3:

        st.metric(
            "Missing Values",
            int(df.isnull().sum().sum())
        )

    st.divider()

    # -----------------------------
    # Dataset
    # -----------------------------

    st.subheader(
        "📋 Dataset"
    )

    st.dataframe(
        df,
        use_container_width=True
    )

    st.divider()

    # -----------------------------
    # Statistics
    # -----------------------------

    st.subheader(
        "📈 Statistical Summary"
    )

    st.dataframe(
        df.describe(include="all"),
        use_container_width=True
    )

    st.divider()

    # -----------------------------
    # Target Distribution
    # -----------------------------

    st.subheader(
        "🎯 Target Distribution"
    )

    target_counts = df["y"].value_counts()

    fig, ax = plt.subplots(
        figsize=(7, 4)
    )

    ax.bar(
        target_counts.index.astype(str),
        target_counts.values
    )

    ax.set_xlabel(
        "Subscription"
    )

    ax.set_ylabel(
        "Customers"
    )

    ax.set_title(
        "Term Deposit Subscription"
    )

    st.pyplot(fig)

    st.divider()

    # -----------------------------
    # Correlation
    # -----------------------------

    st.subheader(
        "🔥 Correlation Heatmap"
    )

    numeric_df = df.select_dtypes(
        include=np.number
    )

    if numeric_df.shape[1] > 1:

        fig, ax = plt.subplots(
            figsize=(12, 7)
        )

        sns.heatmap(
            numeric_df.corr(),
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            ax=ax
        )

        ax.set_title(
            "Feature Correlation"
        )

        st.pyplot(fig)

    else:

        st.info(
            "Not enough numerical features."
        )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "🤖 Model Performance":

    st.header(
        "🤖 Random Forest Model Performance"
    )

    st.write(
        """
        The dataset was split into 80% training data and
        20% testing data.
        """
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Accuracy",
            f"{accuracy:.2%}"
        )

    with col2:

        st.metric(
            "Precision",
            f"{precision:.2%}"
        )

    with col3:

        st.metric(
            "Recall",
            f"{recall:.2%}"
        )

    st.divider()

    # -----------------------------
    # Confusion Matrix
    # -----------------------------

    st.subheader(
        "📊 Confusion Matrix"
    )

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No", "Yes"],
        yticklabels=["No", "Yes"],
        ax=ax
    )

    ax.set_xlabel(
        "Predicted"
    )

    ax.set_ylabel(
        "Actual"
    )

    st.pyplot(fig)

    st.divider()

    # -----------------------------
    # Classification Report
    # -----------------------------

    st.subheader(
        "📋 Classification Report"
    )

    report = classification_report(
        y_test,
        y_pred,
        target_names=[
            "No",
            "Yes"
        ],
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(
        report
    ).transpose()

    st.dataframe(
        report_df,
        use_container_width=True
    )

    st.divider()

    # -----------------------------
    # Feature Importance
    # -----------------------------

    st.subheader(
        "⭐ Feature Importance"
    )

    importance_df = pd.DataFrame(
        {
            "Feature": X.columns,
            "Importance": model.feature_importances_
        }
    )

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.barh(
        importance_df["Feature"],
        importance_df["Importance"]
    )

    ax.set_xlabel(
        "Importance"
    )

    ax.set_ylabel(
        "Feature"
    )

    ax.set_title(
        "Random Forest Feature Importance"
    )

    ax.invert_yaxis()

    st.pyplot(fig)


# ============================================================
# PREDICTION
# ============================================================

elif page == "🎯 Prediction":

    st.header(
        "🎯 Customer Subscription Prediction"
    )

    st.write(
        """
        Enter customer information below to predict whether
        the customer is likely to subscribe to a term deposit.
        """
    )

    st.divider()

    user_input = {}

    # -----------------------------
    # Input Fields
    # -----------------------------

    for column in df.drop(
        "y",
        axis=1
    ).columns:

        # Check if numeric
        if pd.api.types.is_numeric_dtype(
            df[column]
        ):

            # Safe numeric median
            median_value = df[column].median()

            if pd.isna(median_value):

                median_value = 0.0

            user_input[column] = st.number_input(
                column,
                value=float(median_value)
            )

        else:

            # Categorical column
            options = (
                df[column]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            if len(options) > 0:

                user_input[column] = st.selectbox(
                    column,
                    options
                )

            else:

                user_input[column] = ""

    st.divider()

    # -----------------------------
    # Prediction Button
    # -----------------------------

    if st.button(
        "🔮 Predict Subscription",
        use_container_width=True
    ):

        input_df = pd.DataFrame(
            [user_input]
        )

        # -------------------------
        # Convert input values
        # -------------------------

        for column in input_df.columns:

            if column in encoders:

                encoder = encoders[column]

                value = str(
                    input_df[column].iloc[0]
                )

                if value in encoder.classes_:

                    input_df[column] = encoder.transform(
                        [value]
                    )

                else:

                    st.error(
                        f"Unknown category in {column}"
                    )

                    st.stop()

        # -------------------------
        # Feature order
        # -------------------------

        input_df = input_df[
            X.columns
        ]

        # -------------------------
        # Prediction
        # -------------------------

        prediction = model.predict(
            input_df
        )[0]

        probability = model.predict_proba(
            input_df
        )[0][1]

        st.divider()

        st.subheader(
            "Prediction Result"
        )

        if prediction == 1:

            st.success(
                "✅ The customer is likely to subscribe "
                "to a term deposit."
            )

        else:

            st.warning(
                "❌ The customer is unlikely to subscribe "
                "to a term deposit."
            )

        st.metric(
            "Subscription Probability",
            f"{probability:.2%}"
        )

        st.progress(
            float(probability)
        )


# ============================================================
# FOOTER
# ============================================================

st.sidebar.divider()

st.sidebar.caption(
    "Bank Marketing ML Project"
)

st.sidebar.caption(
    "Random Forest Classification"
)