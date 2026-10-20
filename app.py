"""
Intelligent Supply Chain Analytics and Shipment Delay Prediction System
Experiment 10 | Intelligent Data Systems-1 Lab | Streamlit app

Upload a permitted CSV dataset to run the analytics. The application does not bundle
or publish the DataCo dataset. Sensitive customer-identifying fields are excluded from
on-screen descriptive summaries and are not used as model predictors.
"""
from pathlib import Path
import heapq
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from scipy.stats import ttest_ind
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, mean_absolute_error,
    mean_squared_error, r2_score, silhouette_score
)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

st.set_page_config(page_title="Intelligent Supply Chain Analytics", page_icon="🚚", layout="wide")


ALIASES = {
    "late_delivery_risk": ["Late_delivery_risk", "Late Delivery Risk", "late_delivery_risk"],
    "shipping_mode": ["Shipping Mode", "Shipping_Mode", "shipping_mode"],
    "scheduled_days": ["Days for shipment (scheduled)", "Days_for_shipment_scheduled", "scheduled_days"],
    "actual_days": ["Days for shipping (real)", "Days_for_shipping_real", "actual_days"],
    "sales": ["Sales", "Sales per customer", "Sales_per_customer", "sales"],
    "quantity": ["Order Item Quantity", "Order_Item_Quantity", "quantity"],
    "product_price": ["Order Item Product Price", "Product Price", "Order_Item_Product_Price", "product_price"],
    "discount_rate": ["Order Item Discount Rate", "Order_Item_Discount_Rate", "discount_rate"],
    "profit_ratio": ["Order Item Profit Ratio", "Order_Item_Profit_Ratio", "profit_ratio"],
    "market": ["Market", "market"],
    "region": ["Order Region", "Order_Region", "region"],
    "segment": ["Customer Segment", "Customer_Segment", "segment"],
    "category": ["Category Name", "Category_Name", "category"],
    "department": ["Department Name", "Department_Name", "department"],
    "order_date": ["order date (DateOrders)", "Order Date", "Order_Date", "order_date"],
    "order_year": ["OrderYear", "order_year"],
    "order_month": ["OrderMonth", "order_month"],
    "order_weekday": ["OrderDayOfWeek", "order_weekday"],
    "latitude": ["Latitude", "latitude"],
    "longitude": ["Longitude", "longitude"],
    "country": ["Order Country", "Order_Country", "country"],
    "pay_type": ["Type", "Payment Type", "pay_type"],
    "ship_date": ["shipping date (DateOrders)", "Shipping Date", "ship_date"],
}

def find_col(df, logical_name):
    for name in ALIASES.get(logical_name, []):
        if name in df.columns:
            return name
    return None

def standardize_columns(df):
    """Create standard internal columns without deleting original columns."""
    out = df.copy()
    for logical, aliases in ALIASES.items():
        found = find_col(out, logical)
        if found and logical not in out.columns:
            out[logical] = out[found]
    if "order_date" in out.columns:
        parsed = pd.to_datetime(out["order_date"], errors="coerce")
        if "order_year" not in out.columns:
            out["order_year"] = parsed.dt.year
        if "order_month" not in out.columns:
            out["order_month"] = parsed.dt.month
        if "order_weekday" not in out.columns:
            out["order_weekday"] = parsed.dt.dayofweek
        if "order_hour" not in out.columns:
            out["order_hour"] = parsed.dt.hour
        # Days between order and the recorded shipping date (only known AFTER dispatch).
        if "ship_date" in out.columns:
            ship_parsed = pd.to_datetime(out["ship_date"], errors="coerce")
            out["ship_lag_days"] = (ship_parsed - parsed).dt.total_seconds() / 86400
    return out

def load_uploaded(uploaded):
    """Load only a user-supplied CSV; never silently substitute sample/demo data."""
    if uploaded is None:
        return None
    try:
        uploaded.seek(0)
        return pd.read_csv(uploaded, encoding="latin-1", low_memory=False)
    except Exception:
        uploaded.seek(0)
        return pd.read_csv(uploaded, low_memory=False)

def clean_data(df):
    """Conservative cleaning: standardize fields, remove duplicate rows, coerce known numerics."""
    data = standardize_columns(df)
    before = len(data)
    duplicate_count = int(data.duplicated().sum())
    data = data.drop_duplicates().copy()
    numeric_logicals = ["late_delivery_risk", "scheduled_days", "actual_days", "sales",
                        "quantity", "product_price", "discount_rate", "profit_ratio",
                        "order_year", "order_month", "order_weekday",
                        "latitude", "longitude", "order_hour", "ship_lag_days"]
    for col in numeric_logicals:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    for col in data.select_dtypes(include="object").columns:
        data[col] = data[col].astype("string").str.strip()
        data[col] = data[col].replace({"": pd.NA, "NA": pd.NA, "N/A": pd.NA, "nan": pd.NA})
    return data, before, duplicate_count

def choose_first(data, candidates):
    return next((c for c in candidates if c in data.columns), None)

def make_features(data, target):
    # Deliberately excludes actual shipping duration and delivery-status fields to reduce target leakage.
    numeric_candidates = ["scheduled_days", "quantity", "product_price", "discount_rate",
                          "profit_ratio", "sales", "order_year", "order_month", "order_weekday"]
    categorical_candidates = ["shipping_mode", "market", "region", "segment", "category", "department"]
    numeric = [c for c in numeric_candidates if c in data.columns and c != target]
    categorical = [c for c in categorical_candidates if c in data.columns and c != target]
    features = numeric + categorical
    if not features:
        return None, None, None, None
    model_data = data[features + [target]].copy()
    model_data[target] = pd.to_numeric(model_data[target], errors="coerce")
    model_data = model_data.dropna(subset=[target])
    # Exclude impossible / non-binary target values for classification.
    if target == "late_delivery_risk":
        model_data = model_data[model_data[target].isin([0, 1])]
    X = model_data[features]
    y = model_data[target]
    num = [c for c in numeric if c in X.columns]
    cat = [c for c in categorical if c in X.columns]
    return X, y, num, cat

def build_preprocessor(num_cols, cat_cols):
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")),
                             ("scaler", StandardScaler())])
    categorical_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                 ("encoder", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("numeric", numeric_pipe, num_cols),
                              ("categorical", categorical_pipe, cat_cols)],
                             remainder="drop")

def make_classification_features(data, target, use_shipping_date=False):
    """Features for the late-delivery classifier.

    Default (use_shipping_date=False): only information available when the order is placed.
    use_shipping_date=True adds ship_lag_days (shipping date minus order date). In DataCo this
    column reproduces the realised shipping time almost exactly, so it leaks the outcome. It is
    offered only as a clearly-labelled post-dispatch mode.
    """
    numeric_candidates = ["scheduled_days", "quantity", "product_price", "discount_rate",
                          "profit_ratio", "sales", "order_year", "order_month", "order_weekday",
                          "order_hour", "latitude", "longitude"]
    categorical_candidates = ["shipping_mode", "market", "region", "segment", "category",
                              "department", "country", "pay_type"]
    if use_shipping_date:
        numeric_candidates.append("ship_lag_days")
    numeric = [c for c in numeric_candidates if c in data.columns and c != target]
    # HistGradientBoosting supports at most 255 categories per feature.
    categorical = [c for c in categorical_candidates
                   if c in data.columns and c != target and data[c].nunique(dropna=True) <= 255]
    features = numeric + categorical
    if not features:
        return None, None, None, None
    model_data = data[features + [target]].copy()
    model_data[target] = pd.to_numeric(model_data[target], errors="coerce")
    model_data = model_data.dropna(subset=[target])
    model_data = model_data[model_data[target].isin([0, 1])]
    for c in categorical:
        model_data[c] = model_data[c].astype(object).where(model_data[c].notna(), "Missing").astype(str)
    return model_data[features], model_data[target].astype(int), numeric, categorical

def build_boosted_classifier(num_cols, cat_cols):
    """Histogram gradient boosting: captures non-linear / interaction effects (e.g. shipping mode x
    schedule x location x time) that a linear Logistic Regression cannot."""
    pre = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median"), num_cols),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)),
        ]), cat_cols),
    ], remainder="drop")
    model = HistGradientBoostingClassifier(
        categorical_features=[False] * len(num_cols) + [True] * len(cat_cols),
        max_iter=300, learning_rate=0.08, random_state=42)
    return Pipeline([("preprocess", pre), ("model", model)])

def a_star(graph, start, goal, heuristic):
    """A* shortest path for the illustrative network in the app."""
    open_heap = [(heuristic.get(start, 0), 0, start)]
    came_from = {}
    g_score = {start: 0}
    closed = set()
    while open_heap:
        _, current_cost, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1], current_cost
        closed.add(current)
        for neighbour, edge_cost in graph.get(current, {}).items():
            tentative = g_score[current] + edge_cost
            if tentative < g_score.get(neighbour, float("inf")):
                came_from[neighbour] = current
                g_score[neighbour] = tentative
                heapq.heappush(open_heap, (tentative + heuristic.get(neighbour, 0),
                                           tentative, neighbour))
    return [], float("inf")

st.title("🚚 Intelligent Supply Chain Analytics")
st.caption("Shipment risk prediction • sales regression • statistical analysis • K-Means • PCA • shortest-path planning")
st.write("An interactive analytics application for exploring supply-chain data, evaluating baseline machine-learning models, and demonstrating core Intelligent Data Systems lab methods.")

with st.sidebar:
    st.header("Dataset")
    uploaded = st.file_uploader("Upload a supply-chain CSV", type=["csv"], help="Upload a dataset you have permission to use. DataCoSupplyChainDataset.csv is supported.")
    st.caption("CSV files are processed for this session. Do not upload confidential customer or business data to a public deployment.")
    st.markdown("[DataCo dataset source on Kaggle](https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis)")
    st.divider()
    st.markdown("**Included methods**")
    st.markdown("1. Data cleaning and EDA\n2. Descriptive statistics and t-test\n3. Logistic Regression\n4. Linear Regression\n5. K-Means clustering\n6. PCA comparison\n7. A* shortest-path algorithm")

raw = load_uploaded(uploaded)
if raw is None:
    st.info("To begin, upload a CSV using the Dataset panel on the left. No synthetic or demo dataset is bundled or loaded automatically.")
    st.markdown("**Recommended dataset:** DataCo Smart Supply Chain for Big Data Analysis. After downloading the CSV, upload it above to unlock the analytics tabs.")
    st.stop()

if raw.empty:
    st.error("The uploaded CSV contains no records. Please upload a valid CSV file.")
    st.stop()

data, original_rows, duplicate_count = clean_data(raw)

# Hide personal identifiers and credentials from any displayed data summaries.
def is_sensitive_or_identifier(column):
    name = str(column).strip().lower()
    tokens = ["password", "email", "e-mail", "customer fname", "customer lname", "customer name",
              "first name", "last name", "street", "address", "zipcode", "zip code", "postal code",
              "customer id", "order customer id", "customer id", "order id", "order item id",
              "transaction id", "tracking number", "card number"]
    return any(token in name for token in tokens) or name.endswith(" id") or name == "id"

safe_columns = [c for c in data.columns if not is_sensitive_or_identifier(c)]
with st.expander("Dataset quality checks", expanded=False):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows loaded", f"{original_rows:,}")
    c2.metric("Rows after duplicate removal", f"{len(data):,}")
    c3.metric("Duplicate rows removed", f"{duplicate_count:,}")
    c4.metric("Columns loaded", f"{data.shape[1]:,}")
    st.caption("Customer identifiers, contact details, addresses, and credential-like fields are not displayed in the quality summary.")
    safe_meta = pd.DataFrame({"Column": safe_columns,
                              "Data type": [str(data[c].dtype) for c in safe_columns],
                              "Missing values": [int(data[c].isna().sum()) for c in safe_columns]})
    st.dataframe(safe_meta, use_container_width=True, hide_index=True)

# Main summary KPIs
risk_col = "late_delivery_risk" if "late_delivery_risk" in data.columns else None
sales_col = "sales" if "sales" in data.columns else None
c1, c2, c3, c4 = st.columns(4)
c1.metric("Dataset records", f"{len(data):,}")
if risk_col:
    risk = pd.to_numeric(data[risk_col], errors="coerce")
    c2.metric("Late-risk rate", f"{risk.mean()*100:.1f}%" if risk.notna().any() else "N/A")
else:
    c2.metric("Late-risk rate", "Target not found")
if sales_col:
    sales = pd.to_numeric(data[sales_col], errors="coerce")
    c3.metric("Total sales (sum)", f"{sales.sum():,.0f}")
    c4.metric("Average sales", f"{sales.mean():,.2f}")
else:
    c3.metric("Total sales", "N/A")
    c4.metric("Average sales", "N/A")

tab_names = ["Overview & EDA", "Statistics", "Delay Prediction", "Sales Regression",
             "K-Means Segmentation", "PCA Analysis", "A* Route Planner"]
tabs = st.tabs(tab_names)

with tabs[0]:
    st.subheader("Exploratory Data Analysis")
    numeric_cols = [c for c in data.select_dtypes(include=np.number).columns.tolist() if not is_sensitive_or_identifier(c)]
    if numeric_cols:
        selected_num = st.selectbox("Choose a numeric column for distribution", numeric_cols,
                                    index=0, key="eda_numeric")
        fig, ax = plt.subplots()
        ax.hist(pd.to_numeric(data[selected_num], errors="coerce").dropna(), bins=30)
        ax.set_title(f"Distribution of {selected_num}")
        ax.set_xlabel(selected_num); ax.set_ylabel("Frequency")
        st.pyplot(fig); plt.close(fig)
    group_col = choose_first(data, ["shipping_mode", "market", "region", "category", "department"])
    if group_col:
        st.write(f"Record count by **{group_col}**")
        counts = data[group_col].astype("string").fillna("Missing").value_counts().head(15)
        fig, ax = plt.subplots()
        counts.sort_values().plot(kind="barh", ax=ax)
        ax.set_xlabel("Number of records"); ax.set_ylabel(group_col)
        st.pyplot(fig); plt.close(fig)
    if sales_col and risk_col:
        plot_df = data[["sales", "late_delivery_risk"]].copy()
        plot_df["sales"] = pd.to_numeric(plot_df["sales"], errors="coerce")
        plot_df["late_delivery_risk"] = pd.to_numeric(plot_df["late_delivery_risk"], errors="coerce")
        plot_df = plot_df.dropna()
        if len(plot_df):
            fig, ax = plt.subplots()
            plot_df.boxplot(column="sales", by="late_delivery_risk", ax=ax)
            ax.set_title("Sales distribution by late-delivery risk")
            ax.set_xlabel("Late delivery risk (0 = no, 1 = yes)")
            ax.set_ylabel("Sales")
            fig.suptitle("")
            st.pyplot(fig); plt.close(fig)
    if len(numeric_cols) >= 2:
        st.write("Correlation heatmap (numeric features)")
        corr = data[numeric_cols[:15]].corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(9, 5))
        im = ax.imshow(corr, aspect="auto")
        ax.set_xticks(range(len(corr.columns))); ax.set_xticklabels(corr.columns, rotation=75, ha="right")
        ax.set_yticks(range(len(corr.index))); ax.set_yticklabels(corr.index)
        fig.colorbar(im, ax=ax, shrink=.8)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    st.caption("Interpret charts as descriptive evidence. Associations do not by themselves establish causation.")

with tabs[1]:
    st.subheader("Statistical analysis and hypothesis test")
    available_numeric = [c for c in data.select_dtypes(include=np.number).columns.tolist() if not is_sensitive_or_identifier(c)]
    if len(available_numeric) >= 2:
        stat_col1 = st.selectbox("Numeric variable A", available_numeric, index=0, key="stat_a")
        stat_col2 = st.selectbox("Numeric variable B", available_numeric,
                                 index=min(1, len(available_numeric)-1), key="stat_b")
        a = pd.to_numeric(data[stat_col1], errors="coerce")
        b = pd.to_numeric(data[stat_col2], errors="coerce")
        left, right = st.columns(2)
        with left:
            st.markdown("**Descriptive statistics**")
            st.write(pd.DataFrame({
                "Measure": ["Mean", "Median", "Variance", "Standard deviation"],
                stat_col1: [a.mean(), a.median(), a.var(), a.std()],
                stat_col2: [b.mean(), b.median(), b.var(), b.std()]
            }).round(4))
        with right:
            pair = pd.concat([a, b], axis=1).dropna()
            st.metric("Pearson correlation", f"{pair.corr().iloc[0,1]:.3f}" if len(pair)>2 else "N/A")
            st.metric("Covariance", f"{pair.cov().iloc[0,1]:.3f}" if len(pair)>2 else "N/A")
    if risk_col and sales_col:
        st.markdown("**Welch independent t-test: sales for late-risk vs non-late-risk records**")
        tmp = data[[risk_col, sales_col]].copy()
        tmp[risk_col] = pd.to_numeric(tmp[risk_col], errors="coerce")
        tmp[sales_col] = pd.to_numeric(tmp[sales_col], errors="coerce")
        g0 = tmp.loc[tmp[risk_col] == 0, sales_col].dropna()
        g1 = tmp.loc[tmp[risk_col] == 1, sales_col].dropna()
        if len(g0) >= 2 and len(g1) >= 2:
            tstat, pvalue = ttest_ind(g0, g1, equal_var=False, nan_policy="omit")
            st.write(f"Group 0 mean sales: **{g0.mean():.2f}** (n={len(g0):,})")
            st.write(f"Group 1 mean sales: **{g1.mean():.2f}** (n={len(g1):,})")
            st.write(f"t-statistic = **{tstat:.4f}**, p-value = **{pvalue:.6g}**")
            st.write("At α = 0.05: " + ("Reject the null hypothesis of equal group means." if pvalue < .05 else "Fail to reject the null hypothesis of equal group means."))
            st.caption("This is an association test, not evidence that late risk causes a sales difference.")
        else:
            st.warning("Not enough records in both target groups to run the t-test.")
    else:
        st.info("The t-test section needs both sales and Late_delivery_risk columns.")

with tabs[2]:
    st.subheader("Shipment late-delivery risk prediction")
    if risk_col:
        mode_options = ["At order placement (no leakage)"]
        if "ship_lag_days" in data.columns:
            mode_options.append("After dispatch (uses shipping date, leaks outcome)")
        pred_mode = st.radio("Prediction stage", mode_options, key="cls_mode",
                             help="The shipping-date column is only known after the parcel has moved, and in DataCo it "
                                  "reproduces the real shipping time, which defines the target.")
        use_ship_date = pred_mode.startswith("After dispatch")
        if use_ship_date:
            st.warning("This mode includes the shipping date, which effectively reveals the outcome. High accuracy here "
                       "is not a genuine forecast of delay at order time. Report it as a post-dispatch result only.")
        X, y, num_cols, cat_cols = make_classification_features(data, "late_delivery_risk", use_ship_date)
        if X is None or len(y) < 20 or y.nunique() < 2:
            st.warning("Not enough suitable features/target classes for classification. Upload the DataCo dataset or a compatible CSV.")
        else:
            test_size = st.slider("Test set (%)", 10, 40, 20, key="cls_test") / 100
            threshold = st.slider(
                "Late-delivery risk threshold", min_value=0.30, max_value=0.70,
                value=0.50, step=0.05, key="cls_threshold",
                help="0.50 maximises overall accuracy. Lower thresholds detect more at-risk shipments but produce more false alarms."
            )
            if st.button("Train Gradient Boosting classifier", key="train_cls"):
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42, stratify=y)
                pipe = build_boosted_classifier(num_cols, cat_cols)
                with st.spinner("Training and evaluating..."):
                    pipe.fit(X_train, y_train)
                    probabilities = pipe.predict_proba(X_test)[:, 1]
                    pred = (probabilities >= threshold).astype(int)
                st.session_state["classification_model"] = pipe
                st.session_state["classification_features"] = list(X.columns)
                st.session_state["classification_num"] = num_cols
                st.session_state["classification_cat"] = cat_cols
                st.session_state["classification_data_columns"] = list(data.columns)
                st.session_state["classification_threshold"] = threshold
                st.session_state["classification_metrics"] = {
                    "Accuracy": accuracy_score(y_test, pred),
                    "Precision": precision_score(y_test, pred, zero_division=0),
                    "Recall": recall_score(y_test, pred, zero_division=0),
                    "F1-score": f1_score(y_test, pred, zero_division=0)
                }
                st.session_state["classification_cm"] = confusion_matrix(y_test, pred, labels=[0, 1]).tolist()
                st.session_state["classification_report"] = classification_report(y_test, pred, labels=[0, 1], zero_division=0)
            if "classification_metrics" in st.session_state:
                cols = st.columns(4)
                for col, (metric, value) in zip(cols, st.session_state["classification_metrics"].items()):
                    col.metric(metric, f"{value:.3f}")
                st.write("Confusion matrix (rows = actual, columns = predicted)")
                st.dataframe(pd.DataFrame(st.session_state["classification_cm"],
                                          index=["Actual 0", "Actual 1"],
                                          columns=["Predicted 0", "Predicted 1"]))
                st.code(st.session_state["classification_report"])
                st.caption("Predictors exclude actual shipping duration and delivery/order status fields. The shipping date is used only in the labelled post-dispatch mode.")
                st.caption(f"Evaluation threshold: {st.session_state.get('classification_threshold', 0.50):.2f}. Retrain to apply a changed threshold to evaluation metrics.")
                st.markdown("**Try a new shipment**")
                user_input = {}
                model = st.session_state["classification_model"]
                feats = st.session_state["classification_features"]
                for feat in feats:
                    if feat in st.session_state["classification_num"]:
                        vals = pd.to_numeric(data[feat], errors="coerce").dropna()
                        default = float(vals.median()) if len(vals) else 0.0
                        user_input[feat] = st.number_input(feat, value=default, key=f"pred_{feat}")
                    else:
                        opts = sorted(data[feat].dropna().astype(str).unique().tolist())
                        user_input[feat] = st.selectbox(feat, opts, key=f"pred_cat_{feat}") if opts else "Unknown"
                if st.button("Predict shipment risk", key="predict_one"):
                    row = pd.DataFrame([user_input], columns=feats)
                    probability = float(model.predict_proba(row)[0][1]) if hasattr(model, "predict_proba") else float("nan")
                    active_threshold = float(st.session_state.get("classification_threshold", 0.50))
                    pred_class = int(probability >= active_threshold)
                    st.success(f"Predicted class: {'HIGHER LATE-DELIVERY RISK' if pred_class == 1 else 'LOWER LATE-DELIVERY RISK'}")
                    st.write(f"Estimated probability of class 1: **{probability:.1%}**")
                    st.caption(f"Classification threshold used: {active_threshold:.2f}")
    else:
        st.warning("Target column Late_delivery_risk not found.")

with tabs[3]:
    st.subheader("Sales estimation using Linear Regression")
    if sales_col:
        X, y, num_cols, cat_cols = make_features(data, "sales")
        # Avoid target as input and avoid realized delivery days; use order and shipping descriptors.
        if X is not None and len(y) >= 20:
            if st.button("Train sales regression model", key="train_reg"):
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
                pipe = Pipeline([("preprocess", build_preprocessor(num_cols, cat_cols)),
                                 ("model", LinearRegression())])
                pipe.fit(X_train, y_train)
                pred = pipe.predict(X_test)
                st.session_state["reg_metrics"] = {
                    "MAE": mean_absolute_error(y_test, pred),
                    "MSE": mean_squared_error(y_test, pred),
                    "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
                    "R²": r2_score(y_test, pred)
                }
                st.session_state["reg_model"] = pipe
                st.session_state["reg_features"] = list(X.columns)
                st.session_state["reg_num"] = num_cols
            if "reg_metrics" in st.session_state:
                cols = st.columns(4)
                for col, (metric, value) in zip(cols, st.session_state["reg_metrics"].items()):
                    col.metric(metric, f"{value:,.3f}")
                st.caption("Interpret R² together with MAE/RMSE; a negative R² indicates performance worse than a constant-mean baseline.")
                model = st.session_state["reg_model"]
                feats = st.session_state["reg_features"]
                row = {}
                for feat in feats:
                    if feat in st.session_state["reg_num"]:
                        vals = pd.to_numeric(data[feat], errors="coerce").dropna()
                        row[feat] = st.number_input(f"Sales input — {feat}", value=float(vals.median()) if len(vals) else 0.0, key=f"reg_{feat}")
                    else:
                        opts = sorted(data[feat].dropna().astype(str).unique().tolist())
                        row[feat] = st.selectbox(f"Sales input — {feat}", opts, key=f"reg_cat_{feat}") if opts else "Unknown"
                if st.button("Estimate sales", key="predict_sales"):
                    estimate = float(model.predict(pd.DataFrame([row], columns=feats))[0])
                    st.success(f"Estimated sales: {estimate:,.2f}")
        else:
            st.warning("Not enough usable rows/features for regression.")
    else:
        st.warning("Sales column not found. DataCo includes a Sales field.")

with tabs[4]:
    st.subheader("K-Means order segmentation")
    candidates = [c for c in ["scheduled_days", "quantity", "product_price", "discount_rate", "profit_ratio", "sales"] if c in data.columns]
    if len(candidates) >= 2:
        selected = st.multiselect("Choose numeric features for clustering", candidates, default=candidates[:min(4, len(candidates))])
        k = st.slider("Number of clusters (K)", 2, 8, 3, key="kmeans_k")
        if len(selected) >= 2 and st.button("Run K-Means", key="run_kmeans"):
            work = data[selected].apply(pd.to_numeric, errors="coerce")
            work = work.replace([np.inf, -np.inf], np.nan)
            work = work.fillna(work.median()).dropna()
            if len(work) > 10000:
                work = work.sample(10000, random_state=42)
            scaled = StandardScaler().fit_transform(work)
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(scaled)
            st.metric("Silhouette score", f"{silhouette_score(scaled, labels):.3f}" if len(np.unique(labels)) > 1 and len(work) > k else "N/A")
            counts = pd.Series(labels).value_counts().sort_index()
            st.write("Cluster sizes")
            st.dataframe(counts.rename("Records").to_frame())
            if len(selected) >= 2:
                fig, ax = plt.subplots()
                scatter = ax.scatter(work[selected[0]], work[selected[1]], c=labels, s=12, alpha=.65)
                ax.set_xlabel(selected[0]); ax.set_ylabel(selected[1])
                ax.set_title("K-Means clusters (first two selected features)")
                st.pyplot(fig); plt.close(fig)
            st.write("Cluster centroids (original feature units)")
            centers = pd.DataFrame(StandardScaler().fit(work).inverse_transform(km.cluster_centers_), columns=selected)
            st.dataframe(centers.round(3))
    else:
        st.info("At least two numeric logistics features are needed for clustering.")

with tabs[5]:
    st.subheader("PCA dimensionality reduction and model comparison")
    if risk_col:
        X, y, num_cols, cat_cols = make_features(data, "late_delivery_risk")
        if X is not None and len(y) >= 30 and y.nunique() > 1:
            if st.button("Compare Logistic Regression with and without PCA", key="run_pca"):
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2,
                                                                    random_state=42, stratify=y)
                prep = build_preprocessor(num_cols, cat_cols)
                Xt = prep.fit_transform(X_train)
                Xv = prep.transform(X_test)
                # Avoid converting high-cardinality one-hot matrices to dense arrays.
                if Xt.shape[1] > 100:
                    # One-hot cardinality guard: use numeric columns for the PCA illustration.
                    pca_num = [c for c in num_cols if c in X.columns]
                    if len(pca_num) < 2:
                        st.warning("PCA needs at least two numeric features.")
                    else:
                        xtr = X_train[pca_num].apply(pd.to_numeric, errors="coerce")
                        xte = X_test[pca_num].apply(pd.to_numeric, errors="coerce")
                        med = xtr.median()
                        scaler = StandardScaler()
                        xtr_s = scaler.fit_transform(xtr.fillna(med))
                        xte_s = scaler.transform(xte.fillna(med))
                        pca = PCA(n_components=2, random_state=42)
                        pca_tr, pca_te = pca.fit_transform(xtr_s), pca.transform(xte_s)
                        base_model = LogisticRegression(max_iter=1500, class_weight="balanced").fit(xtr_s, y_train)
                        pca_model = LogisticRegression(max_iter=1500, class_weight="balanced").fit(pca_tr, y_train)
                        base_pred, pca_pred = base_model.predict(xte_s), pca_model.predict(pca_te)
                        st.write(f"Numeric-only PCA explained variance retained: **{pca.explained_variance_ratio_.sum():.1%}**")
                        st.write(pd.DataFrame([
                            {"Model": "Original standardized numeric features", "Accuracy": accuracy_score(y_test, base_pred),
                             "Precision": precision_score(y_test, base_pred, zero_division=0), "Recall": recall_score(y_test, base_pred, zero_division=0),
                             "F1": f1_score(y_test, base_pred, zero_division=0)},
                            {"Model": "PCA (2 components)", "Accuracy": accuracy_score(y_test, pca_pred),
                             "Precision": precision_score(y_test, pca_pred, zero_division=0), "Recall": recall_score(y_test, pca_pred, zero_division=0),
                             "F1": f1_score(y_test, pca_pred, zero_division=0)}
                        ]).round(3))
                        fig, ax = plt.subplots()
                        ax.scatter(pca_tr[:, 0], pca_tr[:, 1], c=pd.Series(y_train).astype(int), s=12, alpha=.65)
                        ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_title("Training data projected onto first two PCs")
                        st.pyplot(fig); plt.close(fig)
                else:
                    Xt_dense = Xt.toarray() if hasattr(Xt, "toarray") else Xt
                    Xv_dense = Xv.toarray() if hasattr(Xv, "toarray") else Xv
                    pca = PCA(n_components=2, random_state=42)
                    pca_tr, pca_te = pca.fit_transform(Xt_dense), pca.transform(Xv_dense)
                    base_model = LogisticRegression(max_iter=1500, class_weight="balanced").fit(Xt_dense, y_train)
                    pca_model = LogisticRegression(max_iter=1500, class_weight="balanced").fit(pca_tr, y_train)
                    base_pred, pca_pred = base_model.predict(Xv_dense), pca_model.predict(Xv_dense)
                    st.write(f"PCA explained variance retained: **{pca.explained_variance_ratio_.sum():.1%}**")
                    st.write(pd.DataFrame([
                        {"Model": "Original features", "Accuracy": accuracy_score(y_test, base_pred),
                         "Precision": precision_score(y_test, base_pred, zero_division=0), "Recall": recall_score(y_test, base_pred, zero_division=0),
                         "F1": f1_score(y_test, base_pred, zero_division=0)},
                        {"Model": "PCA (2 components)", "Accuracy": accuracy_score(y_test, pca_pred),
                         "Precision": precision_score(y_test, pca_pred, zero_division=0), "Recall": recall_score(y_test, pca_pred, zero_division=0),
                         "F1": f1_score(y_test, pca_pred, zero_division=0)}
                    ]).round(3))
        else:
            st.warning("Insufficient target classes or rows for PCA comparison.")
    else:
        st.warning("Late_delivery_risk target not found.")

with tabs[6]:
    st.subheader("A* shortest-path planner — sample logistics graph")
    st.warning("The graph uses illustrative edge costs to demonstrate A* pathfinding; it is not connected to live maps or road-distance services.")
    graph = {
        "Warehouse": {"Mumbai": 4, "Pune": 6},
        "Mumbai": {"Warehouse": 4, "Nashik": 5, "Surat": 8},
        "Pune": {"Warehouse": 6, "Nashik": 3, "Hyderabad": 9},
        "Nashik": {"Mumbai": 5, "Pune": 3, "Surat": 4, "Indore": 7},
        "Surat": {"Mumbai": 8, "Nashik": 4, "Ahmedabad": 3},
        "Ahmedabad": {"Surat": 3, "Indore": 6},
        "Indore": {"Nashik": 7, "Ahmedabad": 6, "Hyderabad": 8},
        "Hyderabad": {"Pune": 9, "Indore": 8, "Chennai": 7},
        "Chennai": {"Hyderabad": 7},
    }
    # Admissible zero heuristic reduces A* to uniform-cost search while still demonstrating f(n)=g(n)+h(n).
    nodes = sorted(graph)
    start = st.selectbox("Origin", nodes, index=nodes.index("Warehouse"), key="astar_start")
    goal = st.selectbox("Destination", nodes, index=nodes.index("Chennai"), key="astar_goal")
    if st.button("Find shortest path with A*", key="astar_go"):
        path, cost = a_star(graph, start, goal, {n: 0 for n in nodes})
        if path:
            st.success(" → ".join(path))
            st.metric("Total illustrative path cost", cost)
            st.write("Path cost is the sum of the edge weights. The zero heuristic is admissible and keeps the result optimal.")
        else:
            st.error("No path exists between these nodes.")

st.divider()
st.caption("Academic prototype. Model metrics depend on the uploaded dataset and evaluation split. Validate data definitions, prediction-time feature availability, and business assumptions before operational use.")
