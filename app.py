import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="ChurnGuard AI | Customer Retention",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# MODERN UI
# ============================================================
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }

    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(128,128,128,.18);
    }

    .hero {
        padding: 28px 32px;
        border-radius: 22px;
        margin-bottom: 22px;
        background: linear-gradient(135deg, rgba(80,90,180,.20), rgba(20,160,150,.13));
        border: 1px solid rgba(128,128,128,.18);
    }

    .hero h1 {
        margin: 0;
        font-size: 42px;
        font-weight: 800;
    }

    .hero p {
        margin: 8px 0 0 0;
        font-size: 17px;
        opacity: .82;
    }

    .section-title {
        font-size: 25px;
        font-weight: 750;
        margin-top: 12px;
        margin-bottom: 12px;
    }

    .risk-card {
        padding: 20px;
        border-radius: 18px;
        border: 1px solid rgba(128,128,128,.18);
        background: rgba(128,128,128,.06);
        text-align: center;
    }

    .risk-number {
        font-size: 38px;
        font-weight: 800;
    }

    .small-muted {
        opacity: .68;
        font-size: 13px;
    }

    .feature-card {
        padding: 18px;
        border-radius: 16px;
        border: 1px solid rgba(128,128,128,.15);
        min-height: 130px;
        background: rgba(128,128,128,.045);
    }

    .footer {
        text-align: center;
        padding: 25px 0 5px;
        opacity: .65;
        font-size: 13px;
    }

    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,.16);
        border-radius: 16px;
        padding: 14px;
        background: rgba(128,128,128,.045);
    }

    .stButton > button {
        border-radius: 12px;
        font-weight: 700;
        min-height: 45px;
    }

    /* ========================================================
       PROFESSIONAL SECTION CARDS
       ======================================================== */

    .info-card {
        padding: 20px;
        border-radius: 18px;
        border: 1px solid rgba(128,128,128,.16);
        background: rgba(128,128,128,.05);
        margin-bottom: 15px;
    }

    .info-card h3 {
        margin-bottom: 8px;
    }

    .risk-high {
        padding: 18px;
        border-radius: 16px;
        border-left: 5px solid #ff4b4b;
        background: rgba(255, 75, 75, .08);
    }

    .risk-medium {
        padding: 18px;
        border-radius: 16px;
        border-left: 5px solid #ffa726;
        background: rgba(255, 167, 38, .08);
    }

    .risk-low {
        padding: 18px;
        border-radius: 16px;
        border-left: 5px solid #21c55d;
        background: rgba(33, 197, 93, .08);
    }
    .page-header {
        padding: 22px 26px;
        border-radius: 18px;
        margin-bottom: 20px;
        background: rgba(80,90,180,.10);
        border: 1px solid rgba(128,128,128,.16);
    }

    .page-header h1 {
        margin: 0;
        font-size: 34px;
        font-weight: 800;
    }

    .page-header p {
        margin: 6px 0 0 0;
        opacity: .75;
    }

    .empty-state {
        text-align: center;
        padding: 60px 25px;
        border-radius: 20px;
        border: 1px dashed rgba(128,128,128,.25);
        background: rgba(128,128,128,.04);
    }

    .risk-factor {
        display: flex;
        align-items: center;
        gap: 16px;
        padding: 16px;
        margin: 10px 0;
        border-radius: 14px;
        border: 1px solid rgba(128,128,128,.14);
        background: rgba(128,128,128,.045);
    }

    .risk-icon {
        font-size: 28px;
        min-width: 42px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_model():
    return joblib.load("churn_model.pkl")


try:
    model = load_model()
    model_loaded = True
except Exception as e:
    model = None
    model_loaded = False
    model_error = str(e)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## 🛡️ ChurnGuard AI")
    st.caption("Customer Retention Intelligence")

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "🏠 Executive Dashboard",
            "🔮 Churn Prediction",
            "🔍 Risk Intelligence",
            "💡 Retention Center",
            "📊 Customer Analytics",
            "🤖 Model Center",
            "ℹ️ About",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    st.markdown("### System Status")
    if model_loaded:
        st.success("● ML Model Online")
    else:
        st.error("● ML Model Error")

    st.caption("AI-powered customer retention")
    st.caption("Built with Python + Streamlit")


# ============================================================
# HERO
# ============================================================
def hero(title, subtitle):
    st.markdown(
        f"""
        <div class="hero">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def risk_color_message(risk):
    if risk == "HIGH":
        return "🔴 HIGH RISK"
    if risk == "MEDIUM":
        return "🟠 MEDIUM RISK"
    return "🟢 LOW RISK"


# ============================================================
# CUSTOMER INPUT
# ============================================================
def get_customer_inputs():

    st.markdown(
        '<div class="section-title">👤 Customer Profile</div>',
        unsafe_allow_html=True,
    )

    with st.form("customer_prediction_form"):

        st.markdown("#### Personal & Relationship Details")

        c1, c2, c3 = st.columns(3)

        with c1:
            customer_id = st.text_input("Customer ID", "CUST001")

            gender = st.selectbox(
                "Gender",
                ["Male", "Female"]
            )

            # ==================================================
            # CHANGED: SENIOR CITIZEN UI
            # ==================================================
            senior_citizen = st.selectbox(
                "Senior Citizen",
                ["No", "Yes"]
            )

            # Convert Yes/No to 0/1 for the ML model
            senior_citizen_value = (
                1 if senior_citizen == "Yes" else 0
            )

        with c2:
            partner = st.selectbox(
                "Partner",
                ["Yes", "No"]
            )

            dependents = st.selectbox(
                "Dependents",
                ["Yes", "No"]
            )

            tenure = st.number_input(
                "Tenure (Months)",
                min_value=0,
                max_value=100,
                value=12
            )

        with c3:
            phone_service = st.selectbox(
                "Phone Service",
                ["Yes", "No"]
            )

            multiple_lines = st.selectbox(
                "Multiple Lines",
                ["Yes", "No", "No phone service"]
            )

            paperless_billing = st.selectbox(
                "Paperless Billing",
                ["Yes", "No"]
            )

        st.markdown("#### Services")

        c1, c2, c3 = st.columns(3)

        with c1:

            internet_service = st.selectbox(
                "Internet Service",
                ["DSL", "Fiber optic", "No"]
            )

            online_security = st.selectbox(
                "Online Security",
                ["Yes", "No", "No internet service"]
            )

            online_backup = st.selectbox(
                "Online Backup",
                ["Yes", "No", "No internet service"]
            )

        with c2:

            device_protection = st.selectbox(
                "Device Protection",
                ["Yes", "No", "No internet service"]
            )

            tech_support = st.selectbox(
                "Tech Support",
                ["Yes", "No", "No internet service"]
            )

            streaming_tv = st.selectbox(
                "Streaming TV",
                ["Yes", "No", "No internet service"]
            )

        with c3:

            streaming_movies = st.selectbox(
                "Streaming Movies",
                ["Yes", "No", "No internet service"]
            )

            contract = st.selectbox(
                "Contract",
                ["Month-to-month", "One year", "Two year"]
            )

            payment_method = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )

        st.markdown("#### Billing")

        c1, c2 = st.columns(2)

        with c1:
            monthly_charges = st.number_input(
                "Monthly Charges",
                min_value=0.0,
                value=50.0,
                step=1.0
            )

        with c2:
            total_charges = st.number_input(
                "Total Charges",
                min_value=0.0,
                value=500.0,
                step=10.0
            )

        submitted = st.form_submit_button(
            "🔮 ANALYZE CUSTOMER RISK",
            use_container_width=True,
        )

    # ========================================================
    # CUSTOMER DATA
    # ========================================================
    customer_data = pd.DataFrame(
        {
            "gender": [gender],

            # IMPORTANT:
            # Model receives 0 or 1
            "SeniorCitizen": [senior_citizen_value],

            "Partner": [partner],
            "Dependents": [dependents],
            "tenure": [tenure],
            "PhoneService": [phone_service],
            "MultipleLines": [multiple_lines],
            "InternetService": [internet_service],
            "OnlineSecurity": [online_security],
            "OnlineBackup": [online_backup],
            "DeviceProtection": [device_protection],
            "TechSupport": [tech_support],
            "StreamingTV": [streaming_tv],
            "StreamingMovies": [streaming_movies],
            "Contract": [contract],
            "PaperlessBilling": [paperless_billing],
            "PaymentMethod": [payment_method],
            "MonthlyCharges": [monthly_charges],
            "TotalCharges": [total_charges],
        }
    )

    data = {
        "customer_id": customer_id,
        "gender": gender,

        # Keep the UI value as Yes/No
        "senior_citizen": senior_citizen,

        "tenure": tenure,
        "contract": contract,
        "internet_service": internet_service,
        "payment_method": payment_method,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "online_security": online_security,
        "tech_support": tech_support,
        "customer_data": customer_data,
    }

    return data, submitted


# ============================================================
# CUSTOMER ANALYSIS
# ============================================================
def analyze_customer(data):

    if model is None:
        raise RuntimeError(
            "The trained model could not be loaded."
        )

    customer_data = data["customer_data"]

    prediction = int(
        model.predict(customer_data)[0]
    )

    probability = float(
        model.predict_proba(customer_data)[0][1]
    )

    if probability < 0.30:
        risk_level = "LOW"
    elif probability < 0.70:
        risk_level = "MEDIUM"
    else:
        risk_level = "HIGH"

    monthly_charges = data["monthly_charges"]
    contract = data["contract"]
    tenure = data["tenure"]
    payment_method = data["payment_method"]
    internet_service = data["internet_service"]
    online_security = data["online_security"]
    tech_support = data["tech_support"]

    if monthly_charges >= 70 and probability >= 0.50:
        customer_value = "HIGH-VALUE AT RISK"
    elif monthly_charges >= 50:
        customer_value = "MEDIUM-VALUE"
    else:
        customer_value = "STANDARD"

    reasons = []

    if contract == "Month-to-month":
        reasons.append(
            "Month-to-month contract increases retention risk."
        )

    if monthly_charges > 70:
        reasons.append(
            "Monthly charges are relatively high."
        )

    if tenure < 12:
        reasons.append(
            "Customer has relatively low tenure."
        )

    if payment_method == "Electronic check":
        reasons.append(
            "Customer uses electronic check payment."
        )

    if internet_service == "Fiber optic":
        reasons.append(
            "Customer uses fiber optic internet service."
        )

    if online_security == "No":
        reasons.append(
            "Online security is not subscribed."
        )

    if tech_support == "No":
        reasons.append(
            "Technical support is not subscribed."
        )

    if probability >= 0.70:

        if contract == "Month-to-month":
            recommendation = (
                "Offer a discounted long-term contract "
                "with loyalty benefits."
            )

        elif monthly_charges > 70:
            recommendation = (
                "Offer a personalized pricing plan "
                "or service bundle."
            )

        elif tenure < 12:
            recommendation = (
                "Provide a new-customer loyalty offer "
                "and onboarding support."
            )

        else:
            recommendation = (
                "Prioritize the customer for a personalized "
                "retention campaign."
            )

    elif probability >= 0.30:

        recommendation = (
            "Monitor the customer and provide targeted "
            "engagement offers."
        )

    else:

        recommendation = (
            "No immediate retention intervention required."
        )

    return {
        "prediction": prediction,
        "probability": probability,
        "risk_level": risk_level,
        "customer_value": customer_value,
        "reasons": reasons,
        "recommendation": recommendation,
    }


# ============================================================
# SAVE RESULT
# ============================================================
def save_result(data, result):

    st.session_state["customer_data"] = data
    st.session_state["result"] = result


# ============================================================
# DASHBOARD
# ============================================================
if page == "🏠 Executive Dashboard":

    hero(
        "🛡️ ChurnGuard AI",
        "Predict customer churn, understand risk, and turn predictions into retention actions.",
    )

    st.markdown("### 📌 Executive Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "👥 Customers",
            "7,043",
            "Portfolio"
        )

    with col2:
        st.metric(
            "⚠️ At-Risk Customers",
            "1,869",
            "Needs attention"
        )

    with col3:
        st.metric(
            "💰 Revenue at Risk",
            "₹12.5M",
            "Estimated"
        )

    with col4:
        st.metric(
            "📈 Retention Rate",
            "73.5%",
            "Target"
        )

    st.divider()

    left, right = st.columns([1.1, 0.9])

    with left:

        st.markdown("### 📊 Churn Distribution")

        churn_df = pd.DataFrame(
            {
                "Customer Status": [
                    "Stayed",
                    "Churned"
                ],
                "Customers": [
                    5174,
                    1869
                ],
            }
        )

        fig = px.pie(
            churn_df,
            names="Customer Status",
            values="Customers",
            hole=0.55,
        )

        fig.update_layout(
            margin=dict(
                l=10,
                r=10,
                t=30,
                b=10
            ),
            legend_title="Status",
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with right:

        st.markdown("### 🎯 Retention Priorities")

        priorities = pd.DataFrame(
            {
                "Priority": [
                    "High",
                    "Medium",
                    "Low"
                ],
                "Customers": [
                    1869,
                    1450,
                    3724
                ],
            }
        )

        fig = px.bar(
            priorities,
            x="Priority",
            y="Customers",
            text="Customers",
        )

        fig.update_traces(
            textposition="outside"
        )

        fig.update_layout(
            margin=dict(
                l=10,
                r=10,
                t=30,
                b=10
            ),
            xaxis_title="Risk Priority",
            yaxis_title="Customers",
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.markdown("### 🚀 How the System Works")

    a, b, c, d = st.columns(4)

    with a:
        st.markdown(
            '<div class="feature-card"><b>1. Predict</b><br>'
            'Estimate each customer’s churn probability.</div>',
            unsafe_allow_html=True,
        )

    with b:
        st.markdown(
            '<div class="feature-card"><b>2. Explain</b><br>'
            'Identify practical risk factors behind the prediction.</div>',
            unsafe_allow_html=True,
        )

    with c:
        st.markdown(
            '<div class="feature-card"><b>3. Prioritize</b><br>'
            'Highlight high-value customers who need attention.</div>',
            unsafe_allow_html=True,
        )

    with d:
        st.markdown(
            '<div class="feature-card"><b>4. Retain</b><br>'
            'Recommend targeted retention actions.</div>',
            unsafe_allow_html=True,
        )


# ============================================================
# CHURN PREDICTION
# ============================================================
elif page == "🔮 Churn Prediction":

    hero(
        "🔮 Churn Prediction",
        "Enter customer information and generate an AI-powered churn risk assessment.",
    )

    if not model_loaded:

        st.error(
            "The trained model could not be loaded."
        )

        st.code(model_error)

    else:

        data, submitted = get_customer_inputs()

        if submitted:

            try:

                result = analyze_customer(data)

                save_result(
                    data,
                    result
                )

                st.divider()

                st.markdown("### 🎯 Prediction Result")

                probability_percent = result["probability"] * 100
                risk = result["risk_level"]

                c1, c2, c3 = st.columns(3)

                with c1:
                    st.metric(
                        "Churn Probability",
                        f"{probability_percent:.1f}%"
                    )

                with c2:
                    if risk == "HIGH":
                        st.error("🔴 HIGH RISK")
                    elif risk == "MEDIUM":
                        st.warning("🟠 MEDIUM RISK")
                    else:
                        st.success("🟢 LOW RISK")

                with c3:
                    st.metric(
                        "Customer Segment",
                        result["customer_value"]
                    )

                st.progress(
                    min(max(result["probability"], 0.0), 1.0),
                    text=f"Churn Risk Score: {probability_percent:.1f}%"
                )

                if risk == "HIGH":
                    st.error(
                        "🚨 High churn risk detected. "
                        "Immediate retention action is recommended."
                    )
                elif risk == "MEDIUM":
                    st.warning(
                        "⚠️ Moderate churn risk detected. "
                        "Monitor and engage this customer."
                    )
                else:
                    st.success(
                        "✅ Low churn risk. "
                        "Continue normal customer engagement."
                    )

                fig = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=result["probability"] * 100,
                        title={
                            "text": "Churn Risk Score"
                        },
                        gauge={
                            "axis": {
                                "range": [0, 100]
                            },
                            "steps": [
                                {
                                    "range": [0, 30]
                                },
                                {
                                    "range": [30, 70]
                                },
                                {
                                    "range": [70, 100]
                                },
                            ],
                            "threshold": {
                                "line": {
                                    "width": 4
                                },
                                "value":
                                    result["probability"] * 100,
                            },
                        },
                    )
                )

                fig.update_layout(
                    height=320
                )

                st.plotly_chart(
                    fig,
                    width="stretch"
                )

                st.markdown(
                    "### 👤 Customer Snapshot"
                )

                s1, s2, s3, s4 = st.columns(4)

                s1.metric(
                    "Customer ID",
                    data["customer_id"]
                )

                s2.metric(
                    "Tenure",
                    f"{data['tenure']} months"
                )

                s3.metric(
                    "Monthly Charges",
                    f"₹{data['monthly_charges']:.2f}"
                )

                s4.metric(
                    "Contract",
                    data["contract"]
                )

            except Exception as e:

                st.error(
                    "Prediction failed."
                )

                st.exception(e)


# ============================================================
# RISK INTELLIGENCE
# ============================================================
elif page == "🔍 Risk Intelligence":

    hero(
        "🔍 Risk Intelligence",
        "Understand why this customer is considered at risk."
    )

    if "result" not in st.session_state:

        st.info(
            "💡 Go to 🔮 Churn Prediction and analyze a customer first."
        )

    else:

        result = st.session_state["result"]
        data = st.session_state["customer_data"]

        probability = result["probability"] * 100

        # ----------------------------------------------------
        # RISK SUMMARY
        # ----------------------------------------------------

        st.markdown("### 📊 Risk Summary")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Churn Probability",
            f"{probability:.1f}%"
        )

        c2.metric(
            "Risk Level",
            result["risk_level"]
        )

        c3.metric(
            "Customer Value",
            result["customer_value"]
        )

        c4.metric(
            "Customer ID",
            data["customer_id"]
        )

        st.divider()

        # ----------------------------------------------------
        # RISK METER + FACTORS
        # ----------------------------------------------------

        left, right = st.columns([1, 1])

        with left:

            st.markdown("### 🚦 Risk Score")

            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=probability,
                    title={
                        "text": "Churn Probability (%)"
                    },
                    gauge={
                        "axis": {
                            "range": [0, 100]
                        },
                        "steps": [
                            {
                                "range": [0, 30]
                            },
                            {
                                "range": [30, 70]
                            },
                            {
                                "range": [70, 100]
                            }
                        ],
                        "threshold": {
                            "line": {
                                "width": 4
                            },
                            "value": probability
                        }
                    }
                )
            )

            fig.update_layout(
                height=330,
                margin=dict(
                    l=20,
                    r=20,
                    t=50,
                    b=20
                )
            )

            st.plotly_chart(
                fig,
                width="stretch"
            )

        with right:

            st.markdown("### ⚠️ Key Risk Factors")

            if result["reasons"]:

                for i, reason in enumerate(
                    result["reasons"],
                    start=1
                ):

                    st.warning(
                        f"**{i}.** {reason}"
                    )

            else:

                st.success(
                    "No major rule-based risk factors detected."
                )

        st.divider()

        # ----------------------------------------------------
        # CUSTOMER PROFILE
        # ----------------------------------------------------

        st.markdown("### 👤 Customer Risk Profile")

        p1, p2, p3, p4 = st.columns(4)

        p1.metric(
            "Tenure",
            f"{data['tenure']} months"
        )

        p2.metric(
            "Monthly Charges",
            f"₹{data['monthly_charges']:.2f}"
        )

        p3.metric(
            "Contract",
            data["contract"]
        )

        p4.metric(
            "Payment Method",
            data["payment_method"]
        )

        st.divider()

# ============================================================
# RISK INTELLIGENCE
# ============================================================

        st.divider()

        st.markdown("### 🧪 What-If Churn Simulator")

        st.caption(
            "Explore how selected customer changes may affect "
            "the predicted churn probability."
        )

        original_data = st.session_state["customer_data"]

        simulator_contract = st.selectbox(
            "Try a different contract",
            [
                "Month-to-month",
                "One year",
                "Two year"
            ],
            index=[
                "Month-to-month",
                "One year",
                "Two year"
            ].index(
                original_data["contract"]
            ),
            key="simulator_contract"
        )

        if st.button(
            "🧪 SIMULATE",
            width="stretch"
        ):

            simulated_df = original_data["customer_data"].copy()

            simulated_df["Contract"] = simulator_contract

            simulated_probability = float(
                model.predict_proba(simulated_df)[0][1]
            )

            old_probability = (
                st.session_state["result"]["probability"]
            )

            old_percent = old_probability * 100
            new_percent = simulated_probability * 100

            change = old_percent - new_percent

            st.markdown("### 📊 Simulation Result")

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Current Risk",
                f"{old_percent:.1f}%"
            )

            c2.metric(
                "Simulated Risk",
                f"{new_percent:.1f}%"
            )

            c3.metric(
                "Risk Change",
                f"{change:+.1f}%"
            )

            if change > 0:

                st.success(
                    f"✅ Predicted churn risk decreases by "
                    f"{change:.1f} percentage points."
                )

            elif change < 0:

                st.warning(
                    f"⚠️ Predicted churn risk increases by "
                    f"{abs(change):.1f} percentage points."
                )

            else:

                st.info(
                    "The selected change does not alter "
                    "the predicted probability."
                )

# RETENTION CENTER
# ============================================================
elif page == "💡 Retention Center":

    hero(
        "💡 Retention Center",
        "Convert churn predictions into practical customer retention actions.",
    )

    if "result" not in st.session_state:

        st.info(
            "First analyze a customer from 🔮 Churn Prediction."
        )

    else:

        result = st.session_state["result"]
        data = st.session_state["customer_data"]

        c1, c2 = st.columns(2)

        with c1:

            st.markdown(
                "### 🎯 Recommended Action"
            )

            st.info(
                result["recommendation"]
            )

        with c2:

            st.markdown(
                "### 🚦 Intervention Priority"
            )

            if result["probability"] >= 0.70:

                st.error(
                    "HIGH — Act immediately"
                )

            elif result["probability"] >= 0.30:

                st.warning(
                    "MEDIUM — Monitor and engage"
                )

            else:

                st.success(
                    "LOW — Normal engagement"
                )

        st.divider()

        st.markdown(
            "### 🧩 Suggested Retention Playbook"
        )

        actions = []

        if data["contract"] == "Month-to-month":

            actions.append(
                "Offer a long-term contract with loyalty benefits."
            )

        if data["monthly_charges"] > 70:

            actions.append(
                "Review pricing and offer a suitable service bundle."
            )

        if data["tenure"] < 12:

            actions.append(
                "Provide onboarding and early-stage loyalty support."
            )

        if data["online_security"] == "No":

            actions.append(
                "Offer an online security package."
            )

        if data["tech_support"] == "No":

            actions.append(
                "Offer technical support or a service assistance plan."
            )

        if not actions:

            actions.append(
                "Continue personalized engagement and monitor risk."
            )

        for action in actions:

            st.markdown(
                f"✅ {action}"
            )

        st.divider()

        st.markdown(
            "### 💰 Customer Priority"
        )

        if result["customer_value"] == "HIGH-VALUE AT RISK":

            st.error(
                "This customer combines higher charges with elevated "
                "churn risk. Prioritize personalized retention."
            )

        elif result["customer_value"] == "MEDIUM-VALUE":

            st.warning(
                "This customer has moderate value. "
                "Use targeted engagement."
            )

        else:

            st.success(
                "Standard retention workflow is appropriate."
            )


# ============================================================
# CUSTOMER ANALYTICS
# ============================================================
elif page == "📊 Customer Analytics":

    hero(
        "📊 Customer Analytics",
        "Explore the selected customer’s profile and risk metrics.",
    )

    if "result" not in st.session_state:

        st.info(
            "Analyze a customer first to populate this section."
        )

    else:

        data = st.session_state["customer_data"]
        result = st.session_state["result"]

        chart_df = pd.DataFrame(
            {
                "Metric": [
                    "Churn Probability",
                    "Monthly Charges",
                    "Tenure",
                ],
                "Value": [
                    result["probability"] * 100,
                    data["monthly_charges"],
                    data["tenure"],
                ],
            }
        )

        fig = px.bar(
            chart_df,
            x="Metric",
            y="Value",
            text="Value",
            title="Customer Risk & Profile Metrics",
        )

        fig.update_traces(
            texttemplate="%{text:.1f}",
            textposition="outside"
        )

        fig.update_layout(
            yaxis_title="Value",
            xaxis_title=""
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.markdown(
            "### 📋 Complete Customer Record"
        )

        st.dataframe(
            data["customer_data"],
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "⬇️ Download Customer Record",
            data["customer_data"].to_csv(
                index=False
            ).encode("utf-8"),
            file_name=f"{data['customer_id']}_customer_record.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ============================================================
# MODEL CENTER
# ============================================================
elif page == "🤖 Model Center":

    hero(
        "🤖 Model Center",
        "Inspect the trained machine-learning model used by the application.",
    )

    if model is None:

        st.error(
            "Model is not available."
        )

    else:

        c1, c2 = st.columns(2)

        with c1:

            st.markdown(
                "### 🧠 Model Information"
            )

            st.write(
                "**Model type:**",
                type(model).__name__
            )

            if hasattr(model, "classes_"):

                st.write(
                    "**Classes:**",
                    list(model.classes_)
                )

        with c2:

            st.markdown(
                "### ⚙️ Pipeline Information"
            )

            if hasattr(model, "named_steps"):

                st.write(
                    "**Pipeline steps:**",
                    list(model.named_steps.keys())
                )

            else:

                st.info(
                    "The saved object does not expose pipeline steps."
                )

        st.divider()

        st.markdown(
            "### 📌 Important"
        )

        st.info(
            "This page intentionally displays only information available "
            "from the saved model. Accuracy, precision, recall, F1-score "
            "and ROC-AUC are not invented here because the current app.py "
            "does not contain the original test-set evaluation results."
        )


# ============================================================
# ABOUT
# ============================================================
elif page == "ℹ️ About":

    hero(
        "ℹ️ About ChurnGuard AI",
        "An AI-powered customer retention intelligence system.",
    )

    st.markdown(
        "### 🎯 Problem Statement"
    )

    st.write(
        "Customer churn can reduce revenue and make long-term customer "
        "relationships difficult to maintain. This application uses a "
        "trained machine-learning model to estimate churn probability "
        "and classify customers by risk."
    )

    st.markdown(
        "### 🚀 What This System Provides"
    )

    features = [
        (
            "🔮",
            "Churn Prediction",
            "Predict the probability that a customer may churn."
        ),
        (
            "🔍",
            "Risk Explanation",
            "Highlight practical factors associated with higher risk."
        ),
        (
            "💰",
            "Customer Value",
            "Identify high-value customers who may need attention."
        ),
        (
            "💡",
            "Retention Strategy",
            "Recommend targeted actions based on customer profile."
        ),
        (
            "📊",
            "Analytics",
            "Present customer risk information visually."
        ),
        (
            "🤖",
            "Model Center",
            "Inspect the saved machine-learning model."
        ),
    ]

    for icon, title, description in features:

        st.markdown(
            f"""
            <div class="feature-card">
                <h4>{icon} {title}</h4>
                <p>{description}</p>
            </div>
            <br>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        "### 🛠️ Technology Stack"
    )

    tech1, tech2, tech3, tech4 = st.columns(4)

    tech1.metric(
        "Language",
        "Python"
    )

    tech2.metric(
        "Interface",
        "Streamlit"
    )

    tech3.metric(
        "ML",
        "Scikit-learn"
    )

    tech4.metric(
        "Visualization",
        "Plotly"
    )


# ============================================================
# FOOTER
# ============================================================
st.markdown(
    """
    <div class="footer">
        🛡️ <b>ChurnGuard AI</b> · AI Customer Retention Intelligence System<br>
        Predict · Explain · Prioritize · Retain
    </div>
    """,
    unsafe_allow_html=True,
)
