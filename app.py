import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="AI Customer Churn Dashboard",
    page_icon="📊",
    layout="wide"
)

# ================= CUSTOM CSS =================
# ================= FUTURISTIC UI CSS =================
st.markdown("""
<style>

/* ===== MAIN APP ===== */
.stApp {

    background:
    radial-gradient(circle at top left, #1e3a8a 0%, transparent 30%),
    radial-gradient(circle at bottom right, #7c3aed 0%, transparent 30%),
    linear-gradient(135deg, #050816 0%, #0f172a 100%);

    color: white;
}

/* ===== GLOBAL TEXT ===== */
html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
    color: white;
}

/* ===== PAGE SPACING ===== */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

/* ===== TITLES ===== */
.main-title {

    text-align: center;

    font-size: 58px;

    font-weight: 800;

    background:
    linear-gradient(
        to right,
        #38bdf8,
        #818cf8,
        #c084fc
    );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;

    margin-bottom: 10px;
}

.subtitle {

    text-align: center;

    font-size: 20px;

    color: #cbd5e1;

    margin-bottom: 40px;
}

/* ===== LABELS ===== */
label, p, h1, h2, h3, h4, h5, h6 {
    color: white !important;
}

/* ===== INPUT FIELDS ===== */
div[data-baseweb="select"] > div {

    background-color: rgba(15,23,42,0.8) !important;

    border-radius: 14px !important;

    border: 1px solid #334155 !important;
}

/* ===== NUMBER INPUT ===== */
.stNumberInput input {

    background-color: rgba(15,23,42,0.8) !important;

    color: white !important;

    border-radius: 14px !important;
}

/* ===== BUTTON ===== */
.stButton > button {

    width: 100%;

    height: 68px;

    border-radius: 18px;

    border: none;

    font-size: 24px;

    font-weight: bold;

    color: white;

    background:
    linear-gradient(
        135deg,
        #2563eb,
        #7c3aed,
        #ec4899
    );

    box-shadow:
    0px 0px 25px rgba(124,58,237,0.6);

    transition: 0.3s;
}

/* ===== BUTTON HOVER ===== */
.stButton > button:hover {

    transform: scale(1.02);

    box-shadow:
    0px 0px 40px rgba(236,72,153,0.8);
}

/* ===== RESULT CARDS ===== */
.result-card {

    background:
    linear-gradient(
        145deg,
        rgba(30,41,59,0.92),
        rgba(15,23,42,0.98)
    );

    border-radius: 22px;

    padding: 22px;

    margin-bottom: 20px;

    border: 1px solid rgba(255,255,255,0.08);

    box-shadow:
    0 8px 32px rgba(0,0,0,0.35);

    backdrop-filter: blur(10px);
}

/* ===== SECTION HEADINGS ===== */
.section-heading {

    font-size: 34px;

    font-weight: 700;

    background:
    linear-gradient(
        to right,
        #38bdf8,
        #c084fc
    );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;

    margin-top: 30px;
    margin-bottom: 20px;
}

/* ===== PROGRESS BAR ===== */
.stProgress > div > div > div > div {

    background-image:
    linear-gradient(
        to right,
        #38bdf8,
        #818cf8,
        #ec4899
    );
}

/* ===== DOWNLOAD BUTTON ===== */
.stDownloadButton > button {

    width: 100%;

    border-radius: 15px;

    background:
    linear-gradient(
        to right,
        #06b6d4,
        #3b82f6
    );

    color: white;

    font-weight: bold;

    height: 52px;

    border: none;
}

/* ===== SCROLLBAR ===== */
::-webkit-scrollbar {
    width: 10px;
}

::-webkit-scrollbar-thumb {

    background: #7c3aed;

    border-radius: 20px;
}

</style>
""", unsafe_allow_html=True)

# ================= LOAD FILES =================
log_reg = joblib.load("Models/Logistic Regression.pkl")
dt = joblib.load("Models/Decision Tree.pkl")
rf = joblib.load("Models/Random Forest.pkl")
nn = joblib.load("Models/Neural Network.pkl")

scaler = joblib.load("Models/scaler.pkl")
model_columns = joblib.load("Models/columns.pkl")

models = {
    "Logistic Regression": log_reg,
    "Decision Tree": dt,
    "Random Forest": rf,
    "Neural Network": nn
}

# ================= TITLE =================
st.markdown("""
<h2 style='text-align:center; font-size:55px;'>
📊 AI CUSTOMER CHURN DASHBOARD
</h2>
""", unsafe_allow_html=True)

st.markdown("""
<p style='text-align:center; font-size:20px; color:lightgray;'>
Predict customer churn using Machine Learning & Neural Networks
</p>
""", unsafe_allow_html=True)

st.write("")

# ================= INPUT SECTION =================
st.markdown("## 🧾 Customer Information")


# ---------- ROW 1 ----------
r1c1, r1c2, r1c3, r1c4 = st.columns(4)

with r1c1:
    gender = st.selectbox("Gender", ["Male", "Female"])

with r1c2:
    senior = st.selectbox("Senior Citizen", [0, 1])

with r1c3:
    partner = st.selectbox("Partner", ["Yes", "No"])

with r1c4:
    dependents = st.selectbox("Dependents", ["Yes", "No"])

# ---------- ROW 2 ----------
r2c1, r2c2, r2c3, r2c4 = st.columns(4)

with r2c1:
    tenure = st.slider("Tenure", 0, 100, 12)

with r2c2:
    monthly = st.number_input(
        "Monthly Charges",
        0.0,
        500.0,
        70.0
    )

with r2c3:
    total = st.number_input(
        "Total Charges",
        0.0,
        20000.0,
        1000.0
    )

with r2c4:
    paperless = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

# ---------- ROW 3 ----------
r3c1, r3c2 = st.columns(2)

with r3c1:
    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

with r3c2:
    payment = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer",
            "Credit card"
        ]
    )

# ================= CREATE INPUT =================
input_dict = {
    "gender_Male": 1 if gender == "Male" else 0,
    "SeniorCitizen": senior,
    "Partner_Yes": 1 if partner == "Yes" else 0,
    "Dependents_Yes": 1 if dependents == "Yes" else 0,
    "tenure": tenure,
    "MonthlyCharges": monthly,
    "TotalCharges": total,
    "Contract_One year": 1 if contract == "One year" else 0,
    "Contract_Two year": 1 if contract == "Two year" else 0,
    "PaymentMethod_Credit card": 1 if "Credit" in payment else 0,
    "PaymentMethod_Electronic check": 1 if "Electronic" in payment else 0,
    "PaperlessBilling_Yes": 1 if paperless == "Yes" else 0,
}

input_df = pd.DataFrame([input_dict])

# IMPORTANT
input_df = input_df.reindex(columns=model_columns, fill_value=0)

# SCALE ONLY NUMERIC COLUMNS
num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]

input_df[num_cols] = scaler.transform(input_df[num_cols])

st.write("")

# ================= PDF FUNCTION =================
def generate_pdf(results, risk, avg_prob):

    file_name = "churn_report.pdf"

    doc = SimpleDocTemplate(file_name, pagesize=A4)

    styles = getSampleStyleSheet()

    content = []

    content.append(
        Paragraph("Customer Churn Prediction Report",
                  styles['Title'])
    )

    content.append(Spacer(1, 15))

    content.append(
        Paragraph(f"Final Risk Level: {risk}",
                  styles['Heading2'])
    )

    content.append(
        Paragraph(f"Average Probability: {avg_prob:.2f}",
                  styles['Normal'])
    )

    content.append(Spacer(1, 15))

    for name, pred, prob in results:

        text = f"""
        <b>{name}</b> :
        {'CHURN' if pred==1 else 'NO CHURN'}
        | Probability = {prob:.2f}
        """

        content.append(
            Paragraph(text, styles['BodyText'])
        )

        content.append(Spacer(1, 8))

    doc.build(content)

    return file_name

# ================= PREDICT BUTTON =================
st.markdown('<div class="predict-btn">', unsafe_allow_html=True)

predict_button = st.button("🚀 PREDICT CUSTOMER CHURN")

st.markdown('</div>', unsafe_allow_html=True)

# ================= PREDICTION =================
if predict_button:

    st.write("")
    st.markdown("## 🔮 Prediction Results")

    results = []

    col1, col2 = st.columns(2)

    for i, (name, model) in enumerate(models.items()):

        pred = model.predict(input_df)[0]

        try:
            prob = model.predict_proba(input_df)[0][1]
        except:
            prob = 0.50

        results.append((name, pred, prob))

        accuracy = round((1 - abs(0.5 - prob)) * 100, 2)

        with (col1 if i % 2 == 0 else col2):

            st.markdown(
                f"""
                <div class="result-card">
                    <h3>{name}</h3>
                </div>
                """,
                unsafe_allow_html=True
            )

            if pred == 1:
                st.error("❌ CUSTOMER WILL CHURN")
            else:
                st.success("✅ CUSTOMER WILL NOT CHURN")

            st.progress(float(prob))

            st.write(f"### Probability: {prob:.2f}")

            st.write(f"### Confidence Score: {accuracy}%")

    # ================= FINAL RISK =================
    avg_prob = np.mean([x[2] for x in results])

    st.write("")
    st.markdown("## 📌 Final Customer Risk Analysis")

    if avg_prob > 0.7:
        risk = "🔴 HIGH RISK"
        st.error(risk)

    elif avg_prob > 0.4:
        risk = "🟡 MEDIUM RISK"
        st.warning(risk)

    else:
        risk = "🟢 LOW RISK"
        st.success(risk)

    st.progress(float(avg_prob))
    # ================= FEATURE IMPORTANCE =================

    st.write("")
    st.markdown("## 📊 Feature Importance")

    importances = rf.feature_importances_
    features = model_columns

    fi_df = pd.DataFrame({
        "Feature": features,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).head(10)

    plt.style.use("dark_background")

    fig, ax = plt.subplots(figsize=(10, 6))

    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")

    bars = ax.barh(
        fi_df["Feature"],
        fi_df["Importance"]
    )

    ax.invert_yaxis()

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')

    ax.set_title(
        "Top Features Affecting Churn",
        color="white",
        fontsize=18,
        fontweight='bold'
    )

    ax.set_xlabel(
        "Importance Score",
        color="white"
    )

    ax.grid(alpha=0.1)

    for bar in bars:

        width = bar.get_width()

        ax.text(
            width + 0.001,
            bar.get_y() + bar.get_height()/2,
            f"{width:.3f}",
            va='center',
            color='white'
        )

    plt.tight_layout()

    st.pyplot(fig)

    # ================= PDF DOWNLOAD =================

    st.write("")
    st.markdown("## 📄 Download Report")

    pdf_file = generate_pdf(results, risk, avg_prob)

    with open(pdf_file, "rb") as f:

        st.download_button(
            "⬇ Download Prediction Report",
            data=f,
            file_name="Churn_Report.pdf",
            mime="application/pdf"
        )
