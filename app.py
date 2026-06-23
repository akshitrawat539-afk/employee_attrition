import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
model = joblib.load("../model/attrition_model.pkl")
scaler = joblib.load("../model/scaler.pkl")

df["Attrition_Num"] = df["Attrition"].map(
    {"Yes": 1, "No": 0}
)

st.set_page_config(
    page_title="Employee Attrition Dashboard",
    page_icon="📊",
    layout="wide"
)
st.title("📊 Employee Attrition Dashboard")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Total Employees",
        len(df)
    )

with col2:
    st.metric(
        "Employees Left",
        len(df[df["Attrition"] == "Yes"])
    )

with col3:
    st.metric(
        "Attrition Rate",
        f"{round(df['Attrition_Num'].mean()*100,2)}%"
    )

st.divider()

tab1, tab2 = st.tabs(
    ["📈 Visualizations", "🔮 Prediction"]
)
with tab1:

    st.subheader(
        "Attrition Distribution"
    )

    fig1 = px.histogram(
        df,
        x="Attrition",
        title="Employee Attrition Distribution"
    )

    st.plotly_chart(
        fig1,
        use_container_width=True
    )

    st.subheader(
        "Overtime vs Attrition"
    )

    fig2 = px.histogram(
        df,
        x="OverTime",
        color="Attrition",
        barmode="group",
        title="Overtime Impact on Attrition"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    st.subheader(
        "Monthly Income vs Attrition"
    )

    fig3 = px.box(
        df,
        x="Attrition",
        y="MonthlyIncome",
        title="Income Analysis"
    )

    st.plotly_chart(
        fig3,
        use_container_width=True
    )

    st.subheader(
        "Job Satisfaction vs Attrition"
    )

    fig4 = px.histogram(
        df,
        x="JobSatisfaction",
        color="Attrition",
        barmode="group",
        title="Job Satisfaction Analysis"
    )

    st.plotly_chart(
        fig4,
        use_container_width=True
    )

    st.subheader(
        "Work Life Balance vs Attrition"
    )

    fig5 = px.histogram(
        df,
        x="WorkLifeBalance",
        color="Attrition",
        barmode="group",
        title="Work Life Balance Analysis"
    )

    st.plotly_chart(
        fig5,
        use_container_width=True
    )

# =====================================================
# PREDICTION TAB
# =====================================================

with tab2:

    st.subheader(
        "Predict Employee Attrition"
    )

    age = st.number_input(
        "Age",
        18,
        60,
        30
    )

    income = st.number_input(
        "Monthly Income",
        1000,
        50000,
        10000
    )

    years = st.number_input(
        "Years At Company",
        0,
        40,
        5
    )

    satisfaction = st.slider(
        "Job Satisfaction",
        1,
        4,
        3
    )

    worklife = st.slider(
        "Work Life Balance",
        1,
        4,
        3
    )

    distance = st.number_input(
        "Distance From Home",
        1,
        50,
        10
    )

    overtime = st.selectbox(
        "OverTime",
        ["No", "Yes"]
    )

    if st.button(
        "Predict Attrition"
    ):

        overtime_val = (
            1 if overtime == "Yes"
            else 0
        )

        data = pd.DataFrame(
            [[
                age,
                income,
                years,
                satisfaction,
                worklife,
                distance,
                overtime_val
            ]],
            columns=[
                "Age",
                "MonthlyIncome",
                "YearsAtCompany",
                "JobSatisfaction",
                "WorkLifeBalance",
                "DistanceFromHome",
                "OverTime"
            ]
        )

        scaled_data = scaler.transform(
            data
        )

        prediction = model.predict(
            scaled_data
        )[0]

        probability = model.predict_proba(
            scaled_data
        )[0][1]

        if prediction == 1:

            st.error(
                f"⚠️ Employee likely to leave.\n\nRisk Score: {probability*100:.2f}%"
            )

        else:

            st.success(
                f"✅ Employee likely to stay.\n\nRisk Score: {probability*100:.2f}%"
            )
