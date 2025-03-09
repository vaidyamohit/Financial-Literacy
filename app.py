import streamlit as st
import pandas as pd
import numpy as np
import json
from transformers import pipeline
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# ---- SETUP ----
st.set_page_config(page_title="AI Financial Assistant", layout="wide")
st.title("💰 AI Financial Literacy Assistant")
st.write("🚀 Ask me about budgeting, savings, or fraud detection!")

# ---- USER PROFILE SETUP ----
st.sidebar.header("User Profile")
income = st.sidebar.number_input("Monthly Income (₹):", min_value=0.0, step=1000.0, value=50000.0)
housing = st.sidebar.number_input("Housing Expenses (₹):", min_value=0.0, step=100.0, value=15000.0)
food = st.sidebar.number_input("Food & Groceries (₹):", min_value=0.0, step=100.0, value=8000.0)
transport = st.sidebar.number_input("Transport (₹):", min_value=0.0, step=100.0, value=5000.0)
utilities = st.sidebar.number_input("Utilities (₹):", min_value=0.0, step=100.0, value=3000.0)
savings_goal = st.sidebar.number_input("Savings Goal (₹):", min_value=0.0, step=1000.0, value=10000.0)

expenses = {
    "Housing": housing,
    "Food": food,
    "Transport": transport,
    "Utilities": utilities
}
total_expense = sum(expenses.values())
actual_savings = income - total_expense

st.sidebar.markdown(f"💰 **Total Monthly Expenses:** ₹{total_expense}")
st.sidebar.markdown(f"📈 **Actual Savings:** ₹{actual_savings} (Goal: ₹{savings_goal})")

# ---- CHATBOT FUNCTIONALITY ----
st.subheader("💬 Chat with AI Financial Assistant")
user_input = st.text_input("Ask me anything about finance:")

# Load NLP model for financial Q&A
qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

faq_data = {
    "What is a credit score?": "A credit score is a number that evaluates a consumer's creditworthiness.",
    "How do I save more money?": "Try following the 50/30/20 rule: 50% for needs, 30% for wants, 20% for savings.",
    "What is an emergency fund?": "An emergency fund is savings that cover 3-6 months of living expenses in case of job loss or unexpected costs."
}

def chatbot_response(query):
    query_lower = query.lower()
    for question, answer in faq_data.items():
        if question.lower() in query_lower:
            return answer

    context = "Credit scores are used to assess a borrower's creditworthiness. A good score can help get lower interest rates."
    response = qa_model(question=query, context=context)
    return response["answer"] if response else "I’m not sure, but I can learn!"

if user_input:
    response = chatbot_response(user_input)
    st.markdown(f"**AI Assistant:** {response}")

# ---- BUDGET ANALYSIS & RECOMMENDATIONS ----
st.subheader("📊 Budget Analysis & Recommendations")
st.write("Here's how your spending compares to financial best practices:")

ideal_needs = 0.5 * income
ideal_wants = 0.3 * income
ideal_savings = 0.2 * income

st.markdown(f"💡 **Ideal Savings Goal:** ₹{ideal_savings}")
st.markdown(f"📊 **Your Savings:** ₹{actual_savings}")

if actual_savings < savings_goal:
    st.warning("⚠️ You are not meeting your savings goal. Consider reducing discretionary expenses.")

# ---- FRAUD DETECTION MODULE ----
st.subheader("🛑 Fraud Detection System")

fraud_data = [50, 20, 15, 30, 1500, 80, 100, 5000]  # Example past transactions
scaler = StandardScaler()
scaled_data = scaler.fit_transform(np.array(fraud_data).reshape(-1, 1))

model = IsolationForest(contamination=0.1, random_state=42)
model.fit(scaled_data)

new_transactions = np.array([65, 5000, 25000]).reshape(-1, 1)
new_transactions_scaled = scaler.transform(new_transactions)
predictions = model.predict(new_transactions_scaled)

fraud_results = []
for amount, pred in zip(new_transactions.flatten(), predictions):
    if pred == -1:
        fraud_results.append(f"🚨 Fraud Alert: Transaction of ₹{amount} is suspicious!")
    else:
        fraud_results.append(f"✅ Transaction of ₹{amount} looks normal.")

if st.button("🔍 Run Fraud Check"):
    for result in fraud_results:
        st.warning(result)

st.write("This system analyzes transactions and detects anomalies based on spending patterns.")

# ---- DATA VISUALIZATION ----
st.subheader("📉 Expense Breakdown")
expense_df = pd.DataFrame(list(expenses.items()), columns=["Category", "Amount"])
st.bar_chart(expense_df.set_index("Category"))
