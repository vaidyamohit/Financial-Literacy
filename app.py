import streamlit as st
import pandas as pd
import numpy as np
import ollama  # Ollama for local chatbot
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# ---- SETUP ----
st.set_page_config(page_title="AI Financial Advisor", layout="wide")
st.title("💰 AI Financial Advisor - Smart Investment & Budgeting Assistant")

# ---- SIDEBAR: USER PROFILE ----
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

# ---- BUTTONS FOR NAVIGATION ----
st.subheader("📌 Select a Feature:")
selected_option = st.radio(
    "Choose a section:",
    ["AI Financial Chatbot", "Investment Suggestions", "Fraud Detection System", "Expense Breakdown", "Financial Health Check"],
    index=0
)

# ---- OLLAMA-POWERED CHATBOT ----
if selected_option == "AI Financial Chatbot":
    st.subheader("💬 AI Financial Chatbot")
    st.write("Ask me anything about **investments, stock market, budgeting, or savings!**")

    @st.cache_resource
    def load_ollama():
        """Loads the Ollama model (Mistral-7B)."""
        try:
            return ollama.ChatCompletion.create(model="mistral", messages=[])
        except Exception as e:
            st.error("⚠️ Failed to load Ollama chatbot.")
            return None

    chatbot = load_ollama()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_query = st.chat_input("Ask me about stocks, crypto, real estate, or budget planning!")

    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})

        if chatbot:
            response = ollama.ChatCompletion.create(
                model="mistral",
                messages=[{"role": "system", "content": "You are a financial advisor."},
                          {"role": "user", "content": user_query}]
            )
            llm_response = response["message"]["content"]
        else:
            llm_response = "⚠️ Sorry, the chatbot is unavailable. Try again later."

        st.session_state.messages.append({"role": "assistant", "content": llm_response})

        with st.chat_message("assistant"):
            st.markdown(llm_response)

# ---- INVESTMENT SUGGESTIONS ----
elif selected_option == "Investment Suggestions":
    st.subheader("📊 Investment Suggestions")
    st.write("💡 **Based on your profile, here are personalized investment options:**")

    if actual_savings > 0:
        st.success("✅ You have positive savings! Here’s where you could invest:")

        if actual_savings > income * 0.2:
            st.write("💼 **Stock Market:** Consider investing in blue-chip stocks or index funds.")
            st.write("🏠 **Real Estate:** If you can afford it, consider a real estate investment.")

        if actual_savings > income * 0.1:
            st.write("📈 **Mutual Funds:** A good option for diversification.")

        st.write("💳 **Fixed Deposits & Bonds:** Secure, lower-risk investments.")
    else:
        st.error("⚠️ Your expenses exceed your income. Focus on saving before investing.")

# ---- FRAUD DETECTION SYSTEM ----
elif selected_option == "Fraud Detection System":
    st.subheader("🛑 Fraud Detection System")
    
    fraud_data = [50, 20, 15, 30, 1500, 80, 100, 5000]
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

# ---- EXPENSE BREAKDOWN ----
elif selected_option == "Expense Breakdown":
    st.subheader("📉 Expense Breakdown")
    expense_df = pd.DataFrame(list(expenses.items()), columns=["Category", "Amount"])
    st.bar_chart(expense_df.set_index("Category"))

# ---- FINANCIAL HEALTH CHECK ----
elif selected_option == "Financial Health Check":
    st.subheader("📊 Financial Health Check")
    
    st.write("💡 **Suggestions to improve your financial health:**")
    if actual_savings < savings_goal:
        st.warning("⚠️ You are not meeting your savings goal. Reduce unnecessary expenses.")
    
    if total_expense > income:
        st.error("⚠️ Your expenses exceed your income. Reduce spending or find additional income sources.")
    
    if actual_savings > 0.2 * income:
        st.success("✅ You are saving more than 20% of your income. Keep it up!")
