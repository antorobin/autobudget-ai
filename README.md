# AutoBudgetting AI

An AI-powered finance platform using predictive analytics and anomaly detection to forecast expenses, optimize savings, and adjust budgets for seasonal events. It scores spending habits and flags unusual bills with data-driven, actionable insights—delivering autonomous, smart personal finance management.

---

## Features

### 1. Forecaster  
- **Goal:** Predict next month’s essential expenses (rent, groceries, EMIs, school fees) based on past patterns.  
- **How:** Uses historical data and automatically adjusts forecasts for seasonal changes (e.g., school reopening, Diwali shopping, monsoon electricity spikes).  
- **Benefit:** Warns users ahead if projected expenses might exceed income.

### 2. Savings Advisor  
- **Goal:** Suggest a monthly "safe-to-save" amount without compromising essential expenses.  
- **How:** Analyzes past spending, identifies recurring and surprise costs, and dynamically adjusts buffer goals if unexpected expenses occur (like car repairs).  
- **Benefit:** Helps users save responsibly, even during unplanned financial events.

### 3. Budget Alert  
- **Goal:** Allocate extra funds for festivals (Diwali, Pongal, Raksha Bandhan) and family events (marriages, birthdays).  
- **How:** Looks at prior year’s festival spending, inflation, and current income to set realistic budgets. Automatically creates festival budgets 30–45 days before events and suggests cost-saving tips.  
- **Benefit:** Prevents overspending during special occasions with smart budgeting.

### 4. Spending Health Score  
- **Goal:** Provide a simple monthly score reflecting budgeting discipline (similar to a credit score).  
- **How:** Weighs positive habits (paying bills on time, increasing savings) against negatives (impulse spends, EMIs over 40% of income).  
- **Benefit:** Offers actionable recommendations to improve financial health (e.g., “Cut food delivery spend by ₹1,500 to improve score”).

### 5. Autonomous Bill Guard  
- **Goal:** Monitor recurring bills (electricity, mobile, internet, streaming) for unusual increases.  
- **How:** Detects anomalies (e.g., electricity bill 40% higher than last month), investigates seasonal/usage reasons, and compares service costs.  
- **Benefit:** Suggests remedial actions such as plan changes or appliance checks to optimize bills.


## Technologies Used

- Python 3.x  
- FastAPI  
- MongoDB  
- Prophet (Time-series forecasting)  
- AI/ML for anomaly detection and recommendations
