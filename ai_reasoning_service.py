"""
AI Reasoning Service (FastAPI)

Features implemented in this single-file MVP:
- Forecast essentials (Prophet)
- Savings advisor
- Festival / event budget planner
- Spending health score
- Bill guard (anomaly detection)

Assumptions / integrations:
- Cashflow microservice: provides transactions (POSTGRES-backed)
  - URL: CASHFLOW_URL (env)
  - Endpoint: GET /api/v1/transactions?user_id={user_id}&months={n}
  - Response: {"transactions": [{"date":"2025-07-01","amount":6500,"category":"groceries","meta":{}},...]}

- Budgetting microservice: provides budgets/goals
  - URL: BUDGET_URL (env)
  - Endpoint: GET /api/v1/budgets?user_id={user_id}
  - Response: {"budgets": [{"category":"groceries","monthly":6000}, ...]}
"""

from typing import List, Dict, Any, Optional
import os
import logging
import datetime
import statistics
import json

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
import httpx
import motor.motor_asyncio

# ML / Forecasting
from prophet import Prophet
import pandas as pd
import numpy as np

# LangChain / OpenAI (simple wrapper)
# We keep usage minimal and fallback to direct OpenAI if LangChain not available.
try:
    from langchain.chat_models import ChatOpenAI
    from langchain.schema import HumanMessage
    LANGCHAIN_AVAILABLE = True
except Exception:
    LANGCHAIN_AVAILABLE = False
    import openai

# --- Config & Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ai-reasoning-service")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
CASHFLOW_URL = os.getenv("CASHFLOW_URL", "https://b0c3083f-ee9d-4d61-9bd1-5119d3d94674.mock.pstmn.io")
BUDGET_URL = os.getenv("BUDGET_URL", "https://b0c3083f-ee9d-4d61-9bd1-5119d3d94674.mock.pstmn.io")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "YOUR-GROK-API-KEY")
# GROQ_MODEL = os.getenv("GROQ_MODEL", "mixtral-8x7b-32768")  # default Groq model
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


if not GROQ_API_KEY:
    logger.warning("GROQ_API_KEY not set. Groq API calls will fail until set.")

if not LANGCHAIN_AVAILABLE and OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

# --- FastAPI app ---
app = FastAPI(title="AI Reasoning Service")

# --- DB client ---
mongo_client: motor.motor_asyncio.AsyncIOMotorClient = None
mongo_db = None

# --- HTTP client ---
http_client: Optional[httpx.AsyncClient] = None

# --- Models / Pydantic ---
class ForecastRequest(BaseModel):
    user_id: str
    category_list: Optional[List[str]] = None
    history_months: int = 12

class ForecastResponse(BaseModel):
    predictions: Dict[str, float]
    warnings: Optional[List[str]] = []

class SavingsRequest(BaseModel):
    user_id: str
    income: Optional[float] = None
    history_months: int = 6

class SavingsResponse(BaseModel):
    safe_to_save: float
    reasoning: str

class FestivalBudgetRequest(BaseModel):
    user_id: str
    upcoming_events: List[str]

class FestivalBudgetResponse(BaseModel):
    festival_budgets: Dict[str, float]
    suggestions: List[str]

class BillMonitorItem(BaseModel):
    type: str
    amount: float
    date: Optional[str] = None

class BillMonitorRequest(BaseModel):
    user_id: str
    bills: List[BillMonitorItem]

class BillMonitorResponse(BaseModel):
    alerts: List[Dict[str, Any]]

# --- Utility helpers ---
async def fetch_transactions(user_id: str, months: int = 12) -> List[Dict[str, Any]]:
    url = f"{CASHFLOW_URL}/api/v1/transactions"
    params = {"user_id": user_id, "months": months}
    resp = await http_client.get(url, params=params, timeout=30.0)
    if resp.status_code != 200:
        logger.error("Failed to fetch transactions: %s %s", resp.status_code, resp.text)
        raise HTTPException(status_code=502, detail="Failed to fetch transactions")
    data = resp.json()
    return data.get("transactions", [])

async def fetch_budgets(user_id: str):
    url = f"{BUDGET_URL}/api/v1/budgets"
    params = {"user_id": user_id}
    resp = await http_client.get(url, params=params, timeout=20.0)
    if resp.status_code != 200:
        logger.warning("Failed to fetch budgets: %s", resp.status_code)
        return []
    return resp.json().get("budgets", [])

# Prophet forecasting function (blocking) -> run in threadpool
def _prophet_forecast_series(ts: pd.DataFrame, periods: int = 30) -> pd.DataFrame:
    # ts: DataFrame with columns ['ds','y']
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    try:
        m.fit(ts)
    except Exception as e:
        logger.exception("Prophet fit failed: %s", e)
        raise
    future = m.make_future_dataframe(periods=periods, freq='D')
    forecast = m.predict(future)
    return forecast

async def forecast_category_history(history: List[Dict[str, Any]], days_ahead: int = 30) -> float:
    # Convert history list of {'date': 'YYYY-MM-DD', 'amount': num} into prophet format
    if not history:
        return 0.0
    df = pd.DataFrame([{"ds": h.get("date"), "y": float(h.get("amount", 0))} for h in history])
    df['ds'] = pd.to_datetime(df['ds'])
    # Aggregate to daily sums if needed
    df = df.groupby('ds', as_index=False).sum()
    # Ensure at least reasonable rows
    if len(df) < 3:
        # fallback: average
        return float(df['y'].mean())
    forecast = await run_in_threadpool(_prophet_forecast_series, df, days_ahead)
    # Sum the forecasted period
    future_forecast = forecast.tail(days_ahead)
    total = float(future_forecast['yhat'].sum())
    return total

# Anomaly detection (simple)
def detect_spike(current: float, history: List[float], pct_threshold: float = 0.3) -> bool:
    if not history:
        return False
    avg = statistics.mean(history)
    if avg == 0:
        return current > 0
    return (current - avg) / avg >= pct_threshold

# --- LLM reasoning wrapper ---
async def llm_summarize(prompt: str, max_tokens: int = 300) -> str:
    """
    Calls Groq API for chat completion.
    """
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured")

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.2,
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload
            )
        if resp.status_code != 200:
            logger.error("Groq API error: %s %s", resp.status_code, resp.text)
            raise HTTPException(status_code=502, detail="Groq API request failed")

        data = resp.json()
        return data["choices"][0]["message"]["content"]

    except Exception as e:
        logger.exception("Groq API call failed")
        raise HTTPException(status_code=500, detail=str(e))

"""
# LLM reasoning wrapper with Open AI
async def llm_summarize(prompt: str, max_tokens: int = 300) -> str:
    if LANGCHAIN_AVAILABLE:
        client = ChatOpenAI(model=OPENAI_MODEL, temperature=0.2)
        # LangChain's interface is synchronous; it may support async in some versions. Use blocking call via run_in_threadpool
        def call():
            return client.predict(prompt)
        out = await run_in_threadpool(call)
        return out
    else:
        # direct OpenAI completion (chat)
        def call_openai():
            resp = openai.ChatCompletion.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.2,
            )
            return resp
        resp = await run_in_threadpool(call_openai)
        try:
            return resp['choices'][0]['message']['content']
        except Exception:
            return str(resp)
"""

# --- Endpoint implementations ---
@app.post("/forecast/essentials", response_model=ForecastResponse)
async def forecast_essentials(req: ForecastRequest, background_tasks: BackgroundTasks):
    # 1. fetch transactions for categories
    transactions = await fetch_transactions(req.user_id, months=req.history_months)
    # Group by category and collect monthly sums (we will approximate daily series)
    cat_hist = {}
    for t in transactions:
        cat = t.get('category', 'uncategorized')
        cat_hist.setdefault(cat, []).append({
            'date': t.get('date'),
            'amount': t.get('amount')
        })
    categories = req.category_list or list(cat_hist.keys())
    predictions = {}
    warnings = []
    # Forecast each requested category
    for c in categories:
        hist = cat_hist.get(c, [])
        total_pred = await forecast_category_history(hist, days_ahead=30)
        # convert daily sum to monthly estimate (approx)
        monthly_estimate = total_pred
        predictions[c] = round(monthly_estimate, 2)
    # Compare with income (try to fetch budgets or user income)
    budgets = await fetch_budgets(req.user_id)
    income = None
    for b in budgets:
        if b.get('type') == 'income':
            income = float(b.get('monthly', income or 0))
    predicted_total = sum(predictions.values())
    if income and predicted_total > income:
        warnings.append(f"Predicted expenses may exceed projected income by ₹{round(predicted_total - income,2)}.")
    # Save log to MongoDB asynchronously
    log_doc = {
        'user_id': req.user_id,
        'type': 'forecast_essentials',
        'input': req.dict(),
        'result': predictions,
        'warnings': warnings,
        'created_at': datetime.datetime.utcnow()
    }
    background_tasks.add_task(mongo_db.ai_logs.insert_one, log_doc)
    return ForecastResponse(predictions=predictions, warnings=warnings)

@app.post("/advice/savings", response_model=SavingsResponse)
async def advice_savings(req: SavingsRequest, background_tasks: BackgroundTasks):
    # Fetch income from budgets or request payload
    income = req.income
    budgets = await fetch_budgets(req.user_id)
    if not income:
        for b in budgets:
            if b.get('type') == 'income':
                income = float(b.get('monthly', income or 0))
    if not income:
        raise HTTPException(status_code=400, detail="Income required either in payload or in budgets service")
    transactions = await fetch_transactions(req.user_id, months=req.history_months)
    # Sum essentials by category heuristics (could be improved)
    essentials = [t for t in transactions if t.get('category') in ('rent','groceries','EMI','school_fees')]
    essentials_sum = sum([float(t.get('amount',0)) for t in essentials])
    # detect surprise costs (anomalies in last months)
    # build per-month totals
    df = pd.DataFrame(transactions)
    if df.empty:
        safe_to_save = round(income * 0.2,2)
        reasoning = "No transaction history found; recommending conservative 20% savings."
    else:
        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.to_period('M')
        monthly = df.groupby('month')['amount'].sum().astype(float)
        avg_month = monthly.mean()
        # identify months with spikes
        spikes = [m for m in monthly if m > avg_month * 1.25]
        safety_buffer = float(len(spikes) / max(1, len(monthly)) * avg_month * 0.5)
        safe_to_save = float(max(0, income - avg_month - safety_buffer))
        reasoning = f"Based on avg monthly spend ₹{round(avg_month,2)} and spikes buffer ₹{round(safety_buffer,2)}."
    doc = {
        'user_id': req.user_id,
        'type': 'savings_advice',
        'input': req.dict(),
        'result': {'safe_to_save': safe_to_save, 'reasoning': reasoning},
        'created_at': datetime.datetime.utcnow()
    }
    background_tasks.add_task(mongo_db.ai_logs.insert_one, doc)
    return SavingsResponse(safe_to_save=round(safe_to_save,2), reasoning=reasoning)

@app.post("/alerts/festival-budget", response_model=FestivalBudgetResponse)
async def festival_budget(req: FestivalBudgetRequest, background_tasks: BackgroundTasks):
    # For each event, look up past event spending (we'll assume cashflow has tag 'event')
    transactions = await fetch_transactions(req.user_id, months=36)
    df = pd.DataFrame(transactions)
    suggestions = []
    festival_budgets = {}
    if df.empty:
        # fallback conservative budgets
        for e in req.upcoming_events:
            festival_budgets[e] = 5000
            suggestions.append(f"No history found; assign a conservative budget for {e}.")
    else:
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
        # assume events labeled in transaction.meta.event
        for e in req.upcoming_events:
            event_tx = df[df['meta'].apply(lambda m: isinstance(m, dict) and m.get('event')==e if m is not None else False)]
            if event_tx.empty:
                # try fuzzy approach: look at months known for event (e.g., Diwali -> Oct/Nov)
                # quick map (simplified)
                month_map = {'Diwali':[10,11], 'Pongal':[1], 'Raksha Bandhan':[8]}
                months = month_map.get(e, [])
                proxy = df[df['date'].dt.month.isin(months)] if months else pd.DataFrame()
                if proxy.empty:
                    festival_budgets[e] = 5000
                    suggestions.append(f"No direct history for {e}; fallback budget ₹5,000.")
                else:
                    yearly = proxy.groupby('year')['amount'].sum()
                    avg = yearly.mean() if not yearly.empty else 5000
                    # simple inflation adjustment 6% per year for gap
                    years = max(1, datetime.datetime.now().year - yearly.index.min() if not yearly.empty else 0)
                    adjusted = avg * (1.06 ** years)
                    festival_budgets[e] = round(float(adjusted),2)
                    suggestions.append(f"Estimate for {e} based on proxy months.")
            else:
                yearly = event_tx.groupby('year')['amount'].sum()
                avg = yearly.mean()
                adjusted = avg * 1.06  # adjust one year inflation
                festival_budgets[e] = round(float(adjusted),2)
                suggestions.append(f"Based on past events for {e}.")
    # store log
    doc = {
        'user_id': req.user_id,
        'type': 'festival_budget',
        'input': req.dict(),
        'result': festival_budgets,
        'suggestions': suggestions,
        'created_at': datetime.datetime.utcnow()
    }
    background_tasks.add_task(mongo_db.ai_logs.insert_one, doc)
    return FestivalBudgetResponse(festival_budgets=festival_budgets, suggestions=suggestions)

@app.get("/score/spending-health/{user_id}")
async def spending_health(user_id: str, months: int = 6):
    # fetch transactions and budgets
    tx = await fetch_transactions(user_id, months=months)
    budgets = await fetch_budgets(user_id)

    # compute simple score
    df = pd.DataFrame(tx)
    positive = []
    negative = []
    score = 50
    income = None

    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.to_period('M')
        monthly = df.groupby('month')['amount'].sum()

        # saved more than last month? check trend
        if len(monthly) >= 2 and monthly.iloc[-1] < monthly.iloc[-2]:
            score += 10
            positive.append('Saved more than last month')

        # bills paid on time -> assume flags in tx.meta.paid_on_time
        paid_on_time = df['meta'].apply(lambda m: isinstance(m, dict) and m.get('paid_on_time') == True if m is not None else False).sum()
        if paid_on_time > 0:
            score += min(10, paid_on_time)
            positive.append('Bills paid on time')

        # impulse spends: category 'food_delivery' or 'entertainment'
        impulse = df[df['category'].isin(['food_delivery', 'entertainment'])]['amount'].sum()
        if impulse > (monthly.mean() * 0.2):
            score -= 15
            negative.append(f'High impulse spends ₹{round(impulse, 2)}')

        # EMI ratio
        emi_total = df[df['category'] == 'EMI']['amount'].sum()

        # approximate income from budgets
        for b in budgets:
            if b.get('type') == 'income':
                income = float(b.get('monthly'))
                break

        if income and emi_total / income > 0.4:
            score -= 20
            negative.append('EMI exceeds 40% of income')
    else:
        negative.append('No transaction history')

    # Create a brief summary of recent cashflow trends
    tx_summary = ""
    if not df.empty:
        # Summarize total spent last 3 months per category
        #recent_months = df[df['date'] >= (pd.Timestamp.utcnow() - pd.DateOffset(months=3))]
        cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.DateOffset(months=3)
        recent_months = df[df['date'] >= cutoff]

        if not recent_months.empty:
            cat_sums = recent_months.groupby('category')['amount'].sum()
            cat_summaries = [f"{cat}: ₹{round(amount, 2)}" for cat, amount in cat_sums.items()]
            tx_summary = "; ".join(cat_summaries)
        else:
            tx_summary = "No significant recent transactions."
    else:
        tx_summary = "No transaction data available."

    # Format budgets summary as string for prompt clarity
    budgets_summary = ", ".join([f"{b.get('type')}: ₹{b.get('monthly')}" for b in budgets]) if budgets else "No budget data available."

    # Generate AI recommendation (short) with detailed prompt
    context = (
        f"You are a personal finance advisor. "
        f"The user's spending health score is {score} out of 100, where 100 means excellent financial habits and 0 means very poor. "
        f"Monthly income: ₹{income if income else 'Unknown'}.\n"
        f"Budget summary: {budgets_summary}.\n"
        f"Recent cashflow trends (last 3 months): {tx_summary}.\n"
        f"Positive financial habits observed: {positive}.\n"
        f"Areas needing improvement: {negative}.\n"
        f"Based on this information, give exactly 1–2 short, specific, and actionable recommendations "
        f"to improve the user's financial health over the next 1–3 months. "
        f"Format the output as a concise numbered list."
    )

    ai_text = await llm_summarize(context)
    doc = {
        'user_id': user_id,
        'type': 'spending_health',
        'score': score,
        'positive': positive,
        'negative': negative,
        'ai_advice': ai_text,
        'created_at': datetime.datetime.utcnow()
    }
    await mongo_db.ai_logs.insert_one(doc)
    return {
        'score': score,
        'grade': 'Good' if score>=75 else ('Fair' if score>=50 else 'Poor'),
        'positive_factors': positive,
        'negative_factors': negative,
        'recommendations': ai_text
    }

@app.post("/monitor/bills", response_model=BillMonitorResponse)
async def monitor_bills(req: BillMonitorRequest, background_tasks: BackgroundTasks):
    alerts = []
    # fetch 6 months historical charges for these bill types
    # We assume cashflow stores bills with category matching the bill type
    transactions = await fetch_transactions(req.user_id, months=6)
    df = pd.DataFrame(transactions)

    for bill in req.bills:
        hist = df[df['category'] == bill.type]['amount'].tolist() if not df.empty else []
        avg_hist = sum(hist)/len(hist) if hist else 0

        if not hist:
            # No history, maybe skip or warn no data available
            continue

        # Define thresholds, e.g. 30% increase or decrease
        increase_threshold = 0.3
        decrease_threshold = 0.3

        is_spike_up = (bill.amount - avg_hist)/avg_hist >= increase_threshold if avg_hist > 0 else False
        is_spike_down = (avg_hist - bill.amount)/avg_hist >= decrease_threshold if avg_hist > 0 else False

        if is_spike_up or is_spike_down:
            if is_spike_up:
                issue_text = "Amount is higher than usual"
                advice_intro = "unexpected increase"
            else:
                issue_text = "Amount is lower than usual"
                advice_intro = "unexpected decrease or unusually low amount"

            context = (
                f"You are a financial advisor AI. The user has a bill of type '{bill.type}' "
                f"with a current amount ₹{bill.amount}. "
                f"Historical amounts for this bill over the last 6 months are: {hist[:12]}. "
                f"There is an {advice_intro} compared to their average bill amount ₹{round(avg_hist,2)}. "
                f"Provide 1-2 concise, practical suggestions to help the user understand or manage this situation. "
                f"Keep the advice clear and actionable.\n\n"
                f"Respond ONLY with a JSON object matching this format:\n"
                f"{{\n"
                f'  "bill": "{bill.type}",\n'
                f'  "issue": "{issue_text}",\n'
                f'  "suggestion": "Your suggestions here."\n'
                f"}}"
            )
            print("LLM prompt context:\n", context)

            advice = await llm_summarize(context)

            try:
                alert_obj = json.loads(advice)
            except json.JSONDecodeError:
                alert_obj = {
                    "bill": bill.type,
                    "issue": issue_text,
                    "suggestion": advice.strip()
                }

            alerts.append(alert_obj)


    # Log to MongoDB asynchronously
    """
    doc = {
        'user_id': req.user_id,
        'type': 'bill_monitor',
        'input': req.dict(),
        'alerts': alerts,
        'created_at': datetime.datetime.utcnow()
    }
    background_tasks.add_task(mongo_db.ai_logs.insert_one, doc)
    """

    return BillMonitorResponse(alerts=alerts)


# --- Startup / Shutdown events ---
@app.on_event("startup")
async def startup_event():
    global mongo_client, mongo_db, http_client
    mongo_client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
    mongo_db = mongo_client['ai_reasoning_db']
    http_client = httpx.AsyncClient()
    logger.info("AI Reasoning Service started")

@app.on_event("shutdown")
async def shutdown_event():
    global mongo_client, http_client
    if mongo_client:
        mongo_client.close()
    if http_client:
        await http_client.aclose()
    logger.info("AI Reasoning Service stopped")

# --- Health check ---
@app.get("/health")
async def health():
    return {"status": "ok", "time": datetime.datetime.utcnow().isoformat()}

# EOF

