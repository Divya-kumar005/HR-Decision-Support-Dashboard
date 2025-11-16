# src/preprocessing.py
import pandas as pd
import numpy as np

def load_ibm(path="data/ibm_hr_attrition.csv"):
    df = pd.read_csv(path)
    # Standardize Attrition to binary
    if 'Attrition' in df.columns:
        df['AttritionFlag'] = df['Attrition'].apply(lambda x: 1 if x.strip().lower()=='yes' else 0)
    # Annualize salaries if MonthlyIncome present
    if 'MonthlyIncome' in df.columns:
        df['AnnualIncome'] = df['MonthlyIncome'] * 12
    # Fill missing numeric values with sensible defaults
    for col in ['YearsAtCompany','TrainingTimesLastYear','PerformanceRating','JobSatisfaction','EnvironmentSatisfaction']:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    # Benefits (estimate) and total comp
    df['Benefits'] = df['AnnualIncome'] * 0.20  # configurable assumption
    df['TotalAnnualCompensation'] = df['AnnualIncome'] + df['Benefits']
    return df

def add_market_midpoint(df, midpoints_path="data/market_midpoints.xlsx"):
    mp = pd.read_csv(midpoints_path)
    df = df.merge(mp[['JobRole','MarketMidpoint']], how='left', on='JobRole')
    df['MarketMidpoint'] = df['MarketMidpoint'].fillna(df['AnnualIncome'].median())
    df['CompaRatio'] = df['AnnualIncome'] / df['MarketMidpoint']
    return df

def synthesize_recruitment(path_out="data/recruitment_events.csv", n=100):
    import random
    from datetime import datetime, timedelta
    roles = list(pd.read_csv("data/ibm_hr_attrition.csv")['JobRole'].unique())
    rows=[]
    start = datetime(2023,1,1)
    for i in range(n):
        open_date = start + timedelta(days=random.randint(0,300))
        fill_days = random.randint(10,70)
        filled_date = open_date + timedelta(days=fill_days)
        offers = random.randint(1,6)
        offers_accepted = 1 if random.random()<0.7 else 0
        cost = random.randint(1500,9000)
        rows.append({
            'job_id': f'JOB{i+1}',
            'job_role': random.choice(roles),
            'open_date': open_date.date(),
            'filled_date': filled_date.date(),
            'num_offers': offers,
            'offers_accepted': offers_accepted,
            'cost_of_recruitment': cost
        })
    pd.DataFrame(rows).to_csv(path_out, index=False)
    print(f"Synthesized recruitment events saved to {path_out}")
