# src/metrics.py
import pandas as pd
import numpy as np

# Basic population helpers
def headcount(df):
    return df.shape[0]

def employees_left(df):
    return df[df['AttritionFlag']==1].shape[0]

def employees_stayed(df):
    return df[df['AttritionFlag']==0].shape[0]

# Recruitment metrics (if recruitment_events.csv available)
def time_to_fill(recruit_df):
    recruit_df['open_date'] = pd.to_datetime(recruit_df['open_date'])
    recruit_df['filled_date'] = pd.to_datetime(recruit_df['filled_date'])
    recruit_df['days_to_fill'] = (recruit_df['filled_date'] - recruit_df['open_date']).dt.days
    return recruit_df['days_to_fill'].mean()

def cost_per_hire(recruit_df):
    return recruit_df['cost_of_recruitment'].mean()

def offer_acceptance_rate(recruit_df):
    total_offers = recruit_df['num_offers'].sum()
    accepted = recruit_df['offers_accepted'].sum()
    return (accepted / total_offers) * 100 if total_offers>0 else None

# Retention & turnover
def turnover_rate(df):
    total_left = employees_left(df)
    avg_headcount = headcount(df)  # simple avg for static dataset
    return (total_left / avg_headcount) * 100

def retention_rate(df):
    return (employees_stayed(df) / headcount(df)) * 100

def stability_index(df):
    # % employees with >=1 year
    if 'YearsAtCompany' in df.columns:
        return (df[df['YearsAtCompany']>=1].shape[0] / headcount(df)) * 100
    return None

def avg_years_of_stay(df):
    if 'YearsAtCompany' in df.columns:
        return df['YearsAtCompany'].mean()
    return None

# Compensation
def labour_cost_per_fte(df):
    return df['TotalAnnualCompensation'].mean()

def labour_cost_total(df):
    return df['TotalAnnualCompensation'].sum()

def compa_ratio_avg(df):
    return df['CompaRatio'].mean()

# Performance & training
def training_coverage(df):
    if 'TrainingTimesLastYear' in df.columns:
        return (df[df['TrainingTimesLastYear']>0].shape[0] / headcount(df)) * 100
    return None

def avg_performance_rating(df):
    return df['PerformanceRating'].mean()

def percentage_goals_attained(df):
    # If dataset has a 'GoalsAttained' column use it; else compute from proxy
    if 'GoalsAttained' in df.columns and 'GoalsAssigned' in df.columns:
        return (df['GoalsAttained'].sum() / df['GoalsAssigned'].sum()) * 100
    return None

# Turnover costs
def cost_of_turnover_one_employee(avg_cost_per_hire=20000, onboarding_loss=10000, productivity_loss=15000):
    return avg_cost_per_hire + onboarding_loss + productivity_loss

def total_cost_of_turnover(df, avg_cost_replacement=45000):
    total_left = employees_left(df)
    return total_left * avg_cost_replacement
