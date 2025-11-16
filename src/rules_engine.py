# src/rules_engine.py
def generate_recommendations(metrics, thresholds=None):
    # metrics: dict containing keys like turnover_rate, compa_ratio, training_coverage, avg_performance
    recs = []
    # thresholds default
    if thresholds is None:
        thresholds = {
            'turnover_high': 12,
            'compa_low': 0.95,
            'training_low': 50,
            'performance_low': 3.0
        }
    if metrics.get('turnover_rate',0) > thresholds['turnover_high']:
        recs.append("Turnover is high. Prioritize stay interviews and retention programs for high-risk roles.")
    if metrics.get('CompaRatio', metrics.get('compa_ratio',1.0)) < thresholds['compa_low']:
        recs.append("Average compa-ratio below market. Review salary bands and adjust pay for critical roles.")
    if metrics.get('training_coverage',100) < thresholds['training_low']:
        recs.append("Training coverage is low. Increase targeted training, especially for low-performing teams.")
    if metrics.get('avg_performance',5) < thresholds['performance_low']:
        recs.append("Average performance rating is low. Introduce performance improvement plans and coaching.")
    # Financial rule
    if metrics.get('labour_cost_pct_of_revenue') and metrics['labour_cost_pct_of_revenue'] > 40:
        recs.append("Labour cost exceeds 40% revenue. Consider process automation or cost rebalancing.")
    if not recs:
        recs.append("No urgent issues detected — maintain current strategies and monitor KPIs.")
    return recs
