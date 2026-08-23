import pandas as pd
import numpy as np
from typing import Dict, List, Any
from src import config

class StockoutEngine:
    """Evaluates stockout risks, calculates urgency thresholds, and generates reorder proposals."""

    @staticmethod
    def calculate_stockout_probability(days_to_stockout: float, lead_time_days: float) -> float:
        """
        Calculates probability of stockout before warehouse replenishment arrives.
        Uses a smooth logistic function anchored around the lead time buffer.
        """
        if days_to_stockout <= 0:
            return 1.0
        if days_to_stockout >= 100.0:
            return 0.01
            
        # If days to stockout is less than lead time, probability is very high
        z = (lead_time_days * 1.2 - days_to_stockout) / max(1.0, lead_time_days * 0.5)
        z_clipped = float(np.clip(z, -15.0, 15.0))
        prob = 1.0 / (1.0 + np.exp(-2.0 * z_clipped))
        return float(np.clip(prob, 0.01, 0.99))

    @classmethod
    def evaluate_inventory_risk(
        cls,
        forecast_df: pd.DataFrame,
        inventory_df: pd.DataFrame,
        facilities_df: pd.DataFrame,
        medicines_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merges forecasts with real-time inventory and facility lead-times to score stockout risk.
        """
        # Get latest inventory snapshot for each PHC and Drug
        latest_inv = inventory_df.sort_values(by='date').groupby(['phc_id', 'drug_id']).last().reset_index()
        
        # Merge with facilities to get warehouse lead times
        latest_inv = pd.merge(
            latest_inv,
            facilities_df[['phc_id', 'district', 'warehouse_lead_time_days', 'distance_to_district_warehouse_km']],
            on='phc_id',
            how='left'
        )
        
        # Merge with medicines master
        latest_inv = pd.merge(
            latest_inv,
            medicines_df[['drug_id', 'category', 'unit_cost_inr']],
            on='drug_id',
            how='left'
        )
        
        # Merge with predictions
        eval_df = pd.merge(
            latest_inv,
            forecast_df[['phc_id', 'drug_id', 'target_7d_total'] + [f'target_day_{i}' for i in range(1, 8)]],
            on=['phc_id', 'drug_id'],
            how='inner'
        )
        
        eval_df['predicted_consumption_7d'] = eval_df['target_7d_total'].round(1)
        eval_df['daily_burn_rate'] = (eval_df['predicted_consumption_7d'] / 7.0).round(2)
        
        # Days to stockout
        eval_df['days_to_stockout'] = np.where(
            eval_df['daily_burn_rate'] > 0,
            (eval_df['closing_stock'] / eval_df['daily_burn_rate']).round(2),
            999.0
        )
        
        # Calculate stockout probability
        eval_df['stockout_probability'] = eval_df.apply(
            lambda row: cls.calculate_stockout_probability(
                row['days_to_stockout'],
                row['warehouse_lead_time_days']
            ),
            axis=1
        ).round(3)
        
        # Determine Alert Level
        conditions = [
            (eval_df['days_to_stockout'] <= eval_df['warehouse_lead_time_days']) | 
            (eval_df['days_to_stockout'] <= config.STOCKOUT_CRITICAL_DAYS) | 
            (eval_df['closing_stock'] < eval_df['safety_stock']),
            
            (eval_df['days_to_stockout'] <= config.STOCKOUT_WARNING_DAYS) | 
            (eval_df['closing_stock'] < eval_df['reorder_point'])
        ]
        choices = ['CRITICAL', 'WARNING']
        eval_df['alert_level'] = np.select(conditions, choices, default='SAFE')
        
        # Calculate Deficit / Shortfall Quantity
        # Required buffer = (daily_burn_rate * lead_time) + safety_stock
        eval_df['required_stock_buffer'] = (
            eval_df['daily_burn_rate'] * eval_df['warehouse_lead_time_days'] + eval_df['safety_stock']
        ).round(0)
        
        eval_df['shortfall_qty'] = np.maximum(
            0,
            (eval_df['required_stock_buffer'] - eval_df['closing_stock']).round(0)
        ).astype(int)
        
        # Calculate Surplus Quantity (for potential transfer donors in Module 4)
        eval_df['surplus_qty'] = np.maximum(
            0,
            (eval_df['closing_stock'] - eval_df['required_stock_buffer']).round(0)
        ).astype(int)
        
        # Actionable Recommendation Text
        def generate_recommendation(row):
            if row['alert_level'] == 'CRITICAL':
                return f"URGENT: Stockout in {row['days_to_stockout']}d. Shortfall: {row['shortfall_qty']} units. Initiate immediate warehouse reorder or peer transfer."
            elif row['alert_level'] == 'WARNING':
                return f"CAUTION: Stock below reorder threshold. Projected depletion in {row['days_to_stockout']}d. Place standard procurement order."
            else:
                return "HEALTHY: Inventory levels adequate for projected 7-day demand."
                
        eval_df['recommendation'] = eval_df.apply(generate_recommendation, axis=1)
        
        return eval_df

    @staticmethod
    def generate_procurement_orders(risk_df: pd.DataFrame, current_date: pd.Timestamp) -> List[Dict[str, Any]]:
        """Generates structured purchase orders for district warehouse procurement."""
        orders = []
        critical_and_warning = risk_df[risk_df['alert_level'].isin(['CRITICAL', 'WARNING'])]
        
        for _, row in critical_and_warning.iterrows():
            lead_days = int(row['warehouse_lead_time_days'])
            order_qty = max(row['shortfall_qty'], int(row['reorder_point'] - row['closing_stock'] + row['safety_stock']))
            if order_qty <= 0:
                order_qty = int(row['safety_stock'])
                
            orders.append({
                "phc_id": row['phc_id'],
                "district": row['district'],
                "drug_id": row['drug_id'],
                "category": row['category'],
                "current_stock": int(row['closing_stock']),
                "days_to_stockout": float(row['days_to_stockout']),
                "alert_level": row['alert_level'],
                "recommended_order_qty": int(order_qty),
                "estimated_cost_inr": round(float(order_qty * row['unit_cost_inr']), 2),
                "lead_time_days": lead_days,
                "order_placed_date": current_date.strftime("%Y-%m-%d"),
                "estimated_arrival_date": (current_date + pd.Timedelta(days=lead_days)).strftime("%Y-%m-%d"),
                "urgency": "HIGH" if row['alert_level'] == 'CRITICAL' else "MEDIUM"
            })
            
        return orders
