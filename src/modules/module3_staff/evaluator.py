import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from src import config

class StaffingEvaluator:
    """Rule-based compliance and demand-weighted shortage evaluator for medical personnel."""

    @staticmethod
    def calculate_shortage_severity(
        role_shortfalls: Dict[str, int],
        role_required: Dict[str, int],
        bed_occupancy_pct: float = 50.0,
        patient_footfall_ratio: float = 1.0
    ) -> Dict[str, Any]:
        """
        Calculates role-level and overall severity scores weighted by facility operational demand.
        """
        role_scores = {}
        for role in ['doctor', 'nurse', 'pharmacist']:
            req = max(1, role_required.get(role, 1))
            short = max(0, role_shortfalls.get(role, 0))
            role_scores[role] = short / req
            
        # Demand weighting multiplier
        mult = 1.0
        if bed_occupancy_pct > config.BED_OCCUPANCY_CRITICAL_PCT:
            mult *= 1.5
        elif bed_occupancy_pct > config.BED_OCCUPANCY_WARNING_PCT:
            mult *= 1.25
            
        if patient_footfall_ratio > 1.2:
            mult *= 1.3
            
        # Role weighted composite score (doctor shortages have highest weight)
        composite_score = (
            role_scores['doctor'] * 0.55 +
            role_scores['nurse'] * 0.30 +
            role_scores['pharmacist'] * 0.15
        ) * mult
        
        # Hard threshold: if half or more doctors are missing, escalate to CRITICAL
        doctor_crisis = (role_shortfalls.get('doctor', 0) >= max(1, role_required.get('doctor', 2) / 2.0))
        
        if doctor_crisis or composite_score >= 0.8:
            severity = 'CRITICAL'
        elif composite_score >= 0.5:
            severity = 'HIGH'
        elif composite_score >= 0.2:
            severity = 'MEDIUM'
        else:
            severity = 'LOW' if composite_score > 0 else 'HEALTHY'
            
        return {
            "composite_score": round(float(composite_score), 3),
            "severity": severity,
            "demand_multiplier": round(float(mult), 2),
            "doctor_crisis": bool(doctor_crisis)
        }

    @classmethod
    def evaluate_facility_staffing(
        cls,
        attendance_df: pd.DataFrame,
        facilities_df: pd.DataFrame,
        current_date: pd.Timestamp,
        bed_df: Optional[pd.DataFrame] = None,
        tx_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Evaluates daily staffing performance across all facilities for current_date.
        """
        # Filter attendance for current date
        daily_att = attendance_df[attendance_df['date'] == current_date]
        
        # Aggregate by phc_id and role
        summary = daily_att.groupby(['phc_id', 'role']).agg(
            scheduled_count=('scheduled', 'sum'),
            present_count=('present', 'sum')
        ).reset_index()
        summary['shortfall'] = np.maximum(0, summary['scheduled_count'] - summary['present_count'])
        
        # Prepare lookup per facility
        results = []
        for _, fac in facilities_df.iterrows():
            pid = fac['phc_id']
            fac_summary = summary[summary['phc_id'] == pid].set_index('role')
            
            # Roles required from master
            req_doc = int(fac['doctors_required'])
            req_nurse = int(fac['nurses_required'])
            req_pharm = int(fac['pharmacists_required'])
            
            # Present count
            pres_doc = int(fac_summary.loc['doctor', 'present_count']) if 'doctor' in fac_summary.index else req_doc
            pres_nurse = int(fac_summary.loc['nurse', 'present_count']) if 'nurse' in fac_summary.index else req_nurse
            pres_pharm = int(fac_summary.loc['pharmacist', 'present_count']) if 'pharmacist' in fac_summary.index else req_pharm
            
            short_doc = max(0, req_doc - pres_doc)
            short_nurse = max(0, req_nurse - pres_nurse)
            short_pharm = max(0, req_pharm - pres_pharm)
            
            # Get bed occupancy % for this facility
            occ_pct = 50.0
            if bed_df is not None:
                b_match = bed_df[(bed_df['date'] == current_date) & (bed_df['phc_id'] == pid)]
                if len(b_match) > 0:
                    occ_pct = float(b_match['occupancy_pct'].iloc[0])
                    
            # Get footfall ratio
            ff_ratio = 1.0
            if tx_df is not None:
                tx_match = tx_df[(tx_df['date'] == current_date) & (tx_df['phc_id'] == pid)]
                if len(tx_match) > 0:
                    footfall = float(tx_match['patients_served_estimate'].iloc[0])
                    expected_footfall = fac['population_served'] / 6000.0
                    ff_ratio = footfall / max(1.0, expected_footfall)
                    
            # Score
            sev_info = cls.calculate_shortage_severity(
                role_shortfalls={'doctor': short_doc, 'nurse': short_nurse, 'pharmacist': short_pharm},
                role_required={'doctor': req_doc, 'nurse': req_nurse, 'pharmacist': req_pharm},
                bed_occupancy_pct=occ_pct,
                patient_footfall_ratio=ff_ratio
            )
            
            # Recommendations
            recs = []
            if short_doc > 0:
                recs.append(f"Doctor deficit ({short_doc} needed): Call on-call roster or request visiting physician.")
            if short_nurse > 0:
                recs.append(f"Nurse deficit ({short_nurse} needed): Reassign shift tasks or call backup reserve.")
            if short_pharm > 0:
                recs.append(f"Pharmacist deficit ({short_pharm} needed): Enable supervised auxiliary dispensing.")
            if not recs:
                recs.append("Full staffing compliance: All critical clinical shifts staffed.")
                
            # Operational Feasibility for receiving patient transfers
            # Rejects incoming transfers if CRITICAL shortage or missing >50% doctors/nurses
            is_feasible = (
                sev_info['severity'] != 'CRITICAL' and
                short_doc < max(1, req_doc / 2.0) and
                short_nurse < max(1, req_nurse / 2.0)
            )
            
            results.append({
                "phc_id": pid,
                "district": fac['district'],
                "date": current_date.strftime("%Y-%m-%d"),
                "doctors_required": req_doc,
                "doctors_present": pres_doc,
                "doctors_shortfall": short_doc,
                "nurses_required": req_nurse,
                "nurses_present": pres_nurse,
                "nurses_shortfall": short_nurse,
                "pharmacists_required": req_pharm,
                "pharmacists_present": pres_pharm,
                "pharmacists_shortfall": short_pharm,
                "bed_occupancy_pct": occ_pct,
                "composite_score": sev_info['composite_score'],
                "severity": sev_info['severity'],
                "operationally_feasible": bool(is_feasible),
                "recommendation": " | ".join(recs)
            })
            
        return pd.DataFrame(results)
