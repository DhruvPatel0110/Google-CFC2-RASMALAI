import pandas as pd
from typing import Dict, List, Any, Optional
from src.data.loader import DataLoader
from src.modules.module1_medicine.service import MedicineModuleService
from src.modules.module2_beds.service import BedModuleService
from src.modules.module3_staff.service import StaffModuleService
from .network import FacilityNetwork
from .optimizer import RedistributionOptimizer

class RedistributionService:
    """End-to-End Service Facade for Module 4 (Cross-District Redistribution Engine)."""

    def __init__(
        self,
        medicine_service: Optional[MedicineModuleService] = None,
        bed_service: Optional[BedModuleService] = None,
        staff_service: Optional[StaffModuleService] = None
    ):
        self.medicine_service = medicine_service or MedicineModuleService()
        self.bed_service = bed_service or BedModuleService()
        self.staff_service = staff_service or StaffModuleService()
        self.network: Optional[FacilityNetwork] = None
        self.optimizer: Optional[RedistributionOptimizer] = None

    def initialize(self):
        """Initializes network topology and sub-module services."""
        if self.network is None:
            fac_df = DataLoader.load_facilities()
            dist_df = DataLoader.load_distance_matrix()
            self.network = FacilityNetwork(fac_df, dist_df)
            self.optimizer = RedistributionOptimizer(self.network)
            
            # Ensure sub-modules are initialized
            self.medicine_service.initialize_data()
            self.bed_service.initialize_data()
            self.staff_service.initialize_data()

    def generate_plan(self, as_of_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Executes end-to-end multi-module coordination and generates unified redistribution plan.
        """
        self.initialize()
        eval_date = as_of_date or "2025-08-31"
        
        # 1. Fetch Module 1 Signals
        med_deficits = self.medicine_service.get_critical_stockouts(eval_date)
        med_surpluses = self.medicine_service.get_surplus_facilities(eval_date)
        
        # 2. Fetch Module 2 Signals
        overcrowded_beds = self.bed_service.get_overcrowded_facilities(eval_date)
        surplus_beds = self.bed_service.get_capacity_surplus_facilities(eval_date)
        
        # Gather patient candidates for each overloaded facility
        patient_candidates: Dict[str, List[Dict[str, Any]]] = {}
        for fac in overcrowded_beds:
            pid = fac['phc_id']
            patient_candidates[pid] = self.bed_service.get_transferable_patients(pid, eval_date)
            
        # 3. Fetch Module 3 Staffing Gatekeeper
        staff_checker = lambda pid: self.staff_service.is_facility_operationally_feasible(pid, eval_date)
        
        # 4. Optimize
        plan = self.optimizer.optimize_network_redistribution(
            medicine_deficits=med_deficits,
            medicine_surpluses=med_surpluses,
            overcrowded_facilities=overcrowded_beds,
            bed_surpluses=surplus_beds,
            patient_candidates_by_facility=patient_candidates,
            staff_feasibility_checker=staff_checker,
            snapshot_date=eval_date
        )
        
        return plan

    def get_medicine_manifest(self, as_of_date: Optional[str] = None) -> List[Dict[str, Any]]:
        """Returns only medicine transfer orders."""
        plan = self.generate_plan(as_of_date)
        return plan.get('medicine_transfers', [])

    def get_patient_manifest(self, as_of_date: Optional[str] = None) -> List[Dict[str, Any]]:
        """Returns only patient transfer orders."""
        plan = self.generate_plan(as_of_date)
        return plan.get('patient_transfers', [])
