from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from src.data.loader import DataLoader
from src.modules.module1_medicine.service import MedicineModuleService
from src.modules.module2_beds.service import BedModuleService
from src.modules.module3_staff.service import StaffModuleService
from src.modules.module4_redistribution.service import RedistributionService
from src.modules.module5_federation.service import FederationModuleService

app = FastAPI(
    title="PHC Supply Chain Resilience API",
    description="Federated AI Platform for Health Resource Management, Forecasting, and Cross-District Redistribution",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for cross-origin access and browser dashboards
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize service singletons
medicine_service = MedicineModuleService()
bed_service = BedModuleService()
staff_service = StaffModuleService()
redis_service = RedistributionService(medicine_service, bed_service, staff_service)
federation_service = FederationModuleService()

@app.on_event("startup")
def startup_event():
    """Warm up and pre-load all datasets, feature stores, and models on server boot."""
    try:
        medicine_service.initialize_data()
        bed_service.initialize_data()
        staff_service.initialize_data()
        redis_service.initialize()
        federation_service.initialize_data()
    except Exception as e:
        print(f"[WARNING] API startup pre-initialization encountered an issue: {e}")

class FederationRequest(BaseModel):
    rounds: int = 5
    epsilon: float = 0.5
    enable_dp: bool = True

@app.get("/", include_in_schema=False)
def root_redirect():
    """Redirect root path to interactive Swagger documentation."""
    return RedirectResponse(url="/docs")

@app.get("/health")
def health_check():
    return {
        "status": "HEALTHY",
        "version": "1.0.0",
        "services": {
            "medicine_service": "ONLINE",
            "bed_service": "ONLINE",
            "staff_service": "ONLINE",
            "redistribution_service": "ONLINE",
            "federation_service": "ONLINE"
        }
    }

@app.get("/api/v1/network/facilities")
def get_facilities():
    df = DataLoader.load_facilities()
    return df.to_dict(orient="records")

@app.get("/api/v1/medicine/stockouts")
def get_medicine_stockouts(date: Optional[str] = Query(None, description="Snapshot date YYYY-MM-DD")):
    return medicine_service.get_critical_stockouts(date)

@app.get("/api/v1/medicine/facility/{phc_id}")
def get_phc_medicine(phc_id: str, date: Optional[str] = None):
    summary = medicine_service.get_facility_summary(phc_id, date)
    if summary.get("total_skus", 0) == 0:
        raise HTTPException(status_code=404, detail=f"PHC facility '{phc_id}' not found.")
    return summary

@app.get("/api/v1/beds/occupancy")
def get_bed_occupancy(date: Optional[str] = Query(None, description="Snapshot date YYYY-MM-DD")):
    df = bed_service.run_assessment(date)
    return df.to_dict(orient="records")

@app.get("/api/v1/beds/facility/{phc_id}")
def get_phc_beds(phc_id: str, date: Optional[str] = None):
    report = bed_service.get_facility_bed_report(phc_id, date)
    if "error" in report:
        raise HTTPException(status_code=404, detail=report["error"])
    return report

@app.get("/api/v1/beds/candidates/{phc_id}")
def get_transfer_candidates(phc_id: str, date: Optional[str] = None):
    return bed_service.get_transferable_patients(phc_id, date)

@app.get("/api/v1/staff/attendance")
def get_staff_attendance(date: Optional[str] = Query(None, description="Snapshot date YYYY-MM-DD")):
    df = staff_service.run_assessment(date)
    return df.to_dict(orient="records")

@app.get("/api/v1/staff/facility/{phc_id}")
def get_phc_staff(phc_id: str, date: Optional[str] = None):
    report = staff_service.get_facility_staffing_report(phc_id, date)
    if "error" in report:
        raise HTTPException(status_code=404, detail=report["error"])
    return report

@app.get("/api/v1/redistribution/plan")
def get_redistribution_plan(date: Optional[str] = Query(None, description="Snapshot date YYYY-MM-DD")):
    return redis_service.generate_plan(date)

@app.post("/api/v1/federation/simulate")
def run_federation_simulation(req: FederationRequest):
    return federation_service.run_simulation(
        n_rounds=req.rounds,
        epsilon=req.epsilon,
        enable_dp=req.enable_dp
    )

@app.get("/api/v1/federation/model-card")
def get_model_card():
    return federation_service.get_model_card()
