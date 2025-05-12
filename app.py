from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from deltalake import DeltaTable
import pandas as pd
from typing import List, Union

app = FastAPI()

# Path to the Delta Lake tables
delta_lake_base_path = "./my_delta_lake"
applications_table_path = f"{delta_lake_base_path}/applications_data"
underwriting_table_path = f"{delta_lake_base_path}/underwriting_data"

# Helper function to read Delta Table into pandas
def read_delta_table(table_path: str):
    try:
        delta_table = DeltaTable(table_path)
        # Load the table data into pandas DataFrame
        return delta_table.to_pandas()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading Delta table: {str(e)}")

# Pydantic Models
class ApplicationDetails(BaseModel):
    age: int
    occupation: str
    coverageType: str
    coverageAmount: str

class Application(BaseModel):
    name: str
    riskScore: Union[int, str]
    status: str
    reviewer: str
    date: str
    details: ApplicationDetails

class RiskSegment(BaseModel):
    name: str
    value: int
    color: str

class ApplicationStats(BaseModel):
    totalReviewed: int
    pendingUnderwriting: int
    approved: int
    declined: int

class RiskDistribution(BaseModel):
    segments: List[RiskSegment]

# API Endpoints
@app.get("/applicationstats", response_model=ApplicationStats)
async def get_application_stats():
    # Read data from the Delta Lake applications table
    df = read_delta_table(applications_table_path)
    
    total_reviewed = len(df)
    pending_underwriting = len(df[df['status'] == "Needs info"])
    approved = len(df[df['status'] == "Approved"])
    declined = len(df[df['status'] == "Declined"])
    
    return ApplicationStats(
        totalReviewed=total_reviewed,
        pendingUnderwriting=pending_underwriting,
        approved=approved,
        declined=declined
    )

@app.get("/riskdistribution", response_model=RiskDistribution)
async def get_risk_distribution():
    # Sample predefined risk segments data
    risk_segments = [
        {"name": "High", "value": 35, "color": "#ef4444"},
        {"name": "Medium", "value": 40, "color": "#f97316"},
        {"name": "Low", "value": 25, "color": "#22c55e"}
    ]
    return RiskDistribution(segments=[RiskSegment(**segment) for segment in risk_segments])

@app.get("/applications", response_model=List[Application])
async def get_applications():
    # Read data from the Delta Lake applications table
    df = read_delta_table(applications_table_path)
    
    applications = []
    for _, row in df.iterrows():
        details = ApplicationDetails(
            age=row['age'],
            occupation=row['occupation'],
            coverageType=row['coverageType'],
            coverageAmount=row['coverageAmount']
        )
        application = Application(
            name=row['name'],
            riskScore=row['riskScore'],
            status=row['status'],
            reviewer=row['reviewer'],
            date=row['date'],
            details=details
        )
        applications.append(application)
    
    return applications

# Run the FastAPI app with uvicorn
# uvicorn app:app --reload
