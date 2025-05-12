from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from deltalake import DeltaTable
import pandas as pd
from typing import List, Union
from datetime import datetime

# Initialize FastAPI app
app = FastAPI()

# Path to the Delta Lake tables
delta_lake_base_path = "./my_delta_lake"
applications_table_path = f"{delta_lake_base_path}/applications_data"
underwriting_table_path = f"{delta_lake_base_path}/underwriting_data"

# Helper function to read Delta Table into pandas DataFrame
def read_delta_table(table_path: str):
    try:
        delta_table = DeltaTable(table_path)
        # Load the table data into pandas DataFrame
        return delta_table.to_pandas()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading Delta table: {str(e)}")

# Pydantic Models to validate and format API responses
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

# API Endpoint to fetch application statistics
@app.get("/applicationstats", response_model=ApplicationStats)
async def get_application_stats():
    """
    Fetches total number of applications, pending, approved, and declined applications.
    """
    df = read_delta_table(applications_table_path)
    
    # Count total reviewed, pending, approved, and declined applications
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

# API Endpoint to fetch risk distribution with predefined segments
@app.get("/riskdistribution", response_model=RiskDistribution)
async def get_risk_distribution():
    """
    Returns predefined risk distribution segments (e.g., High, Medium, Low) with values and color.
    """
    risk_segments = [
        {"name": "High", "value": 35, "color": "#ef4444"},
        {"name": "Medium", "value": 40, "color": "#f97316"},
        {"name": "Low", "value": 25, "color": "#22c55e"}
    ]
    return RiskDistribution(segments=[RiskSegment(**segment) for segment in risk_segments])

# API Endpoint to fetch all application details
@app.get("/applications", response_model=List[Application])
async def get_applications():
    """
    Fetches all application details, including name, risk score, status, and additional details.
    """
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

# API Endpoint to fetch the new applications and percentage change vs the previous period
@app.get("/newapplications")
async def get_new_applications():
    """
    Fetches the total number of new applications in the current period and the percentage change compared to the previous period.
    """
    df = read_delta_table(applications_table_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # Filter applications for current and previous months
    current_period = df[df['date'].dt.month == datetime.now().month]
    previous_period = df[df['date'].dt.month == datetime.now().month - 1]
    
    current_count = len(current_period)
    previous_count = len(previous_period)
    
    # Calculate percentage change
    percentage_change = ((current_count - previous_count) / previous_count) * 100 if previous_count > 0 else 0
    
    return {"total_applications": current_count, "percentage_change": round(percentage_change, 2), "period": "May 2025"}

# API Endpoint to fetch the average processing time of applications
@app.get("/avgprocessingtime")
async def get_avg_processing_time():
    """
    Fetches the average processing time for applications, which is calculated as the difference between submission and policy start date.
    """
    df = read_delta_table(applications_table_path)
    df['date'] = pd.to_datetime(df['date'])
    df['policyStartDate'] = pd.to_datetime(df['policyStartDate'])
    df['processing_time'] = (df['policyStartDate'] - df['date']).dt.days
    
    # Calculate the average processing time
    avg_processing_time = df['processing_time'].mean()
    
    return {"avg_processing_time": round(avg_processing_time, 2), "period": "May 2025"}

# API Endpoint to fetch the application approval rate
@app.get("/approvalrate")
async def get_approval_rate():
    """
    Fetches the approval rate by calculating the percentage of applications that were approved.
    """
    df = read_delta_table(applications_table_path)
    
    approved_count = len(df[df['status'] == "Approved"])
    total_count = len(df)
    
    # Calculate the approval rate
    approval_rate = (approved_count / total_count) * 100 if total_count > 0 else 0
    
    return {"approval_rate": round(approval_rate, 2), "period": "May 2025"}

# API Endpoint to fetch underwriting decisions (Approved, Rejected, Pending)
@app.get("/underwritingdecisions")
async def get_underwriting_decisions():
    """
    Fetches the breakdown of underwriting decisions (Approved, Rejected, Pending).
    """
    df = read_delta_table(underwriting_table_path)
    
    # Count underwriting decisions by status
    underwriting_decisions = df['underwritingStatus'].value_counts()
    
    return {"underwriting_decisions": underwriting_decisions.to_dict(), "period": "May 2025"}

# API Endpoint to fetch policy distribution (e.g., Term Life, Whole Life)
@app.get("/policydistribution")
async def get_policy_distribution():
    """
    Fetches the distribution of different policy types (e.g., Term Life, Whole Life).
    """
    df = read_delta_table(applications_table_path)
    
    # Count policy types distribution
    policy_distribution = df['policyType'].value_counts()
    
    return {"policy_distribution": policy_distribution.to_dict()}

# API Endpoint to fetch risk factors (Risk Assessment)
@app.get("/riskfactors")
async def get_risk_factors():
    """
    Fetches the distribution of risk assessments (High, Medium, Low).
    """
    df = read_delta_table(underwriting_table_path)
    
    # Count risk assessments
    risk_factors = df['riskAssessment'].value_counts()
    
    return {"risk_factors": risk_factors.to_dict()}

# API Endpoint to fetch age vs risk distribution (Scatter Plot Data)
@app.get("/agevsrisk")
async def get_age_vs_risk():
    """
    Fetches data of age vs risk assessment for each application.
    This data is intended for a scatter plot on the frontend.
    """
    try:
        # Load data from applications and underwriting tables
        applications_df = read_delta_table(applications_table_path)
        underwriting_df = read_delta_table(underwriting_table_path)
        
        # Ensure necessary columns exist in both datasets
        if 'age' not in applications_df.columns or 'name' not in applications_df.columns:
            raise HTTPException(status_code=400, detail="Missing 'age' or 'name' column in applications data.")
        
        if 'riskAssessment' not in underwriting_df.columns or 'name' not in underwriting_df.columns:
            raise HTTPException(status_code=400, detail="Missing 'riskAssessment' or 'name' column in underwriting data.")
        
        # Remove NaN values
        applications_df = applications_df.dropna(subset=['age', 'name'])
        underwriting_df = underwriting_df.dropna(subset=['riskAssessment', 'name'])
        
        # Collect age and risk assessment data
        age_vs_risk_data = []
        
        for _, app_row in applications_df.iterrows():
            name = app_row['name']
            age = app_row['age']
            
            # Find corresponding underwriting record by 'name'
            underwriting_record = underwriting_df[underwriting_df['name'] == name]
            
            if not underwriting_record.empty:
                risk_assessment = underwriting_record.iloc[0]['riskAssessment']
                age_vs_risk_data.append({'age': age, 'riskAssessment': risk_assessment})
        
        if not age_vs_risk_data:
            raise HTTPException(status_code=404, detail="No matching data found between applications and underwriting datasets.")
        
        return {"age_vs_risk": age_vs_risk_data}

    except Exception as e:
        # Handle unexpected errors
        raise HTTPException(status_code=500, detail=f"An error occurred while processing the data: {str(e)}")
