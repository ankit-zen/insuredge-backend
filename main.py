import logging
import random
from datetime import datetime
from typing import List, Dict, Union

import pandas as pd
from deltalake import DeltaTable
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI()

# Initialize logger
logging.basicConfig(level=logging.DEBUG)

# Path to the Delta Lake tables
delta_lake_base_path = "./my_delta_lake"
applications_table_path = f"{delta_lake_base_path}/applications"
underwriting_table_path = f"{delta_lake_base_path}/underwriting"


# -------------------------------------------------------------------
# Helper to read Delta tables
# -------------------------------------------------------------------
def read_delta_table(table_path: str) -> pd.DataFrame:
    try:
        logging.info(f"Reading Delta table from path: {table_path}")
        return DeltaTable(table_path).to_pandas()
    except Exception as e:
        logging.error(f"Error reading Delta table: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------------------------------------------------
# Pydantic models
# -------------------------------------------------------------------
class ApplicationDetails(BaseModel):
    age: int
    occupation: str
    coverageType: str
    coverageAmount: str

class AssignedTo(BaseModel):
    id: str
    name: str

class Application(BaseModel):
    id: str
    name: str
    notes: str
    age: int
    type: str
    faceValue: str
    riskScore: str
    status: str
    submissionDate: str
    assignedTo: AssignedTo

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

class NewApplicationsResponse(BaseModel):
    total_applications: int
    percentage_change: float
    period: str

class AvgProcessingTimeResponse(BaseModel):
    avg_processing_time: float
    period: str

class UnderwritingDecision(BaseModel):
    month: str
    approved: int
    declined: int
    pending: int

class CoverageCount(BaseModel):
    name: str
    count: int

class PolicySegment(BaseModel):
    name: str
    value: int
    color: str

class RiskFactorCount(BaseModel):
    name: str
    count: int

class AgeRisk(BaseModel):
    age: int
    riskAssessment: Union[str,int]


# -------------------------------------------------------------------
# /applicationstats
# -------------------------------------------------------------------
@app.get("/applicationstats", response_model=ApplicationStats)
async def get_application_stats():
    df = read_delta_table(applications_table_path)
    total = len(df)
    pending = len(df[df['status'] == "Needs info"])
    approved = len(df[df['status'] == "Approved"])
    declined = len(df[df['status'] == "Declined"])
    return ApplicationStats(
        totalReviewed=total,
        pendingUnderwriting=pending,
        approved=approved,
        declined=declined
    )


# -------------------------------------------------------------------
# /riskdistribution
# -------------------------------------------------------------------
@app.get("/riskdistribution", response_model=List[RiskSegment])
async def get_risk_distribution():
    segments = [
        {"name": "High", "value": 35, "color": "#ef4444"},
        {"name": "Medium", "value": 40, "color": "#f97316"},
        {"name": "Low", "value": 25, "color": "#22c55e"}
    ]
    return [RiskSegment(**s) for s in segments]


# -------------------------------------------------------------------
# /applications
# -------------------------------------------------------------------
@app.get("/applications", response_model=List[Application])
async def get_applications():
    required = ['name','riskScore','status','submissiondate',
                'facevalue','riskfactors','type','assignedTo','age','occupation','notes']
    df = read_delta_table(applications_table_path)
    apps: List[Application] = []
    for _, row in df.iterrows():
        miss = [c for c in required if c not in row]
        if miss:
            logging.warning(f"Skipping row, missing {miss}")
            continue
        assigned = row.get('assignedTo', {"id":"N/A","name":"Unknown"})
        apps.append(Application(
            id=f"UW-{random.randint(100,999)}",
            name=row['name'],
            notes=row.get('notes',''),
            age=int(row['age']),
            type=row['type'],
            faceValue=row['facevalue'],
            riskScore=row['riskScore'],
            status=row['status'],
            submissionDate=row['submissiondate'],
            assignedTo=AssignedTo(**assigned)
        ))
    return apps


# -------------------------------------------------------------------
# /newapplications
# -------------------------------------------------------------------
@app.get("/newapplications", response_model=NewApplicationsResponse)
async def get_new_applications():
    df = read_delta_table(applications_table_path)
    df['date'] = pd.to_datetime(df['date'])
    now = datetime.now()
    current = df[df['date'].dt.month == now.month]
    previous = df[df['date'].dt.month == now.month - 1]
    curr_cnt = len(current)
    prev_cnt = len(previous)
    pct = ((curr_cnt - prev_cnt) / prev_cnt * 100) if prev_cnt else 0
    return NewApplicationsResponse(
        total_applications=curr_cnt,
        percentage_change=round(pct,2),
        period=now.strftime("%b %Y")
    )


# -------------------------------------------------------------------
# /avgprocessingtime
# -------------------------------------------------------------------
@app.get("/avgprocessingtime", response_model=AvgProcessingTimeResponse)
async def get_avg_processing_time():
    df = read_delta_table(applications_table_path)
    df['date'] = pd.to_datetime(df['date'])
    df['policyStartDate'] = pd.to_datetime(df['policyStartDate'])
    df['processing_time'] = (df['policyStartDate'] - df['date']).dt.days
    avg = df['processing_time'].mean()
    return AvgProcessingTimeResponse(
        avg_processing_time=round(avg,2),
        period=datetime.now().strftime("%b %Y")
    )


# -------------------------------------------------------------------
# /approvalrate
# -------------------------------------------------------------------
@app.get("/approvalrate", response_model=float)
async def get_approval_rate():
    df = read_delta_table(applications_table_path)
    approved = len(df[df['status']=="Approved"])
    total = len(df)
    rate = (approved/total*100) if total else 0
    return round(rate,2)


# -------------------------------------------------------------------
# /underwritingdecisions
# -------------------------------------------------------------------
@app.get("/underwritingdecisions", response_model=List[UnderwritingDecision])
async def get_underwriting_decisions():
    try:
        du = read_delta_table(underwriting_table_path)
        da = read_delta_table(applications_table_path)[['name','date']]
        merged = pd.merge(du, da, on='name', how='left')
        merged['date'] = pd.to_datetime(merged['date'], errors='coerce')
        merged['month'] = merged['date'].dt.strftime('%b')
        grouped = merged.groupby(['month','underwritingStatus']).size().reset_index(name='count')

        order = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        months = sorted(grouped['month'].unique(), key=lambda m: order.index(m) if m in order else -1)

        result: List[UnderwritingDecision] = []
        for m in months:
            slice_ = grouped[grouped['month']==m]
            data = {"month":m,"approved":0,"declined":0,"pending":0}
            for _,r in slice_.iterrows():
                s,c = r['underwritingStatus'], r['count']
                if s=='Approved':        data['approved']=c
                elif s in ('Rejected','Declined'): data['declined']=c
                elif s=='Pending':       data['pending']=c
            # scale
            for k in ('approved','declined','pending'):
                data[k] = round(data[k]/70)
            result.append(UnderwritingDecision(**data))

        return result

    except Exception as e:
        logging.error(e)
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------------------------------------------------
# /coverageTypeDistribution
# -------------------------------------------------------------------
@app.get("/coverageTypeDistribution", response_model=List[CoverageCount])
async def get_coverage_type_distribution():
    df = read_delta_table(applications_table_path)
    counts = df['coverageType'].value_counts().to_dict()
    return [CoverageCount(name=k,count=v) for k,v in counts.items()]


# -------------------------------------------------------------------
# /policydistribution
# -------------------------------------------------------------------
@app.get("/policydistribution", response_model=List[PolicySegment])
async def get_policy_distribution():
    df = read_delta_table(applications_table_path)
    counts = df['coverageType'].value_counts()
    color_map = {
        "Term Life": "#0A1744",
        "Whole Life": "#1E40AF",
        "Universal Life": "#60A5FA",
        "Fixed Annuity": "#22C55E",
        "Variable Annuity": "#F97316"
    }
    segments: List[PolicySegment] = []
    for k,v in counts.items():
        if k in color_map:
            segments.append(PolicySegment(name=k, value=int(v), color=color_map[k]))
    return segments


# -------------------------------------------------------------------
# /riskfactors
# -------------------------------------------------------------------
@app.get("/riskfactors", response_model=List[RiskFactorCount])
async def get_risk_factors():
    df = read_delta_table(underwriting_table_path)
    counts = df['riskAssessment'].value_counts().to_dict()
    return [RiskFactorCount(name=str(k), count=int(v)) for k,v in counts.items()]


# -------------------------------------------------------------------
# /agevsrisk
# -------------------------------------------------------------------
@app.get("/agevsrisk", response_model=List[AgeRisk])
async def get_age_vs_risk():
    apps = read_delta_table(applications_table_path)[['name','age']].dropna()
    under = read_delta_table(underwriting_table_path)[['name','riskAssessment']].dropna()
    data: List[AgeRisk] = []
    for _,r in apps.iterrows():
        match = under[under['name']==r['name']]
        if not match.empty:
            data.append(AgeRisk(age=int(r['age']), riskAssessment=match.iloc[0]['riskAssessment']))
    if not data:
        raise HTTPException(status_code=404, detail="No matching age vs risk data")
    return data
