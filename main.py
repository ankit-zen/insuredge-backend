
import logging
import random
from datetime import datetime
from typing import List, Dict, Union, Optional

import pandas as pd
from deltalake import DeltaTable
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pydantic import ConfigDict, Field

# Initialize FastAPI app
app = FastAPI()

model_config = ConfigDict(populate_by_name=True)

# Initialize logger
logging.basicConfig(level=logging.DEBUG)

# Path to the single Delta Lake table (created from dataset.csv)
delta_lake_base_path = "./my_delta_lake"
dataset_table_path = f"{delta_lake_base_path}/dataset"

# -------------------------------------------------------------------
# Helper to read the unified Delta table
# -------------------------------------------------------------------
def read_dataset_table() -> pd.DataFrame:
    try:
        logging.info(f"Reading unified Delta table from path: {dataset_table_path}")
        df = DeltaTable(dataset_table_path).to_pandas()
        # Normalize column names: ensure submissionDate field exists
        if 'date' in df.columns:
            df = df.rename(columns={'date': 'submissionDate'})
        return df
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

class RiskFactorDetail(BaseModel):
    name: str
    score: int
    risk: str
class AgeRisk(BaseModel):
    age: int
    #riskAssessment: Union[str, int]
    riskscore: str
    riskScore: int = Field(..., alias="riskScore")

# -------------------------------------------------------------------
# /applicationstats
# -------------------------------------------------------------------
@app.get("/applicationstats", response_model=ApplicationStats)
async def get_application_stats():
    df = read_dataset_table()
    total = len(df)
    pending = len(df[df['underwritingStatus'] == "Needs info"])
    approved = len(df[df['underwritingStatus'] == "Approved"])
    declined = len(df[df['underwritingStatus'] == "Declined"])
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
    df = read_dataset_table()

    # --- generate prefix-based assignedTo as fallback ---
    def make_prefix_assigned(row):
        reviewer = row.get('reviewer') or row.get('underwriterName') or ''
        # build initials for ID
        initials = ''.join(part[0].upper() for part in reviewer.split() if part) or 'NA'
        return {"id": initials, "name": reviewer}

    # Apply to every row to get a Series of dicts
    prefix_assigned_series = df.apply(make_prefix_assigned, axis=1)

    apps: List[Application] = []
    for idx, row in df.iterrows():
        # original assignedTo or default
        assigned = row.get('assignedTo', {"id": "N/A", "name": "Unknown"})
        # if still default, replace with prefix logic
        if assigned.get('id') == 'N/A' and assigned.get('name') == 'Unknown':
            assigned = prefix_assigned_series.iloc[idx]

        apps.append(Application(
            id=str(row.get('id', f"UW-{random.randint(100,999)}")),
            name=row.get('name', ''),
            notes=row.get('notes', ''),
            age=int(row.get('age', 0)),
            type=row.get('coverageType', ''),
            faceValue=str(row.get('coverageAmount', '')),
            riskScore=str(row.get('riskScore', '')),
            status=row.get('underwritingStatus', ''),
            submissionDate=str(row.get('submissionDate', '')),
            assignedTo=AssignedTo(**assigned)
        ))
    return apps


# -------------------------------------------------------------------
# /newapplications
# -------------------------------------------------------------------
@app.get("/newapplications", response_model=NewApplicationsResponse)
async def get_new_applications():
    df = read_dataset_table()
    df['date'] = pd.to_datetime(df['submissionDate'])
    now = datetime.now()
    current = df[df['date'].dt.month == now.month]
    previous = df[df['date'].dt.month == now.month - 1]
    curr_cnt = len(current)
    prev_cnt = len(previous)
    pct = ((curr_cnt - prev_cnt) / prev_cnt * 100) if prev_cnt else 0
    return NewApplicationsResponse(
        total_applications=curr_cnt,
        percentage_change=round(pct, 2),
        period=now.strftime("%b %Y")
    )

# -------------------------------------------------------------------
# /avgprocessingtime
# -------------------------------------------------------------------
# @app.get("/avgprocessingtime", response_model=AvgProcessingTimeResponse)
# async def get_avg_processing_time():
#     df = read_dataset_table()
#     df['submissionDate'] = pd.to_datetime(df['submissionDate'])
#     df['policyStartDate'] = pd.to_datetime(df['policyStartDate'])
#     df['processing_time'] = (df['policyStartDate'] - df['submissionDate']).dt.days
#     avg = df['processing_time'].mean()
#     return AvgProcessingTimeResponse(
#         avg_processing_time=round(avg, 2),
#         period=datetime.now().strftime("%b %Y")
#     )

@app.get("/avgprocessingtime", response_model=AvgProcessingTimeResponse)
async def get_avg_processing_time():
    df = read_dataset_table()
    df['submissionDate'] = pd.to_datetime(df['submissionDate'], errors='coerce')
    df['policyStartDate'] = pd.to_datetime(df['policyStartDate'], errors='coerce')

    # compute and filter
    df['processing_time'] = (df['policyStartDate'] - df['submissionDate']).dt.days
    df = df[df['processing_time'] >= 0]

    if df.empty:
        raise HTTPException(status_code=404, detail="No valid processing times to average")

    avg = df['processing_time'].mean()

    # derive period from the latest submission
    last = df['submissionDate'].max()
    period = last.strftime("%b %Y") if not pd.isna(last) else datetime.now().strftime("%b %Y")

    return AvgProcessingTimeResponse(
        avg_processing_time=round(avg, 2),
        period=period
    )


# -------------------------------------------------------------------
# /approvalrate
# -------------------------------------------------------------------
@app.get("/approvalrate", response_model=float)
async def get_approval_rate():
    df = read_dataset_table()
    approved = len(df[df['underwritingStatus'] == "Approved"])
    total = len(df)
    rate = (approved / total * 100) if total else 0
    return round(rate, 2)

# -------------------------------------------------------------------
# /underwritingdecisions
# -------------------------------------------------------------------

@app.get("/underwritingdecisions", response_model=List[UnderwritingDecision])
async def get_underwriting_decisions():
    """
    Fetches real underwriting decision data, broken out by month, 
    with raw counts for Approved, Declined, and Pending.
    """
    try:
        # Read unified dataset
        df = read_dataset_table()
        # Parse submission month
        df['date'] = pd.to_datetime(df['submissionDate'], errors='coerce')
        df['month'] = df['date'].dt.strftime('%b')
        
        # Group by month & status
        grouped = (
            df
            .dropna(subset=['month', 'underwritingStatus'])
            .groupby(['month', 'underwritingStatus'])
            .size()
            .reset_index(name='count')
        )

        # Sort months by calendar order
        order = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        months = sorted(
            grouped['month'].unique(),
            key=lambda m: order.index(m) if m in order else float('inf')
        )

        result: List[UnderwritingDecision] = []
        for m in months:
            month_df = grouped[grouped['month'] == m]
            data = {"month": m, "approved": 0, "declined": 0, "pending": 0}
            for _, row in month_df.iterrows():
                status = row['underwritingStatus']
                cnt = int(row['count'])
                if status == 'Approved':
                    data['approved'] = cnt
                elif status in ('Rejected','Declined'):
                    data['declined'] = cnt
                elif status == 'Pending':
                    data['pending'] = cnt
            result.append(UnderwritingDecision(**data))

        return result

    except Exception as e:
        logging.error("Error in /underwritingdecisions: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------------------------------------------------
# /policydistribution
# -------------------------------------------------------------------

@app.get("/policydistribution", response_model=List[PolicySegment])
async def get_policy_distribution():
    """
    Returns the distribution of policy/coverage types, along with color codes.
    """
    # 1) Load the unified dataset
    df = read_dataset_table()

    # 2) Count each coverageType
    counts = df['coverageType'].value_counts()

    # 3) Define your color mapping
    color_map = {
        "Term Life": "#0A1744",
        "Whole Life": "#1E40AF",
        "Universal Life": "#60A5FA",
        "Fixed Annuity": "#22C55E",
        "Variable Annuity": "#F97316"
    }

    # 4) Build the result list in descending order of count (optional)
    segments: List[PolicySegment] = []
    for coverage_type, cnt in counts.items():
        # only include types you have a color for (or you can default)
        color = color_map.get(coverage_type, "#999999")
        segments.append(
            PolicySegment(name=coverage_type, value=int(cnt), color=color)
        )

    # 5) (Optional) sort by value descending to match your UI
    segments.sort(key=lambda seg: seg.value, reverse=True)

    return segments



# -------------------------------------------------------------------
# /riskfactors
# -------------------------------------------------------------------


@app.get("/riskfactors", response_model=List[RiskFactorDetail])
async def get_risk_factors():
    df = read_dataset_table().dropna(subset=['riskFactors','riskScore'])
    # 1) compute mean raw score per factor
    grouped = (
        df
        .groupby('riskFactors')
        .agg(mean_raw=('riskScore','mean'))
        .reset_index()
    )

    # 2) min–max normalize
    lo, hi = grouped['mean_raw'].min(), grouped['mean_raw'].max()
    span = hi - lo or 1
    grouped['score'] = ((grouped['mean_raw'] - lo) / span * 100).round().astype(int)

    # 3) map back to Low/Med/High
    def to_label(s:int):
        if s < 33: return "Low Risk"
        if s < 66: return "Medium Risk"
        return "High Risk"

    result = []
    for _, row in grouped.iterrows():
        result.append(RiskFactorDetail(
            name=row['riskFactors'],
            score=row['score'],
            risk=to_label(row['score'])
        ))
    return result


# -------------------------------------------------------------------
# /agevsrisk
# -------------------------------------------------------------------


from typing import Dict, List, Optional
from fastapi import HTTPException
from pydantic import BaseModel
import pandas as pd

class AgeRisk(BaseModel):
    age: int
    risk: int

@app.get(
    "/agevsrisk",
    response_model=Dict[str, List[AgeRisk]]
)
async def get_age_vs_risk():
    df = read_dataset_table()

    # 1) validate columns exist
    required = ["coverageType", "age", "riskScore"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing required columns: {missing}")

    # 2) drop rows where those are null
    sub = df.dropna(subset=required)

    # 3) prepare our three buckets
    buckets: Dict[str, List[AgeRisk]] = {"termLife": [], "wholeLife": [], "annuity": []}

    # 4) map coverageType → bucket key
    def key_for(ct: str) -> Optional[str]:
        if ct == "Term Life":
            return "termLife"
        if ct == "Whole Life":
            return "wholeLife"
        if "Annuity" in ct:
            return "annuity"
        return None

    # 5) iterate and append
    for _, row in sub.iterrows():
        bucket = key_for(row["coverageType"])
        if not bucket:
            continue
        try:
            age   = int(row["age"])
            risk  = int(row["riskScore"])
        except (ValueError, TypeError):
            continue
        buckets[bucket].append(AgeRisk(age=age, risk=risk))

    # 6) fail if completely empty
    if not any(buckets.values()):
        raise HTTPException(status_code=404, detail="No age-vs-risk data found")

    return buckets
