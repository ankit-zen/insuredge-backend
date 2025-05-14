
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
@app.get("/avgprocessingtime", response_model=AvgProcessingTimeResponse)
async def get_avg_processing_time():
    df = read_dataset_table()
    df['submissionDate'] = pd.to_datetime(df['submissionDate'])
    df['policyStartDate'] = pd.to_datetime(df['policyStartDate'])
    df['processing_time'] = (df['policyStartDate'] - df['submissionDate']).dt.days
    avg = df['processing_time'].mean()
    return AvgProcessingTimeResponse(
        avg_processing_time=round(avg, 2),
        period=datetime.now().strftime("%b %Y")
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
    try:
        df = read_dataset_table()
        df['date'] = pd.to_datetime(df['submissionDate'], errors='coerce')
        df['month'] = df['date'].dt.strftime('%b')
        grouped = df.groupby(['month', 'underwritingStatus']).size().reset_index(name='count')

        order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        months = sorted(grouped['month'].unique(), key=lambda m: order.index(m) if m in order else -1)

        result: List[UnderwritingDecision] = []
        for m in months:
            slice_ = grouped[grouped['month'] == m]
            data = {"month": m, "approved": 0, "declined": 0, "pending": 0}
            for _, r in slice_.iterrows():
                status, count = r['underwritingStatus'], r['count']
                if status == 'Approved':
                    data['approved'] = count
                elif status in ('Rejected', 'Declined'):
                    data['declined'] = count
                elif status == 'Pending':
                    data['pending'] = count
            for k in ('approved', 'declined', 'pending'):
                data[k] = round(data[k] / 70)
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
    df = read_dataset_table()
    counts = df['coverageType'].value_counts().to_dict()
    return [CoverageCount(name=k, count=v) for k, v in counts.items()]

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
    """
    Aggregates risk factors across all applications,
    returning each factor with occurrence count and predominant risk level.
    """
    df = read_dataset_table()
    required = ['riskFactors', 'riskScore', 'riskAssessment']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing required columns: {missing}")

    # Drop rows missing any required fields
    sub = df.dropna(subset=required)
    # Group by factor: count occurrences, most common assessment
    grouped = sub.groupby('riskFactors').agg(
        count=('riskFactors', 'size'),
        mode_risk=('riskAssessment', lambda x: x.mode().iloc[0] if not x.mode().empty else '')
    ).reset_index()

    result: List[RiskFactorDetail] = []
    for _, row in grouped.iterrows():
        name = row['riskFactors']
        score = int(row['count'])
        risk = f"{row['mode_risk'].title()} Risk"
        result.append(RiskFactorDetail(name=name, score=score, risk=risk))
    return result

# -------------------------------------------------------------------
# /agevsrisk
# -------------------------------------------------------------------
# Allow passing riskScore by its JSON name
 
@app.get(
    "/agevsrisk",
    response_model=Dict[str, List[AgeRisk]]
)
async def get_age_vs_risk():
    """
    Returns a dict of age vs numeric riskScore points, grouped by policy type:
    {
      "termLife": [...],
      "wholeLife": [...],
      "annuity": [...]
    }
    """
    df = read_dataset_table()

    # Ensure we have the right columns
    required = ["coverageType", "age", "riskScore"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing required columns: {missing}")

    sub = df.dropna(subset=required)

    buckets: Dict[str, List[AgeRisk]] = {
        "termLife": [],
        "wholeLife": [],
        "annuity": []
    }

    def key_for(ct: str) -> Optional[str]:
        if ct == "Term Life":
            return "termLife"
        if ct == "Whole Life":
            return "wholeLife"
        if "Annuity" in ct:
            return "annuity"
        return None

    for _, row in sub.iterrows():
        bucket = key_for(row["coverageType"])
        if not bucket:
            continue

        try:
            age = int(row["age"])
            score = int(row["riskScore"])
        except (ValueError, TypeError):
            continue

        buckets[bucket].append(AgeRisk(age=age, riskScore=score))

    if not any(buckets.values()):
        raise HTTPException(status_code=404, detail="No age-vs-risk data found")

    return buckets