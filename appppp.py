import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from deltalake import DeltaTable
import pandas as pd
from typing import List, Union
from datetime import datetime
from collections import defaultdict
import random

# Initialize FastAPI app
app = FastAPI()

# Initialize logger
logging.basicConfig(level=logging.DEBUG)

# Path to the Delta Lake tables
delta_lake_base_path = "./my_delta_lake"
applications_table_path = f"{delta_lake_base_path}/applications"
underwriting_table_path = f"{delta_lake_base_path}/underwriting"

# Helper function to read Delta Table into pandas DataFrame
def read_delta_table(table_path: str):
    try:
        logging.info(f"Reading Delta table from path: {table_path}")
        delta_table = DeltaTable(table_path)
        # Load the table data into pandas DataFrame
        return delta_table.to_pandas()
    except Exception as e:
        logging.error(f"Error reading Delta table: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error reading Delta table: {str(e)}")

# Pydantic Models to validate and format API responses
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

class AvgProcessingTimeResponse(BaseModel):
    avg_processing_time: float
    period: str

class NewApplicationsResponse(BaseModel):
    total_applications: int
    percentage_change: float
    period: str

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

# # API Endpoint to fetch all application details
# @app.get("/applications", response_model=List[Application])
# async def get_applications():
#     """
#     Fetches all application details, including name, risk score, status, and additional details.
#     """
#     df = read_delta_table(applications_table_path)
#     applications = []
    
#     for _, row in df.iterrows():
#         details = ApplicationDetails(
#             age=row['age'],
#             occupation=row['occupation'],
#             coverageType=row['coverageType'],
#             coverageAmount=row['coverageAmount']
#         )
#         application = Application(
#             name=row['name'],
#             riskScore=row['riskScore'],
#             status=row['status'],
#             reviewer=row['reviewer'],
#             date=row['date'],
#             details=details
#         )
#         applications.append(application)
    
#     return applications

@app.get("/applications", response_model=List[Application])
async def get_applications():
    """
    Fetches all application details, including name, risk score, status, and additional details.
    """
    required_columns = ['name', 'riskScore', 'status', 'submissiondate', 'facevalue', 'riskfactors', 'type', 'assignedTo', 'age', 'occupation']
    
    try:
        df = read_delta_table(applications_table_path)
        logging.debug(f"Data loaded from Delta table: {df.shape[0]} rows")
    except Exception as e:
        logging.error(f"Error loading Delta table: {e}")
        raise HTTPException(status_code=500, detail="Error reading data from Delta table.")
    
    applications = []
    logging.debug(f"Starting to process the rows in the data")

    for _, row in df.iterrows():
        logging.debug(f"Processing row: {row}")
        missing_columns = [col for col in required_columns if col not in row]
        
        if missing_columns:
            logging.warning(f"Missing required columns: {missing_columns} in row {row}")
            continue

        try:
            # Default values for missing columns if any
            reviewer = row['reviewer'] if 'reviewer' in row else 'Unknown'
            assigned_to = row['assignedTo'] if 'assignedTo' in row else {"id": "N/A", "name": "Unknown"}
            
            application = Application(
                id=f"UW-{random.randint(100, 999)}",  # Add unique ID
                name=row['name'],
                riskScore=row['riskScore'],
                status=row['status'],
                reviewer=reviewer,
                date=row['submissiondate'],
                facevalue=row['facevalue'],
                riskfactors=row['riskfactors'],
                type=row['type'],
                assignedTo=assigned_to,
                age=row['age'],
                occupation=row['occupation'],
            )
            applications.append(application)
            logging.debug(f"Added application: {application}")
        except KeyError as e:
            logging.error(f"Error processing row {row}: Missing column {str(e)}")
            raise HTTPException(status_code=400, detail=f"Error processing data: Missing column {str(e)}")

    if not applications:
        logging.warning("No applications were processed or added to the list.")
        
    return applications


# API Endpoint to fetch the new applications and percentage change vs the previous period
# @app.get("/newapplications")
# async def get_new_applications():
#     """
#     Fetches the total number of new applications in the current period and the percentage change compared to the previous period.
#     """
#     df = read_delta_table(applications_table_path)
#     df['date'] = pd.to_datetime(df['date'])
    
#     # Filter applications for current and previous months
#     current_period = df[df['date'].dt.month == datetime.now().month]
#     previous_period = df[df['date'].dt.month == datetime.now().month - 1]
    
#     current_count = len(current_period)
#     previous_count = len(previous_period)
    
#     # Calculate percentage change
#     percentage_change = ((current_count - previous_count) / previous_count) * 100 if previous_count > 0 else 0
    
#     return {"total_applications": current_count, "percentage_change": round(percentage_change, 2), "period": "May 2025"}

# @app.get("/newapplications")
# async def get_new_applications():
#     """
#     Fetches the total number of new applications in the current period and the percentage change compared to the previous period.
#     """
#     df = read_delta_table(applications_table_path)
#     df['date'] = pd.to_datetime(df['date'])
    
#     # Filter applications for current and previous months
#     current_period = df[df['date'].dt.month == datetime.now().month]
#     previous_period = df[df['date'].dt.month == datetime.now().month - 1]
    
#     current_count = len(current_period)
#     previous_count = len(previous_period)
    
#     # Calculate percentage change
#     percentage_change = ((current_count - previous_count) / previous_count) * 100 if previous_count > 0 else 0
    
#     return {"total_applications": current_count, "percentage_change": round(percentage_change, 2), "period": "May 2025"}

@app.get(
    "/newapplications",
    response_model=NewApplicationsResponse
)
async def get_new_applications():
    """
    Fetches the total number of new applications in the current period
    and the percentage change compared to the previous period.
    """
    df = read_delta_table(applications_table_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # Filter applications for current and previous months
    current_period = df[df['date'].dt.month == datetime.now().month]
    previous_period = df[df['date'].dt.month == datetime.now().month - 1]
    
    current_count = len(current_period)
    previous_count = len(previous_period)
    
    # Calculate percentage change
    percentage_change = (
        (current_count - previous_count) / previous_count * 100
        if previous_count > 0 else 0
    )
    
    return NewApplicationsResponse(
        total_applications=current_count,
        percentage_change=round(percentage_change, 2),
        period="May 2025"
    )

# API Endpoint to fetch the average processing time of applications
# @app.get("/avgprocessingtime")
# async def get_avg_processing_time():
#     """
#     Fetches the average processing time for applications, which is calculated as the difference between submission and policy start date.
#     """
#     df = read_delta_table(applications_table_path)
#     df['date'] = pd.to_datetime(df['date'])
#     df['policyStartDate'] = pd.to_datetime(df['policyStartDate'])
#     df['processing_time'] = (df['policyStartDate'] - df['date']).dt.days
    
#     # Calculate the average processing time
#     avg_processing_time = df['processing_time'].mean()
    
#     return {"avg_processing_time": round(avg_processing_time, 2), "period": "May 2025"}


@app.get(
    "/avgprocessingtime",
    response_model=AvgProcessingTimeResponse
)
async def get_avg_processing_time():
    """
    Fetches the average processing time for applications, 
    calculated as the difference between submission and policy start date.
    """
    df = read_delta_table(applications_table_path)
    df['date'] = pd.to_datetime(df['date'])
    df['policyStartDate'] = pd.to_datetime(df['policyStartDate'])
    df['processing_time'] = (df['policyStartDate'] - df['date']).dt.days

    avg_processing_time = df['processing_time'].mean()

    return AvgProcessingTimeResponse(
        avg_processing_time=round(avg_processing_time, 2),
        period="May 2025"
    )


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
    
    return round(approval_rate, 2)

# # API Endpoint to fetch underwriting decisions (Approved, Rejected, Pending)
# @app.get("/underwritingdecisions")
# async def get_underwriting_decisions():
#     """
#     Fetches the breakdown of underwriting decisions (Approved, Rejected, Pending).
#     """
#     df = read_delta_table(underwriting_table_path)
    
#     # Count underwriting decisions by status
#     underwriting_decisions = df['underwritingStatus'].value_counts()
    
#     return {"underwriting_decisions": underwriting_decisions.to_dict(), "period": "May 2025"}

@app.get("/underwritingdecisions")
async def get_underwriting_decisions():
    """
    Fetches real underwriting decision data but scales it down to more manageable numbers.
    """
    try:
        # Flag to control whether to use real data or test data
        use_real_data = True
        
        if use_real_data:
            # Read data from Delta tables
            df_underwriting = read_delta_table(underwriting_table_path)
            df_applications = read_delta_table(applications_table_path)
            
            # Merge the tables
            merged_df = pd.merge(df_underwriting, df_applications[['name', 'date']], on='name', how='left')
            merged_df['date'] = pd.to_datetime(merged_df['date'], errors='coerce')
            merged_df['month'] = merged_df['date'].dt.strftime('%b')
            
            # Group by month and status
            monthly_status = merged_df.groupby(['month', 'underwritingStatus']).size().reset_index(name='count')
            
            # Process the data
            result = []
            months = sorted(monthly_status['month'].unique(), 
                          key=lambda x: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'].index(x) 
                          if x in ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'] 
                          else 0)
            
            for month in months:
                month_data = {"month": month, "approved": 0, "declined": 0, "pending": 0}
                month_statuses = monthly_status[monthly_status['month'] == month]
                
                for _, row in month_statuses.iterrows():
                    status = row['underwritingStatus']
                    count = row['count']
                    
                    if status == 'Approved':
                        month_data['approved'] = count
                    elif status in ['Rejected', 'Declined']:
                        month_data['declined'] = count
                    elif status == 'Pending':
                        month_data['pending'] = count
                
                # Scale down the numbers to be more manageable (divide by 60-100)
                scaling_factor = 70  # Adjust as needed to get numbers in the 10-100 range
                month_data['approved'] = round(month_data['approved'] / scaling_factor)
                month_data['declined'] = round(month_data['declined'] / scaling_factor)
                month_data['pending'] = round(month_data['pending'] / scaling_factor)
                
                result.append(month_data)
            
            # Get the latest month
            latest_month = merged_df['date'].max().strftime("%b %Y") if not merged_df.empty else "May 2025"
            
        else:
            # Use the test data that exactly matches the expected format
            result = [
                {"month": "Jan", "approved": 65, "declined": 12, "pending": 23},
                {"month": "Feb", "approved": 75, "declined": 15, "pending": 18},
                {"month": "Mar", "approved": 70, "declined": 10, "pending": 20},
                {"month": "Apr", "approved": 85, "declined": 8, "pending": 15},
                {"month": "May", "approved": 80, "declined": 12, "pending": 17}
            ]
            latest_month = "May 2025"
        
        return result
    
    except Exception as e:
        logging.error(f"Error processing underwriting decisions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing underwriting decisions: {str(e)}")

# # API Endpoint to fetch policy distribution (e.g., Term Life, Whole Life)
# @app.get("/policydistribution")
# async def get_policy_distribution():
#     """
#     Fetches the distribution of different policy types (e.g., Term Life, Whole Life).
#     """
#     df = read_delta_table(applications_table_path)
    
#     # Count policy types distribution
#     policy_distribution = df['policyType'].value_counts()
    
#     return {"policy_distribution": policy_distribution.to_dict()}

@app.get("/coverageTypeDistribution")
async def get_coverage_type_distribution():
    """
    Fetches the distribution of different coverage types with their counts.
    """
    try:
        # Read data from the applications Delta table
        df = read_delta_table(applications_table_path)

        # Count the occurrences of each coverage type
        coverage_counts = df['coverageType'].value_counts()

        # Log or return the coverage distribution
        return coverage_counts.to_dict()

    except Exception as e:
        logging.error(f"Error processing coverage type distribution: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing coverage type distribution: {str(e)}")


@app.get("/policydistribution")
async def get_policy_distribution():
    """
    Returns the distribution of different policy types, along with color codes for each.
    """
    try:
        # Read data from the applications Delta table
        df = read_delta_table(applications_table_path)
        
        # Count occurrences of each coverage type
        coverage_counts = df['coverageType'].value_counts()

        # Define color mapping for the policy types
        color_mapping = {
            "Term Life": "#0A1744",
            "Whole Life": "#1E40AF",
            "Universal Life": "#60A5FA",
            "Fixed Annuity": "#22C55E",
            "Variable Annuity": "#F97316"
        }
        
        # Prepare the result in the expected format
        policy_distribution = []
        for coverage_type, count in coverage_counts.items():
            # If the coverage type exists in the color mapping, add it to the result
            if coverage_type in color_mapping:
                policy_distribution.append({
                    "name": coverage_type,
                    "value": count,
                    "color": color_mapping[coverage_type]
                })

        # Return the response
        return policy_distribution

    except Exception as e:
        logging.error(f"Error fetching policy distribution: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching policy distribution: {str(e)}")


# API Endpoint to fetch risk factors (Risk Assessment)
@app.get("/riskfactors")
async def get_risk_factors():
    """
    Fetches the distribution of risk assessments (High, Medium, Low).
    """
    df = read_delta_table(underwriting_table_path)
    
    # Count risk assessments
    risk_factors = df['riskAssessment'].value_counts()
    
    return risk_factors.to_dict()

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
        
        return age_vs_risk_data

    except Exception as e:
        # Handle unexpected errors
        raise HTTPException(status_code=500, detail=f"An error occurred while processing the data: {str(e)}")
