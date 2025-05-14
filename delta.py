# # import os
# # import pandas as pd
# # from deltalake import write_deltalake, DeltaTable
# # from datetime import datetime


# # delta_lake_base_path = "./my_delta_lake"
# # os.makedirs(delta_lake_base_path, exist_ok=True)

# # def create_table_from_csv(csv_file_path, table_name):
# #     """
# #     Creates a Delta Lake table from a CSV file. The schema is dynamically inferred.
    
# #     Parameters:
# #     csv_file_path (str): Path to the CSV file
# #     table_name (str): Name of the table to create
    
# #     Returns:
# #     DeltaTable: The created Delta table
# #     """

# #     print(f"Reading data from {csv_file_path}...")
# #     df = pd.read_csv(csv_file_path)
    

# #     delta_table_path = f"{delta_lake_base_path}/{table_name}"


# #     try:
# #         write_deltalake(delta_table_path, df)
# #         print(f"Created/Updated Delta table: {table_name} with {len(df)} records")
# #     except Exception as e:
# #         print(f"Error writing to Delta table: {e}")
    

# #     return DeltaTable(delta_table_path)


# # def list_delta_tables():
# #     print("\nTables in your Delta Lake:")
# #     for table in os.listdir(delta_lake_base_path):
# #         table_path = f"{delta_lake_base_path}/{table}"
# #         if os.path.isdir(table_path):
# #             dt = DeltaTable(table_path)
# #             print(f"- {table} (version: {dt.version()}, records: {len(dt.to_pandas())})")


# # def query_delta_tables():

# #     for table in os.listdir(delta_lake_base_path):
# #         table_path = f"{delta_lake_base_path}/{table}"
# #         if os.path.isdir(table_path):
# #             dt = DeltaTable(table_path)
# #             df = dt.to_pandas()
# #             print(f"\nSample from table {table}:")
# #             print(df.head())

# # #
# # def process_csv_to_deltalake(csv_file_path):
# #     table_name = os.path.basename(csv_file_path).split(".")[0] 
# #     try:
        
# #         delta_table = create_table_from_csv(csv_file_path, table_name)
# #     except FileNotFoundError:
# #         print(f"CSV file {csv_file_path} not found.")
    
# #     #list and query the tables after the upload
# #     list_delta_tables()
# #     query_delta_tables()

# # csv_file_path = "insurance_data.csv" 
# # process_csv_to_deltalake(csv_file_path)


# import os
# import pandas as pd
# from deltalake import write_deltalake, DeltaTable

# # Define the Delta Lake base path
# delta_lake_base_path = "./my_delta_lake"
# os.makedirs(delta_lake_base_path, exist_ok=True)

# def create_table_from_csv(csv_file_path, table_name):
#     """
#     Creates a Delta Lake table from a CSV file. The schema is dynamically inferred.
    
#     Parameters:
#     csv_file_path (str): Path to the CSV file
#     table_name (str): Name of the table to create
    
#     Returns:
#     DeltaTable: The created Delta table
#     """
#     # Read CSV data into a Pandas DataFrame
#     print(f"Reading data from {csv_file_path}...")
#     df = pd.read_csv(csv_file_path)
    
#     # Define the path for the table in Delta Lake
#     delta_table_path = f"{delta_lake_base_path}/{table_name}"

#     try:
#         # Check if the table already exists
#         if os.path.exists(delta_table_path):
#             print(f"Table {table_name} already exists. Deleting and recreating...")
#             # Optionally, delete the existing table (careful with this in production)
#             # shutil.rmtree(delta_table_path)  # Uncomment to delete
#         else:
#             print(f"Creating new table: {table_name}")
        
#         # Write DataFrame to Delta Lake (overwrite if the table exists)
#         write_deltalake(delta_table_path, df)
#         print(f"Created/Updated Delta table: {table_name} with {len(df)} records")
#     except Exception as e:
#         print(f"Error writing to Delta table: {e}")
    
#     # Return the Delta table object
#     return DeltaTable(delta_table_path)

# def list_delta_tables():
#     """
#     Lists all the Delta tables present in the Delta Lake base path.
#     """
#     print("\nTables in your Delta Lake:")
#     for table in os.listdir(delta_lake_base_path):
#         table_path = f"{delta_lake_base_path}/{table}"
#         if os.path.isdir(table_path):
#             try:
#                 dt = DeltaTable(table_path)
#                 print(f"- {table} (version: {dt.version()}, records: {len(dt.to_pandas())})")
#             except Exception as e:
#                 print(f"Error reading table {table}: {e}")

# def query_delta_tables():
#     """
#     Queries all the Delta tables and prints the first few records for each.
#     """
#     for table in os.listdir(delta_lake_base_path):
#         table_path = f"{delta_lake_base_path}/{table}"
#         if os.path.isdir(table_path):
#             try:
#                 dt = DeltaTable(table_path)
#                 df = dt.to_pandas()
#                 print(f"\nSample from table {table}:")
#                 print(df.head())
#             except Exception as e:
#                 print(f"Error querying table {table}: {e}")

# def process_csv_to_deltalake(csv_file_path):
#     """
#     Processes a CSV file and loads it into Delta Lake.
    
#     Parameters:
#     csv_file_path (str): Path to the CSV file
#     """
#     # Dynamically generate table name from the CSV filename
#     table_name = os.path.basename(csv_file_path).split(".")[0]  # Remove the extension
#     try:
#         # Create or update the Delta table
#         delta_table = create_table_from_csv(csv_file_path, table_name)
#     except FileNotFoundError:
#         print(f"CSV file {csv_file_path} not found.")
    
#     # List and query the Delta tables after the upload
#     list_delta_tables()
#     query_delta_tables()

# # Example usage for underwriting and application data
# underwriting_csv_path = "data\\underwriting.csv"  
# applications_csv_path = "data\\applications.csv"  

# # Process both underwriting and application data and add to Delta Lake
# process_csv_to_deltalake(underwriting_csv_path)
# process_csv_to_deltalake(applications_csv_path)




import os
import pandas as pd
from deltalake import write_deltalake, DeltaTable
import shutil

# Define the Delta Lake base path
delta_lake_base_path = "./my_delta_lake"
os.makedirs(delta_lake_base_path, exist_ok=True)

def create_table_from_csv(csv_file_path, table_name):
    """
    Creates a Delta Lake table from a CSV file. The schema is dynamically inferred.
    
    Parameters:
    csv_file_path (str): Path to the CSV file
    table_name (str): Name of the table to create
    
    Returns:
    DeltaTable: The created Delta table
    """
    print(f"Reading data from {csv_file_path}...")
    df = pd.read_csv(csv_file_path)
    
    delta_table_path = f"{delta_lake_base_path}/{table_name}"
    try:
        if os.path.exists(delta_table_path):
              print(f"Table {table_name} already exists. Deleting for clean overwrite…")
              shutil.rmtree(delta_table_path)

        print(f"Writing fresh Delta table: {table_name}")
        write_deltalake(delta_table_path, df)     # same call as initial create
        print(f"Created Delta table `{table_name}` with {len(df)} rows")
    except Exception as e:
        print(f"Error writing Delta table: {e}")
        raise
    
    return DeltaTable(delta_table_path)

def list_delta_tables():
    """
    Lists all the Delta tables present in the Delta Lake base path.
    """
    print("\nTables in your Delta Lake:")
    for table in os.listdir(delta_lake_base_path):
        table_path = f"{delta_lake_base_path}/{table}"
        if os.path.isdir(table_path):
            try:
                dt = DeltaTable(table_path)
                print(f"- {table} (version: {dt.version()}, records: {len(dt.to_pandas())})")
            except Exception as e:
                print(f"Error reading table {table}: {e}")

def query_delta_tables():
    """
    Queries all the Delta tables and prints the first few records for each.
    """
    for table in os.listdir(delta_lake_base_path):
        table_path = f"{delta_lake_base_path}/{table}"
        if os.path.isdir(table_path):
            try:
                dt = DeltaTable(table_path)
                df = dt.to_pandas()
                print(f"\nSample from table {table}:")
                print(df.head())
            except Exception as e:
                print(f"Error querying table {table}: {e}")

def process_csv_to_deltalake(csv_file_path):
    """
    Processes a CSV file and loads it into Delta Lake.
    
    Parameters:
    csv_file_path (str): Path to the CSV file
    """
    table_name = os.path.splitext(os.path.basename(csv_file_path))[0]
    try:
        create_table_from_csv(csv_file_path, table_name)
    except FileNotFoundError:
        print(f"CSV file {csv_file_path} not found.")
    
    list_delta_tables()
    query_delta_tables()

if __name__ == "__main__":
    # === point this to your single CSV file ===
    dataset_csv = "data/dataset.csv"
    process_csv_to_deltalake(dataset_csv)
