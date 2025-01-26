#Libraries
import os
from contextlib import asynccontextmanager

import pandas as pd
from PIL import Image
import cv2
import segmentation_models_pytorch as smp
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse
from pydantic import BaseModel

#matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
import seaborn as sns
import torch
import shutil
from segmentation import run_segment
from CFG import CFG
from dotenv import load_dotenv
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import mysql.connector
from mysql.connector import pooling

# Declare here such that we don't need to
model = None
pool = None


#Load the model "once"
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv()
    global model
    global pool

    # Load the model and set to eval mode
    model = build_model(CFG.backbone, CFG.num_classes, CFG.device)
    model.load_state_dict(torch.load("model_weight.pth", map_location=CFG.device))
    model.eval()

    print("Model successfully loaded")

    # Load the database pool
    pool = pooling.MySQLConnectionPool(
        pool_name="mypool",
        pool_size=5,  # Number of connections in the pool
        pool_reset_session=True,
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_DATABASE"),
        port=3306
    )

    print("Database successfully loaded")

    # Yield control to indicate successful startup
    yield

    # Cleanup logic (if needed) goes here
    print("Application is shutting down")

def get_cursor():

    if pool is not None:

        connection = pool.get_connection()
        cursor = connection.cursor()
        return cursor, connection

    return "Database failed to initialize"

#Start the app
app = FastAPI(lifespan=lifespan)


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your frontend's URL
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)



UPLOAD_DIR = "uploaded_images"
GEN_DIR = "generated_images"



# Define the model structure
def build_model(backbone, num_classes, device):

    model_builder = smp.Unet(
        encoder_name=backbone,
        encoder_weights=None,  # Pretrained weights already used during training
        in_channels=3,
        classes=num_classes,
        activation=None,
    )
    model_builder.to(device)
    return model_builder


class AddData(BaseModel):
    user_id: str
    acne_cells: int = None
    acne_coverage: float = None
    date: str = None
    filename: str = None

class AddUser(BaseModel):
    user_id: str
    name: str
    email: str



# Add a new user
@app.post("/update-user/")
async def update_user(request: AddUser):

    response = add_user(request.user_id, request.name, request.email)

    return response


# Update a database
@app.post("/update-database/")
async def update_database(request: AddData):

    user_id = request.user_id

    #Authenticate the user
    if authenticate_user(user_id) is True:
        add_data(request.acne_cells, request.acne_coverage, request.date, user_id, request.filename)
        return {"response" : str("Successfully added data!")}

    # Otherwise, don't add the data
    else:

        return {"response" : str("Could not authenticate user!")}


# To retrieve user data
@app.get("/get-user-data/{user_id}/{limit}")
async def get_user_data(user_id: str, limit: int):
    cursor, connection = get_cursor()

    try:

        results = get_data(user_id, limit)

        if not results:
            raise HTTPException(status_code=404, detail="No data found for the given user_id.")

        # FastAPI will automatically convert the list of dictionaries to JSON
        return {"data": results}

    finally:
        cursor.close()
        connection.close()



# Receives image from frontend
@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):

    print("Detected")
    # Ensure the upload directory exists
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    # Define the path to save the file
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    # Save the image locally
    with open(file_path, "wb") as f:
        f.write(await file.read())


    # Runs the model automatically
    file_path, acneCover, acne_areas = run_segment(file_path, model)

    # Returns model output
    return {"file_path" : str(file_path), "cover": float(acneCover), "number": float(acne_areas)}


class FilePathRequest(BaseModel):
    file_path: str

# For requesting a model from the backend
@app.get("/get-image/")
async def get_image(file_path: str = Query(...)):
    print("HERE " + file_path)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    else:
        raise HTTPException(status_code=404, detail="Image not found")


# Authentication, checks to see if user exists in users table.
@app.get("/authenticate/{user_id}")
async def auth(user_id: str):
    print("Id is " + user_id)
    exists = authenticate_user(user_id)
    return exists







# Database Interaction Functions

# Debugging to ensure structure is correct
def print_table_structure(table_name):
    try:
        # Get a connection from the pool
        cursor, connection = get_cursor()

        # Execute DESCRIBE query
        query = f"DESCRIBE {table_name};"
        cursor.execute(query)

        # Fetch and display results
        columns = cursor.fetchall()
        print(f"Structure of table '{table_name}':")
        print(f"{'Field':<20}{'Type':<20}{'Null':<10}{'Key':<10}{'Default':<20}{'Extra':<10}")
        print("-" * 90)

        for col in columns:
            print(f"{col[0]:<20}{col[1]:<20}{col[2]:<10}{col[3]:<10}{str(col[4]):<20}{col[5]:<10}")

    finally:
        # Close the connection

        cursor.close()
        if connection.is_connected():
            connection.close()

# This is for debugging
def get_inputs(user_id):

    cursor, connection = get_cursor()

    with open("sq/num_inputs.sql", 'r') as f:
        cmd = f.read()

    cmd += user_id

    print(cmd)
    #cursor.execute(command)

    cursor.close()
    connection.close()

# Authenticates the user
def authenticate_user(user_id: str):


    cursor, connection = get_cursor()

    query = "SELECT EXISTS( SELECT 1 FROM users WHERE user_id = %s) AS user_exists"
    cursor.execute(query, (user_id,))

    result = cursor.fetchone()

    cursor.close()
    connection.close()

    return result[0] == 1

# Adds a new user if the user doesn't exist already
def add_user(user_id, name, email):
    cursor, connection = get_cursor()

    try:
        query = "INSERT INTO users (user_id, name, email) VALUES (%s, %s, %s)"
        cursor.execute(query, (user_id, name, email,))
        connection.commit()
        return (f"User {user_id} : {name} added successfully")

    except mysql.connector.IntegrityError as e:
        raise HTTPException(status_code=400, detail="User ID already exists.")

    finally:
        cursor.close()
        connection.close()

# Removes a user
def remove_user(user_id):
    cursor, connection = get_cursor()
    query = "DELETE FROM users WHERE user_id = %s"
    cursor.execute(query, (user_id, ))
    connection.commit()
    print(f"User {user_id} removed successfully")
    cursor.close()
    connection.close()


# Adds data into the user_data table
def add_data(acne_cells, acne_coverage, date, user_id, file_name):

    cursor, connection = get_cursor()

    query = "INSERT INTO user_data (acne_cells, acne_coverage, date, user_id, filename) VALUES (%s, %s, %s, %s, %s)"

    cursor.execute(query, (acne_cells, acne_coverage, date, user_id, file_name,))
    connection.commit()
    print(f"Data {user_id} : {file_name} added successfully")
    cursor.close()
    connection.close()


def get_data(user_id, limit):
    cursor, connection = get_cursor()
    # Selects queries of limit, orders by most recent
    query = """
    SELECT * FROM user_data 
    WHERE user_id = %s
    ORDER BY date DESC
    LIMIT %s
    """
    cursor.execute(query, (user_id, limit))

    results = cursor.fetchall()

    cursor.close()
    connection.close()

    return results
