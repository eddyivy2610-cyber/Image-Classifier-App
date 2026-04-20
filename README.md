# IMAGE CLASSIFICATION WEB APP
(CIFAR-100 • CNN • FastAPI • React)

*This guide explains exactly how to run this project on any computer after downloading it from GitHub — even if you are not a developer.

# WHAT THIS PROJECT DOES

You upload an image (e.g. animal, object, vehicle)

The system:

> Sends the image to a backend server

> Uses a trained AI image recognition model

> Returns:

> The predicted class

> Confidence score

> Results are shown in a friendly web interface



# STEP 1 — INSTALL REQUIRED SOFTWARE
1️⃣ Install Python

> Go to: https://filehippo.com/download_python/3.10.0

> Download Python 3.10

> During installation:
✅ Check “Add Python to PATH”

> Click Install

Verify: run    

            python --version

in your terminal or command prompt   

2️⃣ Install Node.js

> Go to: https://nodejs.org/en/download/

> Download Node.js (LTS version)

> Verify: run    

            node --version

in your terminal or command prompt   


# STEP 2 — DOWNLOAD THE PROJECT

> GitHub Download

> Open the GitHub repository

> Click Code → Download ZIP

> Extract the ZIP file to a location of your choice (e.g. Desktop, Documents)



# STEP 3 — BACKEND SETUP (AI SERVER)

> Go to backend folder

> Open a terminal in the backend folder

  OR

> run this command in your terminal:

            cd image-classifier-app\backend

> NOTE: if extracted to desktop, run this command instead:

            cd Desktop\image-classifier-app\backend
    OR

        if extracted to documents:
            cd Documents\image-classifier-app\backend


> Create virtual environment (RECOMMENDED)

            python -m venv venv
            venv\Scripts\activate
    You should see: (venv) at the start of your terminal


> If you don't have a virtual environment, skip this step.

> Install dependencies:

            pip install -r requirements.txt

> Start Backend Server: run this command in your terminal:

            uvicorn app:app --reload

You should see: 

            INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
            INFO:     Started reloader process [12345]
            INFO:     Started server process [12346]
            INFO:     Waiting for application startup.
            INFO:     Application startup complete.
⚠️ Do NOT close this terminal


# STEP 4 — TEST BACKEND (OPTIONAL)

> Open your browser and go to:

            http://127.0.0.1:8000

You should see: 

            {"message": "CIFAR-10 API is running", "model_status": "loaded"}


# STEP 5 — FRONTEND SETUP (WEB APP) 

> open NEW terminal in the frontend folder      

            cd image-classifier-app\client\frontend
        OR

        if extracted to desktop:
            cd Desktop\image-classifier-app\client\frontend
    OR

        if extracted to documents:
            cd Documents\image-classifier-app\client\frontend

> Install dependencies run this command in your terminal:

            npm install

> Start Frontend Server: run this command in your terminal:

            npm run dev

You should see: 
            Local: http://localhost:5173/

> Open your browser and go to: 
            http://localhost:5173

# HOW TO STOP THE SERVERS:

> Backend Server: Press CTRL + C in the terminal
> Frontend Server: Press CTRL + C in the terminal
            


            


            
