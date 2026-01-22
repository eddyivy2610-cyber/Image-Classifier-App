#TO RUN APP#

cd Image-Classifier-App\backend
venv\Scripts\activate

uvicorn app:app --reload

cd Image-Classifier-App\frontend\client

npm run dev