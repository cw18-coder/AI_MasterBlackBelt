@echo off
echo Installing required packages...
pip install -r requirements_upload.txt

echo.
echo Running upload script...
python upload_to_huggingface.py

pause
