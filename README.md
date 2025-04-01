AUTOMATED MULTIPLE-CHOICE EXAM GRADING

PROJECT OVERVIEW
This project automates the grading process for multiple-choice exams by processing scanned answer sheets and answer keys. It aims to simplify evaluation, reduce human error, and provide quick, accurate results with a user-friendly interface.

FEATURES

Upload scanned answer sheets (images).
Upload answer keys (Excel files).
Automatically grade exams and calculate scores.
Display results with annotated images.
Download graded results as images.
Web-based user interface for easy interaction.
REQUIREMENTS

Python 3.7+
A web browser (Chrome, Firefox, etc.)
Optional: Ngrok (for public access)
Create a requirements.txt file with the following dependencies:
fastapi==0.110.0
uvicorn==0.29.0
opencv-python==4.11.0.80
pandas==2.2.3
Pillow==10.3.0

Install them using:
pip install -r requirements.txt

INSTALLATION

Clone the repository: git clone https://github.com/nhmanhDev/Automated-Multiple-Choice-Exam-Grading.git
Navigate to the project directory: cd Automated-Multiple-Choice-Exam-Grading
Install dependencies: pip install -r requirements.txt
USAGE

Running Locally

Prepare Files:
Ensure you have scanned answer sheets (image files, e.g., .jpg) and an answer key (Excel file, e.g., .xlsx).
Run the Server:
Open a terminal in the project directory and start the FastAPI server: uvicorn main:app --host 0.0.0.0 --port 8000
Access the User Interface:
Open a web browser and go to: http://localhost:8000/static/index.html
The interface will load, allowing you to upload an answer sheet and answer key.
Using the Interface:
Upload the scanned answer sheet (image) and answer key (Excel file).
Click "Tải lên" to process.
View results (student ID, test code, score, etc.) and the annotated image.
Click "Tải ảnh xuống" to download the graded image.
Running Publicly (Optional with Ngrok)
If you want others to access your interface over the internet:

Download Ngrok:
Get it from https://ngrok.com/download and extract ngrok.exe into the project directory.
Run the Server:
Start the FastAPI server (as above): uvicorn main:app --host 0.0.0.0 --port 8000
Run Ngrok:
Open a second terminal in the project directory and run: ./ngrok http 8000
Ngrok will provide a public URL (e.g., https://xxxx.ngrok-free.app).
Access Publicly:
Use the Ngrok URL with the interface path: https://xxxx.ngrok-free.app/static/index.html
Share this URL with others.
Notes:
Without an Ngrok account, tunnels last 2 hours and URLs change each run.
Register at https://ngrok.com for a free authtoken to remove the time limit: ./ngrok authtoken <your-authtoken>
TECHNOLOGIES USED

Python: Core programming language.
FastAPI: Web framework for the backend.
Uvicorn: ASGI server to run FastAPI.
OpenCV: Image processing for answer sheet analysis.
Pandas: Handling answer key data.
Pillow: Image manipulation.
HTML/CSS/JavaScript: Frontend interface.
CONTRIBUTING
Contributions are welcome! Please fork the repository and submit a pull request.

LICENSE
This project is licensed under the MIT License. See the LICENSE file for details.

CONTACT
For questions or feedback, please contact nhmanh.dev@gmail.com.