# Automated Multiple-Choice Exam Grading

## Project Overview
This project is designed to automate the grading process for multiple-choice exams. It simplifies the evaluation process, reduces human error, and provides quick results.

## Features
- Upload scanned answer sheets.
- Define answer keys for exams.
- Automatically grade exams and calculate scores.
- Generate detailed reports for students and instructors.

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/nhmanhDev/Automated-Multiple-Choice-Exam-Grading.git
    ```
2. Navigate to the project directory:
    ```bash
    cd Automated-Multiple-Choice-Exam-Grading
    ```
3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
1. Prepare the answer key and scanned answer sheets.
2. Run the grading script:
    ```bash
    python main.py
    ```
3. View the generated reports in the `output` folder.

## Technologies Used
- Python
- OpenCV (for image processing)
- Pandas (for data handling)
- Matplotlib (for report visualization)
- CNN

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact
For questions or feedback, please contact [nhmanh.dev@gmail.com].


## Requirements
Create a `requirements.txt` file with the following dependencies:

```
python==3.10.12
tensorflow==2.12.0
numpy==1.26.4
pandas==2.2.3
matplotlib==3.6.2
scikit-learn==1.1.3
flask==2.2.3
opencv-python==4.11.0.80
Pillow==10.3.0
pdf2image==1.17.0
imutils==0.5.4
```

These libraries are essential for the project's functionality. Install them using the command:

```bash
pip install -r requirements.txt
```