# Sử dụng image Python 3.10 làm base
FROM python:3.10

# Cài đặt các phụ thuộc hệ thống cần thiết
RUN apt-get update && apt-get install -y \
    poppler-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src

# RUN apt-get update 
# RUN apt-get -y install python3

# Cập nhật pip để tránh cảnh báo (tùy chọn)
RUN pip install --upgrade pip

# Sao chép file requirements.txt và cài đặt các thư viện
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép các file Python và file trọng số
COPY user_interface.py ./user_interface.py
COPY main.py ./main.py
COPY process_sbd_mdt.py ./process_sbd_mdt.py
COPY process_answer.py ./process_answer.py
COPY model_answer.py ./model_answer.py
COPY weight.keras ./weight.keras
RUN ls -l /src/weight.keras  # Add this line to check if the file exists

# Sao chép thư mục static cho giao diện Front-end
COPY favicon.ico ./favicon.ico
COPY static/ static/


# Sao chép thư mục Exam và AnswerKey vào thư mục làm việc
COPY create_dataset/ create_dataset/
COPY Exam/ Exam/
COPY AnswerKey/ AnswerKey/
COPY results/ results/


# Mở cổng 8000 để truy cập vào ứng dụng FastAPI
EXPOSE 8000

# Chạy ứng dụng FastAPI với uvicorn
CMD ["uvicorn", "user_interface:app", "--host", "0.0.0.0", "--port", "8000"]
# CMD ["uvicorn", "user_interface:app", "--reload"]
