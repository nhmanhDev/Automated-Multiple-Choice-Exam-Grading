from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, RedirectResponse
import tempfile
import os
import uuid
from main import process_exam_sheet
from pdf2image import convert_from_bytes
from PIL import Image
import io

app = FastAPI()

# Tạo thư mục 'results' và 'static' nếu chưa tồn tại
if not os.path.exists("results"):
    os.makedirs("results")
if not os.path.exists("static"):
    os.makedirs("static")

@app.post("/upload-image")
async def upload_files(image: UploadFile = File(...), answer_key: UploadFile = File(...)):
    """Xử lý ảnh bài thi và file đáp án tải lên, trả về kết quả và URL ảnh chú thích."""
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Đọc dữ liệu file ảnh vào bộ nhớ
            image_content = await image.read()
            file_extension = image.filename.split('.')[-1].lower()

            # Đường dẫn file tạm thời cho ảnh JPG
            image_path = os.path.join(temp_dir, "image.jpg")

            # Xử lý theo định dạng file
            if file_extension in ["jpg", "jpeg"]:
                # Nếu là JPG, lưu trực tiếp
                with open(image_path, "wb") as f:
                    f.write(image_content)
            elif file_extension == "png":
                # Nếu là PNG, chuyển sang JPG
                img = Image.open(io.BytesIO(image_content))
                img = img.convert("RGB")
                img.save(image_path, "JPEG")
            elif file_extension == "pdf":
                # Nếu là PDF, chuyển sang JPG
                try:
                    pages = convert_from_bytes(image_content)
                    if not pages:
                        raise ValueError("File PDF rỗng hoặc không thể đọc.")
                    pages[0].save(image_path, "JPEG")  # Lưu trang đầu tiên
                except Exception as pdf_error:
                    raise ValueError(f"Lỗi khi chuyển PDF sang JPG: {str(pdf_error)}")
            else:
                raise ValueError("Định dạng file không hỗ trợ. Chỉ hỗ trợ JPG, PNG, PDF.")

            # Lưu file đáp án vào file tạm thời
            answer_key_path = os.path.join(temp_dir, "answer_key.xlsx")
            with open(answer_key_path, "wb") as f:
                f.write(await answer_key.read())

            # Tạo tên file duy nhất cho ảnh chú thích
            output_image_name = f"result_{uuid.uuid4().hex}.jpg"
            output_image_path = os.path.join("results", output_image_name)

            # Gọi hàm xử lý từ main.py
            results = process_exam_sheet(image_path, answer_key_path, output_image_path)

            # Chuẩn bị phản hồi JSON
            response = {
                "status": results["status"],
                "annotated_image_url": f"/results/{output_image_name}"
            }
            if results["status"] == "success":
                response.update({
                    "sbd": results["sbd"],
                    "mdt": results["mdt"],
                    "correct_answers": results["correct_answers"],
                    "total_questions": results["total_questions"],
                    "final_score": results["final_score"]
                })
            else:
                response["error"] = results["message"]

            return JSONResponse(content=response)

        except Exception as e:
            return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/")
async def redirect_to_index():
    return RedirectResponse(url="/static/index.html")

app.mount("/static", StaticFiles(directory="static", html=True), name="static")
app.mount("/results", StaticFiles(directory="results"), name="results")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)