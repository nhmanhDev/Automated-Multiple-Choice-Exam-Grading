import cv2
import pandas as pd
from process_sbd_mdt import (
    resize_image, detect_mid_contours, process_sbd_id_block, process_mdt_block, detect_mid_contours_with_coords,
    process_all_columns, check_all_columns_filled, convert_filled_to_numbers_per_column, annotate_block
)
from process_answer import crop_image, process_ans_blocks, process_list_ans, get_answers, annotate_answers
from pdf2image import convert_from_path
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # Bỏ giới hạn kiểm tra kích thước ảnh lớn


def read_answer_key(answer_key_path):
    try:
        df = pd.read_excel(answer_key_path)
        answer_key = {row.iloc[0]: row.iloc[1] for _, row in df.iterrows()}
        return answer_key
    except Exception as e:
        print(f"❌ Lỗi khi đọc file Excel: {e}")
        exit(1)

def calculate_score(answers_exam, answer_key):
    score = 0
    total_questions = len(answer_key)
    for question, correct_answer in answer_key.items():
        student_answer = answers_exam.get(question, [])
        if len(student_answer) != 1:
            continue  
        if student_answer[0] == correct_answer:
            score += 1
    return score, total_questions

def extract_id_and_code(result_sbd, result_mdt):
    sbd_str = "".join(str(col[0]) for col in result_sbd)
    mdt_str = "".join(str(col[0]) for col in result_mdt)
    return sbd_str, mdt_str

def convert_pdf_to_images(pdf_path, output_format="jpeg", dpi=300, output_size=(1056, 1500)):
    """
    Chuyển file PDF thành ảnh JPEG, resize về (1056, 1500) và lưu lại.

    Args:
        pdf_path (str): Đường dẫn file PDF.
        output_format (str): Định dạng file ảnh đầu ra ("jpeg" hoặc "png").
        dpi (int): Độ phân giải của ảnh khi chuyển đổi.
        output_size (tuple): Kích thước ảnh sau khi resize (mặc định 1056x1500).

    Returns:
        List[str]: Danh sách các file ảnh đã lưu.
    """
    # Chuyển đổi PDF thành danh sách ảnh (mỗi trang là 1 ảnh)
    images = convert_from_path(pdf_path, dpi=dpi)

    output_files = []
    for i, image in enumerate(images):
        # Chuyển sang chế độ RGB để lưu thành JPEG
        image = image.convert("RGB")

        # Resize ảnh về kích thước chuẩn (1056x1500)
        image = image.resize(output_size, resample=Image.LANCZOS)

        # Đặt tên file và lưu dưới định dạng JPEG
        output_file = f"page_{i+1}.{output_format}"
        image.save(output_file, "JPEG", quality=95)  # Lưu với chất lượng cao
        output_files.append(output_file)

        # print(f"Đã lưu {output_file} với kích thước {output_size}")

    return output_files


def main(image_path, answer_key_path):
    
    # Resize image
    output_resized_path = "output_resized.jpg"
    resize_image(image_path, output_resized_path)

    # Detect and extract student ID and test code regions
    sbd, mdt = detect_mid_contours(output_resized_path)
    if sbd is None or mdt is None:
        print("❌ Không thể phát hiện đủ vùng cần thiết.")
        return

    # Process student ID
    sbd_columns = process_sbd_id_block(sbd)
    all_sbd_cells = process_all_columns(sbd_columns)
    filled_sbd = check_all_columns_filled(all_sbd_cells)
    result_sbd = convert_filled_to_numbers_per_column(filled_sbd, 6)

    # Process test code
    mdt_columns = process_mdt_block(mdt)
    all_mdt_cells = process_all_columns(mdt_columns)
    filled_mdt = check_all_columns_filled(all_mdt_cells)
    result_mdt = convert_filled_to_numbers_per_column(filled_mdt, 3)

    # Annotate images (lưu riêng cho debug)
    annotate_block(sbd_columns, filled_sbd, label="sbd")
    annotate_block(mdt_columns, filled_mdt, label="mdt")

    # Lấy tọa độ SBD và MDT
    sbd, mdt, sbd_coords, mdt_coords = detect_mid_contours_with_coords(output_resized_path)
    sbd_x, sbd_y, sbd_w, sbd_h = sbd_coords
    mdt_x, mdt_y, mdt_w, mdt_h = mdt_coords

    img_annotated = cv2.imread(output_resized_path)

    # Kiểm tra hợp lệ của SBD & MDT, vẽ hộp lỗi nếu có
    for i, col in enumerate(result_sbd):
        if len(col) == 0:
            print(f"❌ Lỗi: Cột {i+1} của Số báo danh không có giá trị.")
            # Vẽ hộp đỏ đậm quanh cột lỗi (không có giá trị)
            error_x = sbd_x + (i * sbd_w // 6)
            cv2.rectangle(img_annotated, (error_x, sbd_y), (error_x + sbd_w // 6, sbd_y + sbd_h), (0, 0, 255), 4)
            cv2.imwrite('error_image.jpg', img_annotated)
            exit(1)
        if len(col) > 1:
            print(f"❌ Lỗi: Cột {i+1} của Số báo danh có nhiều hơn một giá trị: {col}.")
            # Vẽ hộp đỏ đậm quanh cột lỗi (nhiều giá trị)
            error_x = sbd_x + (i * sbd_w // 6)
            cv2.rectangle(img_annotated, (error_x, sbd_y), (error_x + sbd_w // 6, sbd_y + sbd_h), (0, 0, 255), 4)
            cv2.imwrite('error_image.jpg', img_annotated)
            exit(1)

    for i, col in enumerate(result_mdt):
        if len(col) == 0:
            print(f"❌ Lỗi: Cột {i+1} của Mã đề thi không có giá trị.")
            # Vẽ hộp đỏ đậm quanh cột lỗi (không có giá trị)
            error_x = mdt_x + (i * mdt_w // 3)
            cv2.rectangle(img_annotated, (error_x, mdt_y), (error_x + mdt_w // 3, mdt_y + mdt_h), (0, 0, 255), 4)
            cv2.imwrite('error_image.jpg', img_annotated)
            exit(1)
        if len(col) > 1:
            print(f"❌ Lỗi: Cột {i+1} của Mã đề thi có nhiều hơn một giá trị: {col}.")
            # Vẽ hộp đỏ đậm quanh cột lỗi (nhiều giá trị)
            error_x = mdt_x + (i * mdt_w // 3)
            cv2.rectangle(img_annotated, (error_x, mdt_y), (error_x + mdt_w // 3, mdt_y + mdt_h), (0, 0, 255), 4)
            cv2.imwrite('error_image.jpg', img_annotated)
            exit(1)

    # Chuyển danh sách thành chuỗi số
    sbd_str, mdt_str = extract_id_and_code(result_sbd, result_mdt)

    # Process answer sheet
    img = cv2.imread(output_resized_path)
    list_ans_boxes = crop_image(img)
    list_ans = process_ans_blocks(list_ans_boxes)
    list_ans = process_list_ans(list_ans)
    answers = get_answers(list_ans)
    # print(answers)

    # Đọc file đáp án từ Excel
    answer_key = read_answer_key(answer_key_path)

    # Tính điểm
    score, total_questions = calculate_score(answers, answer_key)
    final_score = score * 10 / total_questions

    # In kết quả
    print(f"Số câu đúng: {score}/{total_questions}.\nThí sinh số báo danh {sbd_str} mã đề {mdt_str}: {final_score:.2f} điểm.")

    # Bắt đầu vẽ lên ảnh gốc

    # Vẽ SBD
    for col_idx, col in enumerate(result_sbd):
        for row_idx in col:
            cell_y = sbd_y + (row_idx * sbd_h // 10)
            cell_x = sbd_x + (col_idx * sbd_w // 6)
            cv2.rectangle(img_annotated, (cell_x, cell_y), (cell_x + sbd_w // 6, cell_y + sbd_h // 10), (0, 255, 0), 2)

    # Vẽ MDT
    for col_idx, col in enumerate(result_mdt):
        for row_idx in col:
            cell_y = mdt_y + (row_idx * mdt_h // 10)
            cell_x = mdt_x + (col_idx * mdt_w // 3)
            cv2.rectangle(img_annotated, (cell_x, cell_y), (cell_x + mdt_w // 3, cell_y + mdt_h // 10), (0, 255, 0), 2)

    # Thêm text theo thứ tự từ trên xuống dưới ở góc trên bên trái
    text_x = 50  # Tọa độ x cố định
    text_y_start = 150  # Tọa độ y bắt đầu
    line_spacing = 80  # Khoảng cách giữa các dòng

    # SBD
    cv2.putText(img_annotated, f"SO BAO DANH: {sbd_str}", (text_x, text_y_start), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 51, 51), 2)
    
    # MDT
    cv2.putText(img_annotated, f"MA DE THI: {mdt_str}", (text_x, text_y_start + line_spacing), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 51, 51), 2)
    
    # Số câu đúng
    cv2.putText(img_annotated, f"TONG SO CAU DUNG: {score}/{total_questions}", (text_x, text_y_start + 2 * line_spacing), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 51, 51), 2)
    
    # Điểm
    cv2.putText(img_annotated, f"DIEM: {final_score:.2f}", (text_x, text_y_start + 3 * line_spacing), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 51, 51), 2)

    # Vẽ đáp án lên cùng ảnh đã vẽ SBD/MDT
    annotate_answers(list_ans_boxes, answers, answer_key, questions_per_block=30, img=img_annotated)

    # Lưu ảnh cuối cùng
    cv2.imwrite('Final_result.jpg', img_annotated)



if __name__ == "__main__":

    # # Convert pdf to JPEG
    # pdf_path = "Test_pdf1.pdf"  # Thay bằng đường dẫn file PDF thực tế
    # output_files = convert_pdf_to_images(pdf_path, output_format="jpeg")

    # if output_files:
    #     first_image = Image.open(output_files[0])
    #     first_image = first_image.resize((1056, 1500), resample=Image.LANCZOS)
    #     image_path = "input.jpg"
    #     first_image.save(image_path, "JPEG")

    # Nếu không cần chuyển đổi PDF, có thể sử dụng ảnh JPEG đã có sẵn
    image_path = "Exam/Test678.jpg"
    answer_key_path = "AnswerKey/678.xlsx"
    main(image_path, answer_key_path)