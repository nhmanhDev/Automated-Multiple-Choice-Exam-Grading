import cv2
import pandas as pd
from process_sbd_mdt import (
    detect_mid_contours, process_sbd_id_block, process_mdt_block, detect_mid_contours_with_coords,
    process_all_columns, check_all_columns_filled, convert_filled_to_numbers_per_column, annotate_block
)
from process_answer import crop_image, process_ans_blocks, process_list_ans, get_answers, annotate_answers
from pdf2image import convert_from_path
from PIL import Image
import logging

# Cấu hình logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

Image.MAX_IMAGE_PIXELS = None

class ProcessingError(Exception):
    pass

def read_answer_key(answer_key_path):
    try:
        logger.debug(f"Reading answer key from {answer_key_path}")
        df = pd.read_excel(answer_key_path)
        if df.shape[1] != 2 or 'STT' not in df.columns or 'Answer' not in df.columns:
            raise ProcessingError("File Excel không có cấu trúc đúng (cần cột 'STT' và 'Answer')")
        stt_values = df['STT'].tolist()
        expected_stt = list(range(1, 121))
        if stt_values != expected_stt:
            raise ProcessingError("Số thứ tự (STT) không đúng, phải từ 1 đến 120")
        valid_answers = {'A', 'B', 'C', 'D'}
        for answer in df['Answer']:
            if pd.isna(answer) or answer not in valid_answers:
                raise ProcessingError(f"Đáp án không hợp lệ: '{answer}'. Chỉ chấp nhận A, B, C, D")
        answer_key = dict(zip(df['STT'], df['Answer']))
        logger.debug(f"Answer key loaded with {len(answer_key)} questions")
        return answer_key
    except Exception as e:
        raise ProcessingError(f"Lỗi khi đọc file Excel: {e}")

def calculate_score(answers_exam, answer_key):
    score = 0
    total_questions = 120
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

def process_exam_sheet(image_path, answer_key_path, output_image_path):
    logger.info(f"Processing exam sheet: image={image_path}, answer_key={answer_key_path}")
    
    # Đọc ảnh
    img = cv2.imread(image_path)
    if img is None:
        logger.error(f"Cannot read image at {image_path}")
        return {"status": "error", "message": "Không thể đọc file ảnh"}

    # Phát hiện vùng SBD và MDT
    logger.debug("Detecting SBD and MDT contours")
    sbd, mdt = detect_mid_contours(image_path)
    if sbd is None or mdt is None:
        img_annotated = img.copy()
        cv2.putText(img_annotated, "Error: Cannot detect required regions", (30, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        cv2.imwrite(output_image_path, img_annotated)
        return {"status": "error", "message": "Không thể phát hiện đủ vùng cần thiết"}

    # Xử lý SBD
    logger.debug("Processing SBD")
    sbd_columns = process_sbd_id_block(sbd)
    all_sbd_cells = process_all_columns(sbd_columns)
    filled_sbd = check_all_columns_filled(all_sbd_cells)
    result_sbd = convert_filled_to_numbers_per_column(filled_sbd, 6)

    # Xử lý MDT
    logger.debug("Processing MDT")
    mdt_columns = process_mdt_block(mdt)
    all_mdt_cells = process_all_columns(mdt_columns)
    filled_mdt = check_all_columns_filled(all_mdt_cells)
    result_mdt = convert_filled_to_numbers_per_column(filled_mdt, 3)

    # Chú thích SBD và MDT (debug)
    annotate_block(sbd_columns, filled_sbd, label="sbd")
    annotate_block(mdt_columns, filled_mdt, label="mdt")

    # Lấy tọa độ
    sbd, mdt, sbd_coords, mdt_coords = detect_mid_contours_with_coords(image_path)
    sbd_x, sbd_y, sbd_w, sbd_h = sbd_coords
    mdt_x, mdt_y, mdt_w, mdt_h = mdt_coords

    img_annotated = img.copy()

    # Kiểm tra tính hợp lệ của SBD và MDT
    sbd_error = False
    for i, col in enumerate(result_sbd):
        if len(col) == 0 or len(col) > 1:
            sbd_error = True
            error_x = sbd_x + (i * sbd_w // 6)
            cv2.rectangle(img_annotated, (error_x, sbd_y), (error_x + sbd_w // 6, sbd_y + sbd_h), (0, 0, 255), 4)

    mdt_error = False
    for i, col in enumerate(result_mdt):
        if len(col) == 0 or len(col) > 1:
            mdt_error = True
            error_x = mdt_x + (i * mdt_w // 3)
            cv2.rectangle(img_annotated, (error_x, mdt_y), (error_x + mdt_w // 3, mdt_y + mdt_h), (0, 0, 255), 4)

    if sbd_error or mdt_error:
        # Tạo thông báo lỗi
        message = "SBD hoac MDT khong hop le"
        if sbd_error:
            message += ": SBD Error"
        if mdt_error:
            message += ": MDt Error"
        logger.error(message)

        # Vẽ thông báo lỗi lên ảnh
        text_x, text_y_start, line_spacing = 30, 100, 40
        cv2.putText(img_annotated, message, (text_x, text_y_start), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)  # Màu đỏ cho lỗi

        # Lưu ảnh đã chú thích
        if not cv2.imwrite(output_image_path, img_annotated):
            logger.error(f"Không thể lưu ảnh lỗi tại {output_image_path}")
            return {"status": "error", "message": "Không thể lưu ảnh kết quả"}

        # Trả về kết quả với thông báo lỗi
        return {"status": "error", "message": message}

    # Chuyển thành chuỗi
    sbd_str, mdt_str = extract_id_and_code(result_sbd, result_mdt)
    logger.debug(f"SBD: {sbd_str}, MDT: {mdt_str}")

    # Xử lý đáp án
    logger.debug("Cropping answer blocks")
    list_ans_boxes = crop_image(img)
    logger.info(f"Found {len(list_ans_boxes)} answer blocks")
    list_ans = process_ans_blocks(list_ans_boxes)
    list_ans = process_list_ans(list_ans)
    answers = get_answers(list_ans)
    logger.debug(f"Student answers: {len(answers)} questions")

    # Đọc đáp án mẫu
    answer_key = read_answer_key(answer_key_path)

    # Tính điểm
    score, total_questions = calculate_score(answers, answer_key)
    final_score = score * 10 / total_questions
    logger.debug(f"Score: {score}/{total_questions}, Final: {final_score}")

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

    # Vẽ thông tin
    text_x, text_y_start, line_spacing = 50, 100, 40
    cv2.putText(img_annotated, f"SO BAO DANH: {sbd_str}", (text_x, text_y_start), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 51, 51), 2)
    cv2.putText(img_annotated, f"MA DE THI: {mdt_str}", (text_x, text_y_start + line_spacing), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 51, 51), 2)
    cv2.putText(img_annotated, f"TONG SO CAU DUNG: {score}/{total_questions}", (text_x, text_y_start + 2 * line_spacing), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 51, 51), 2)
    cv2.putText(img_annotated, f"DIEM: {final_score:.2f}", (text_x, text_y_start + 3 * line_spacing), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 51, 51), 2)

    # Chú thích đáp án
    annotate_answers(list_ans_boxes, answers, answer_key, questions_per_block=30, img=img_annotated)

    # Lưu ảnh
    logger.debug(f"Saving annotated image to {output_image_path}")
    if not cv2.imwrite(output_image_path, img_annotated):
        logger.error(f"Failed to save image to {output_image_path}")
        return {"status": "error", "message": "Không thể lưu ảnh kết quả"}

    # Trả về kết quả
    result = {
        "status": "success",
        "sbd": sbd_str,
        "mdt": mdt_str,
        "correct_answers": score,
        "total_questions": total_questions,
        "final_score": final_score
    }
    logger.info("Processing completed successfully")
    return result

if __name__ == "__main__":
    image_path = "Exam/Test10diem.jpg"
    # output_resized = "output_resized.jpg"
    # image = cv2.imread(image_path)
    # cv2.imwrite(output_resized, cv2.resize(image, (1056, 1500)))

    # image_path = output_resized
    answer_key_path = "AnswerKey/FullC.xlsx"
    output_image_path = "Final_result.jpg"  # For testing
    results = process_exam_sheet(image_path, answer_key_path, output_image_path)
    print(results)