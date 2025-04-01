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

class ProcessingError(Exception):
    pass

def read_answer_key(answer_key_path):
    try:
        df = pd.read_excel(answer_key_path)
        answer_key = {row.iloc[0]: row.iloc[1] for _, row in df.iterrows()}
        answer_key = dict(list(answer_key.items())[:120])  # Giới hạn tối đa 120 giá trị
        return answer_key
    except Exception as e:
        raise ProcessingError(f"Lỗi khi đọc file Excel: {e}")


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

def process_exam_sheet(image_path, answer_key_path, output_image_path):
    # Resize image
    output_resized_path = "output_resized.jpg"
    resize_image(image_path, output_resized_path)

    # Detect and extract student ID and test code regions
    sbd, mdt = detect_mid_contours(output_resized_path)
    if sbd is None or mdt is None:
        img_annotated = cv2.imread(output_resized_path)
        cv2.putText(img_annotated, "Error: Cannot detect required regions", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 5)

        cv2.imwrite(output_image_path, img_annotated)
        return {"status": "error", "message": "Không thể phát hiện đủ vùng cần thiết"}

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

    # Annotate images (for debug)
    annotate_block(sbd_columns, filled_sbd, label="sbd")
    annotate_block(mdt_columns, filled_mdt, label="mdt")

    # Get coordinates for SBD and MDT
    sbd, mdt, sbd_coords, mdt_coords = detect_mid_contours_with_coords(output_resized_path)
    sbd_x, sbd_y, sbd_w, sbd_h = sbd_coords
    mdt_x, mdt_y, mdt_w, mdt_h = mdt_coords

    img_annotated = cv2.imread(output_resized_path)

    # Check validity of SBD and MDT, draw error boxes if invalid
    sbd_error = False
    for i, col in enumerate(result_sbd):
        if len(col) == 0:
            sbd_error = True
            error_x = sbd_x + (i * sbd_w // 6)
            cv2.rectangle(img_annotated, (error_x, sbd_y), (error_x + sbd_w // 6, sbd_y + sbd_h), (0, 0, 255), 4)
        if len(col) > 1:
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
        cv2.imwrite(output_image_path, img_annotated)
        message = "SBD hoặc MDT không hợp lệ"
        if sbd_error:
            message += ": Lỗi SBD"
        if mdt_error:
            message += ": Lỗi MDT"
        return {"status": "error", "message": message}

    # Convert lists to strings
    sbd_str, mdt_str = extract_id_and_code(result_sbd, result_mdt)

    # Process answer sheet
    img = cv2.imread(output_resized_path)
    list_ans_boxes = crop_image(img)
    list_ans = process_ans_blocks(list_ans_boxes)
    list_ans = process_list_ans(list_ans)
    answers = get_answers(list_ans)

    # Read answer key from Excel
    answer_key = read_answer_key(answer_key_path)

    # Calculate score
    score, total_questions = calculate_score(answers, answer_key)
    final_score = score * 10 / total_questions

    # Draw SBD
    for col_idx, col in enumerate(result_sbd):
        for row_idx in col:
            cell_y = sbd_y + (row_idx * sbd_h // 10)
            cell_x = sbd_x + (col_idx * sbd_w // 6)
            cv2.rectangle(img_annotated, (cell_x, cell_y), (cell_x + sbd_w // 6, cell_y + sbd_h // 10), (0, 255, 0), 2)

    # Draw MDT
    for col_idx, col in enumerate(result_mdt):
        for row_idx in col:
            cell_y = mdt_y + (row_idx * mdt_h // 10)
            cell_x = mdt_x + (col_idx * mdt_w // 3)
            cv2.rectangle(img_annotated, (cell_x, cell_y), (cell_x + mdt_w // 3, cell_y + mdt_h // 10), (0, 255, 0), 2)

    # Add text from top to bottom on the top-left corner
    text_x = 50
    text_y_start = 150
    line_spacing = 80

    cv2.putText(img_annotated, f"SO BAO DANH: {sbd_str}", (text_x, text_y_start), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 51, 51), 2)
    cv2.putText(img_annotated, f"MA DE THI: {mdt_str}", (text_x, text_y_start + line_spacing), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 51, 51), 2)
    cv2.putText(img_annotated, f"TONG SO CAU DUNG: {score}/{total_questions}", (text_x, text_y_start + 2 * line_spacing), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 51, 51), 2)
    cv2.putText(img_annotated, f"DIEM: {final_score:.2f}", (text_x, text_y_start + 3 * line_spacing), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 51, 51), 2)

    # Annotate answers on the same image
    annotate_answers(list_ans_boxes, answers, answer_key, questions_per_block=30, img=img_annotated)

    # Save the final annotated image
    cv2.imwrite(output_image_path, img_annotated)

    # Return results
    return {
        "status": "success",
        "sbd": sbd_str,
        "mdt": mdt_str,
        "correct_answers": score,
        "total_questions": total_questions,
        "final_score": final_score
    }

if __name__ == "__main__":
    image_path = "Exam/Test678_Loi09.jpg"
    answer_key_path = "AnswerKey/001.xlsx"
    output_image_path = "Final_result.jpg"  # For testing
    results = process_exam_sheet(image_path, answer_key_path, output_image_path)
    print(results)