import os
from hwp5.dataio import ParseError
from hwp5.xmlmodel import Hwp5File

def extract_text_from_hwp(file_path):
    try:
        hwp = Hwp5File(file_path)
        content = ""
        for section in hwp.bodytext.sections:
            for paragraph in section.paragraphs:
                content += paragraph.text + "\n"
        return content
    except ParseError:
        return "HWP 파일을 파싱할 수 없습니다."
    except Exception as e:
        return f"오류 발생: {str(e)}"

# 기존의 hwp_to_pdf 함수를 대체
def hwp_to_pdf(input_file, output_file):
    text_content = extract_text_from_hwp(input_file)
    # 텍스트 내용을 PDF로 저장하는 로직 추가
    # 예: reportlab 라이브러리를 사용하여 PDF 생성
    # 여기서는 간단히 텍스트 파일로 저장
    with open(output_file.replace('.pdf', '.txt'), 'w', encoding='utf-8') as f:
        f.write(text_content)
    print(f"텍스트 내용이 {output_file.replace('.pdf', '.txt')}에 저장되었습니다.")