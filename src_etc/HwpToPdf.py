import os
import win32com.client as win32
import win32gui
import win32con
import time
import tkinter as tk
from tkinter import filedialog, messagebox
import sys

def hwp_to_pdf(hwp_path, pdf_path):
#    hwp = win32.client.gencache.EnsureDispatch("HWPFrame.HwpObject")
    hwp = win32.gencache.EnsureDispatch("HWPFrame.HwpObject")
    hwp.RegisterModule("FilePathCheckDLL", "FilePathCheckerModule")
    
    try:
        hwp.Open(hwp_path)
        hwp.SaveAs(pdf_path, "PDF")
        pdf_path = pdf_path.replace("\\", "/")
        print(f"변환 완료: {pdf_path}")
    except Exception as e:
        print(f"변환 중 오류 발생: {str(e)}")
    finally:
        hwp.Quit()

def close_hwp_popups():
    def callback(hwnd, _):
        if win32gui.GetWindowText(hwnd) == "한글":
            win32gui.PostMessage(hwnd, win32con.WM_CLOSE, 0, 0)
    
    win32gui.EnumWindows(callback, None)

def handle_security_dialog():
    def callback(hwnd, _):
        window_text = win32gui.GetWindowText(hwnd)
        if "보안" in window_text or "HwpCtrl" in window_text:  # 다이얼로그 제목에 "보안" 또는 "접근"이 포함된 경우
            # '모두 허용' 버튼 클릭
            allow_button = win32gui.FindWindowEx(hwnd, 0, "Button", "모두 허용(&N)")
            if allow_button:
                win32gui.PostMessage(allow_button, win32con.WM_LBUTTONDOWN, 0, 0)
                win32gui.PostMessage(allow_button, win32con.WM_LBUTTONUP, 0, 0)
            else:
                print("'모두 허용' 버튼을 찾을 수 없습니다.")
    
    win32gui.EnumWindows(callback, None)

def handle_completion_dialog():
    def callback(hwnd, _):
        if "완료" in win32gui.GetWindowText(hwnd):
            # '확인' 버튼 클릭
            ok_button = win32gui.FindWindowEx(hwnd, 0, "Button", "확인")
            win32gui.PostMessage(ok_button, win32con.WM_LBUTTONDOWN, 0, 0)
            win32gui.PostMessage(ok_button, win32con.WM_LBUTTONUP, 0, 0)
    
    win32gui.EnumWindows(callback, None)

def select_directory(title):
    root = tk.Tk()
    root.withdraw()
    directory = filedialog.askdirectory(title=title)
    return directory

def main():
    # HWP 파일이 있는 디렉토리 선택
    hwp_dir = select_directory("HWP 파일이 있는 디렉토리를 선택하세요")
    if not hwp_dir:
        messagebox.showwarning("경고", "HWP 파일 디렉토리가 선택되지 않았습니다. 프로그램을 종료합니다.")
        return

    # PDF 파일을 저장할 디렉토리 선택
    pdf_dir = select_directory("PDF 파일을 저장할 디렉토리를 선택하세요")
    if not pdf_dir:
        messagebox.showwarning("경고", "PDF 저장 디렉토리가 선택되지 않았습니다. 프로그램을 종료합니다.")
        return

    # 선택된 디렉토리 출력
    print(f"선택된 HWP 파일 디렉토리: {hwp_dir}")
    print(f"선택된 PDF 저장 디렉토리: {pdf_dir}")

    # 디렉토리가 없으면 생성
    if not os.path.exists(pdf_dir):
        os.makedirs(pdf_dir)
   
    
    # HWP 파일 목록 가져오기
    hwp_files = [f for f in os.listdir(hwp_dir) if f.endswith('.hwp')]
    
    if not hwp_files:
        messagebox.showinfo("정보", "선택한 디렉토리에 HWP 파일이 없습니다.")
        return

    for hwp_file in hwp_files:
        hwp_path = os.path.join(hwp_dir, hwp_file)
        pdf_path = os.path.join(pdf_dir, os.path.splitext(hwp_file)[0] + '.pdf')
        
        # 보안 다이얼로그 처리
        #handle_security_dialog()
        
        hwp_to_pdf(hwp_path, pdf_path)
        
        # 완료 다이얼로그 처리
        time.sleep(0.5)  # 다이얼로그가 나타날 시간을 주기 위해 잠시 대기
        handle_completion_dialog()
        
        # 팝업 창 닫기
        time.sleep(0.5)  # 팝업이 나타날 시간을 주기 위해 잠시 대기
        close_hwp_popups()

    print("모든 파일 변환이 완료되었습니다.")
    messagebox.showinfo("완료", "[by mbs] 모든 파일 변환이 완료되었습니다. 프로그램을 종료합니다.")
    sys.exit()

if __name__ == "__main__":
    main()