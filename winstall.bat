@echo off
python -m venv venv
call ragenv\Scripts\activate
pip install -r requirements.txt
echo 설치가 완료되었습니다.
pause