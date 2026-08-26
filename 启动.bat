@echo off
chcp 65001 >nul
title 产品净值计算 - 一键启动

echo ============================================
echo   产品净值计算 - AI 策略归因分析系统
echo ============================================
echo.

:: 检查后端 venv
if not exist "%~dp0backend\.venv\Scripts\python.exe" (
    echo [错误] 未找到后端虚拟环境: backend\.venv
    echo 请先运行: cd backend ^&^& python -m venv .venv ^&^& .venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

:: 检查前端 node_modules
if not exist "%~dp0frontend\node_modules" (
    echo [提示] 首次运行，正在安装前端依赖...
    cd /d "%~dp0frontend"
    call npm install --silent
    cd /d "%~dp0"
)

echo [1/2] 启动后端服务 (http://127.0.0.1:8103) ...
start "后端 - FastAPI" cmd /k "cd /d "%~dp0backend" && .venv\Scripts\python.exe -X utf8 -m uvicorn app.main:app --host 127.0.0.1 --port 8103 --reload"

:: 等待后端就绪
timeout /t 2 /nobreak >nul

echo [2/2] 启动前端服务 (http://localhost:5275) ...
start "前端 - Vite" cmd /k "cd /d "%~dp0frontend" && npx vite --host 127.0.0.1 --port 5275"

:: 等待前端就绪后打开浏览器
timeout /t 3 /nobreak >nul
start http://localhost:5275

echo.
echo ============================================
echo   启动完成！浏览器已打开 http://localhost:5275
echo   关闭此窗口不影响服务运行。
echo   如需停止，请关闭"后端"和"前端"两个命令行窗口。
echo ============================================
echo.
pause
