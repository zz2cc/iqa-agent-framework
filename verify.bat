@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================
echo   IQA Agent 框架 — 复现验证
echo ============================================
echo.
echo [步骤 1/2] 缓存验证 + R6 管线重算 (零 API)
echo   预计 10-15 分钟，请耐心等待...
echo.

python -u scripts/verify.py
if %errorlevel% neq 0 (
    echo.
    echo [错误] 验证异常，请检查上述输出。
    pause
    exit /b 1
)

echo.
echo ============================================
echo   缓存验证完成。主表和 R6 管线结果见上方。
echo ============================================
echo.
set /p RUN_API="是否运行 API 抽样验证? (y/n, 约 1200 次调用, 5-10 元): "

if /i "%RUN_API%"=="y" (
    echo.
    echo [步骤 2/2] API 抽样验证 (KonIQ+SPAQ 各 200 张)
    echo   预计 5-10 分钟...
    echo.
    python -u scripts/verify.py --api-only
    echo.
    echo API 抽样完成。账本见上方。
) else (
    echo.
    echo [跳过] API 抽样验证。
)

echo.
set /p RUN_HTML="是否生成并打开 HTML 报告? (y/n): "

if /i "%RUN_HTML%"=="y" (
    echo.
    echo 正在生成 HTML 报告...
    python scripts/verify.py --html-only
    echo.
    echo HTML 报告: verify_report.html
    start verify_report.html
) else (
    echo.
    echo [跳过] HTML 报告。
)

echo.
echo ============================================
echo   验证全部完成。
echo ============================================
pause
