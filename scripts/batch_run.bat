@echo off
chcp 65001 >nul

set CONOP_DIR=D:\GitHub\geology-big-data-hw\CONOP-run
set RESULTS_DIR=D:\GitHub\geology-big-data-hw\results
set REPEAT=3

cd /d "%CONOP_DIR%"

echo ============================================
echo   CONOP 批量参数扫描脚本（每组 %REPEAT% 次）
echo   工作目录：%CONOP_DIR%
echo   结果保存：%RESULTS_DIR%
echo ============================================
echo.

if not exist "%RESULTS_DIR%" mkdir "%RESULTS_DIR%"

call :run_one baseline      0.980  250  600
call :run_one ratio_099    0.990  250  600
call :run_one ratio_095    0.950  250  600
call :run_one temp_500     0.980  500  600
call :run_one temp_100     0.980  100  600
call :run_one steps_1200   0.980  250 1200
call :run_one steps_0300   0.980  250  300

echo.
echo ===== 全部完成！ =====
echo 结果保存在 %RESULTS_DIR%
echo.
echo 接下来提交到 GitHub：
echo   cd /d D:\GitHub\geology-big-data-hw
echo   git add -A
echo   git commit -m "批量参数扫描结果（每组重复3次）"
echo   git push
echo.
pause
exit /b

:run_one
set TAG=%1
set NEW_RATIO=%2
set NEW_TEMP=%3
set NEW_STEPS=%4

for /l %%i in (1,1,%REPEAT%) do (
    echo ----------------------------------------
echo [%TAG%] 第 %%i 次: RATIO=%NEW_RATIO%  STARTEMP=%NEW_TEMP%  STEPS=%NEW_STEPS%
echo.

    :: 修改配置文件
    powershell -Command "(Get-Content conop9.cfg) -replace 'RATIO=[0-9.]+', 'RATIO=%NEW_RATIO%' | Set-Content conop9.cfg" 2>nul
    powershell -Command "(Get-Content conop9.cfg) -replace 'STARTEMP=[0-9.]+', 'STARTEMP=%NEW_TEMP%' | Set-Content conop9.cfg" 2>nul
    powershell -Command "(Get-Content conop9.cfg) -replace 'STEPS=[0-9]+', 'STEPS=%NEW_STEPS%' | Set-Content conop9.cfg" 2>nul

    echo 配置已更新，请双击 CONOP64ver8p621.exe 运行
echo 跑完后关掉窗口，按任意键继续...
echo.
    pause

    :: 保存结果到独立文件夹
    setlocal
    set RUN_DIR=%RESULTS_DIR%\%TAG%\run_%%i
    mkdir %RUN_DIR% 2>nul

    copy /Y trajectory.txt  %RUN_DIR%\trajectory.txt  >nul
    copy /Y bestsoln.dat    %RUN_DIR%\bestsoln.dat    >nul
    copy /Y outmain.txt     %RUN_DIR%\outmain.txt     >nul
    copy /Y runlog.txt      %RUN_DIR%\runlog.txt      >nul
    copy /Y conop9.cfg      %RUN_DIR%\conop9.cfg      >nul
    copy /Y soln.dat        %RUN_DIR%\soln.dat        >nul
    copy /Y ordr.dat        %RUN_DIR%\ordr.dat        >nul

    if exist %RUN_DIR%\trajectory.txt (
        echo [%TAG%] 第 %%i 次 - 保存成功
    ) else (
        echo [%TAG%] 第 %%i 次 - 保存失败！检查文件是否存在
        dir trajectory.txt 2>nul
    )
    endlocal
)

echo ----------------------------------------
exit /b
