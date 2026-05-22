@echo off
chcp 65001 >nul

set CONOP_DIR=D:\GitHub\geology-big-data-hw\CONOP-run
cd /d "%CONOP_DIR%"

echo ============================================
echo   CONOP 批量参数扫描脚本（重复 3 次）
echo   工作目录：%CONOP_DIR%
echo   每组参数跑 3 次，观察解的稳定性
echo ============================================
echo.

mkdir ..\results 2>nul

:: 每个实验重复 3 次
set REPEAT=3

call :run_one baseline        0.980    250    600
call :run_one ratio_099      0.990    250    600
call :run_one ratio_095      0.950    250    600
call :run_one temp_500       0.980    500    600
call :run_one temp_100       0.980    100    600
call :run_one steps_1200     0.980    250   1200
call :run_one steps_0300     0.980    250    300

echo.
echo ===== 全部完成！共 %REPEAT% x 7 = %COUNT% 次运行 =====
echo 结果保存在 D:\GitHub\geology-big-data-hw\results\ 下
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

setlocal enabledelayedexpansion

for /l %%i in (1,1,%REPEAT%) do (
    echo ----------------------------------------
    echo [%TAG%] 第 %%i / %REPEAT% 次: RATIO=%NEW_RATIO%  STARTEMP=%NEW_TEMP%  STEPS=%NEW_STEPS%
    echo.

    :: 修改 RATIO
    powershell -Command "(Get-Content conop9.cfg) -replace 'RATIO=[0-9.]+', 'RATIO=%NEW_RATIO%' | Set-Content conop9.cfg" 2>nul
    :: 修改 STARTEMP
    powershell -Command "(Get-Content conop9.cfg) -replace 'STARTEMP=[0-9.]+', 'STARTEMP=%NEW_TEMP%' | Set-Content conop9.cfg" 2>nul
    :: 修改 STEPS
    powershell -Command "(Get-Content conop9.cfg) -replace 'STEPS=[0-9]+', 'STEPS=%NEW_STEPS%' | Set-Content conop9.cfg" 2>nul

    echo 配置已更新，请双击 CONOP64ver8p621.exe 运行
    echo 跑完后关掉窗口，按任意键继续...
    echo.
    pause

    :: 保存结果，用 run_%%i 区分重复次数
    set OUTDIR=..\results\%TAG%\run_%%i
    mkdir !OUTDIR! 2>nul

    copy /Y trajectory.txt  !OUTDIR!\trajectory.txt  >nul
    copy /Y bestsoln.dat    !OUTDIR!\bestsoln.dat    >nul
    copy /Y outmain.txt     !OUTDIR!\outmain.txt     >nul
    copy /Y runlog.txt      !OUTDIR!\runlog.txt      >nul
    copy /Y conop9.cfg      !OUTDIR!\conop9.cfg      >nul
    copy /Y soln.dat        !OUTDIR!\soln.dat        >nul
    copy /Y ordr.dat        !OUTDIR!\ordr.dat        >nul

    set /a COUNT=COUNT+1
    echo [%TAG%] 第 %%i 次完成
)

endlocal
echo ----------------------------------------
exit /b
