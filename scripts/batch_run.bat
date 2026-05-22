@echo off
chcp 65001 >nul

:: 切换到 CONOP 程序所在目录
set CONOP_DIR=D:\GitHub\geology-big-data-hw\CONOP-run
cd /d "%CONOP_DIR%"

echo ============================================
echo   CONOP 批量参数扫描脚本
echo   工作目录：%CONOP_DIR%
echo   每次跑完后关掉 CONOP 窗口
echo   然后在脚本里按任意键继续
echo ============================================
echo.

:: 确保结果目录存在
mkdir ..\results 2>nul

:: ====== 实验方案 ======

call :run_one baseline        0.980    250    600
call :run_one ratio_099      0.990    250    600
call :run_one ratio_095      0.950    250    600
call :run_one temp_500       0.980    500    600
call :run_one temp_100       0.980    100    600
call :run_one steps_1200     0.980    250   1200
call :run_one steps_0300     0.980    250    300

echo.
echo ===== 全部实验完成！ =====
echo 结果保存在 D:\GitHub\geology-big-data-hw\results\ 下
echo.
echo 别忘了提交到 GitHub：
echo   cd /d D:\GitHub\geology-big-data-hw
echo   git add -A
echo   git commit -m "批量扫描结果"
echo   git push
echo.
pause
exit /b

:run_one
set TAG=%1
set NEW_RATIO=%2
set NEW_TEMP=%3
set NEW_STEPS=%4

echo ----------------------------------------
echo [%TAG%] RATIO=%NEW_RATIO%  STARTEMP=%NEW_TEMP%  STEPS=%NEW_STEPS%
echo.

:: 修改 conop9.cfg
powershell -Command ^
"$cfg = Get-Content conop9.cfg -Raw;" ^
"$cfg = $cfg -replace 'RATIO=[0-9.]+', 'RATIO=%NEW_RATIO%';" ^
"$cfg = $cfg -replace 'STARTEMP=[0-9.]+', 'STARTEMP=%NEW_TEMP%';" ^
"$cfg = $cfg -replace 'STEPS=[0-9]+', 'STEPS=%NEW_STEPS%';" ^
"Set-Content conop9.cfg -Value $cfg"

echo 配置已更新，请双击 CONOP64ver8p621.exe 运行
echo 跑完后关掉窗口，回到这里按任意键继续...
echo.
pause

:: 把结果和 cfg 都保存到 results/ 下
set OUTDIR=..\results\%TAG%
mkdir %OUTDIR% 2>nul

copy /Y trajectory.txt  %OUTDIR%\trajectory.txt  >nul
copy /Y bestsoln.dat    %OUTDIR%\bestsoln.dat    >nul
copy /Y outmain.txt     %OUTDIR%\outmain.txt     >nul
copy /Y runlog.txt      %OUTDIR%\runlog.txt      >nul
copy /Y conop9.cfg      %OUTDIR%\conop9.cfg      >nul
copy /Y soln.dat        %OUTDIR%\soln.dat        >nul
copy /Y ordr.dat        %OUTDIR%\ordr.dat        >nul

echo [%TAG%] 完成 → 已保存至 results\%TAG%\
echo ----------------------------------------
exit /b
