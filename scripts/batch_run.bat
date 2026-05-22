@echo off
chcp 65001 >nul
cd /d "%~dp0CONOP-run"

echo ============================================
echo   CONOP 批量参数扫描脚本
echo   当前时间：%date% %time%
echo   每次跑完后关掉 CONOP 窗口
echo   脚本会自动准备下一次的参数
echo ============================================
echo.

:: ====== 实验方案 ======
:: 每次改一个参数，其他保持不变

:: 方案 0：基线（原始参数）
set RATIO=0.980
set STARTEMP=250
set STEPS=600
set TAG=baseline
call :run_one

:: 方案 1：衰减率更慢
set RATIO=0.990
set STARTEMP=250
set STEPS=600
set TAG=ratio_099
call :run_one

:: 方案 2：衰减率更快
set RATIO=0.950
set STARTEMP=250
set STEPS=600
set TAG=ratio_095
call :run_one

:: 方案 3：更高的初始温度
set RATIO=0.980
set STARTEMP=500
set STEPS=600
set TAG=temp_500
call :run_one

:: 方案 4：更低的初始温度
set RATIO=0.980
set STARTEMP=100
set STEPS=600
set TAG=temp_100
call :run_one

:: 方案 5：更多步数
set RATIO=0.980
set STARTEMP=250
set STEPS=1200
set TAG=steps_1200
call :run_one

:: 方案 6：更少步数
set RATIO=0.980
set STARTEMP=250
set STEPS=300
set TAG=steps_0300
call :run_one

echo.
echo ===== 全部实验完成！ =====
echo 结果保存在 results/ 文件夹下
pause
exit /b

:run_one
echo ----------------------------------------
echo 准备运行：%TAG%
echo   RATIO=%RATIO%  STARTEMP=%STARTEMP%  STEPS=%STEPS%
echo.

:: 确保结果目录存在
mkdir ..\results\%TAG% 2>nul

:: 备份当前的输出文件
copy trajectory.txt ..\results\%TAG%\trajectory.txt.bak >nul 2>nul
copy bestsoln.dat ..\results\%TAG%\bestsoln.dat.bak >nul 2>nul
copy outmain.txt ..\results\%TAG%\outmain.txt.bak >nul 2>nul
copy runlog.txt ..\results\%TAG%\runlog.txt.bak >nul 2>nul

:: 修改 conop9.cfg
powershell -Command ^
"$cfg = Get-Content conop9.cfg -Raw;" ^
"$cfg = $cfg -replace 'RATIO=[0-9.]+', 'RATIO=%RATIO%';" ^
"$cfg = $cfg -replace 'STARTEMP=[0-9.]+', 'STARTEMP=%STARTEMP%';" ^
"$cfg = $cfg -replace 'STEPS=[0-9]+', 'STEPS=%STEPS%';" ^
"Set-Content conop9.cfg -Value $cfg"

echo 配置已更新
echo.
echo 请双击 CONOP64ver8p621.exe 运行
echo 跑完后按任意键继续下一个方案...
echo.
pause

:: 运行结束后，把结果移到对应文件夹
move trajectory.txt ..\results\%TAG%\trajectory.txt >nul 2>nul
move bestsoln.dat ..\results\%TAG%\bestsoln.dat >nul 2>nul
move outmain.txt ..\results\%TAG%\outmain.txt >nul 2>nul
move runlog.txt ..\results\%TAG%\runlog.txt >nul 2>nul
copy conop9.cfg ..\results\%TAG%\conop9.cfg >nul

echo %TAG% 完成！
echo ----------------------------------------
exit /b
