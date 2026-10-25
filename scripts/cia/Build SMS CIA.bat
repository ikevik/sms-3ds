@echo off
rem Windows: double-click to build the HOME Menu .cia from your Super Mario Sunshine disc image.
cd /d "%~dp0"
where py >nul 2>nul
if not errorlevel 1 (
    py -3 -I build_cia.py %*
    goto done
)
where python >nul 2>nul
if not errorlevel 1 (
    python -I build_cia.py %*
    goto done
)
echo Python 3 is needed: https://www.python.org/downloads/
echo (when installing it, tick "Add python.exe to PATH")
:done
echo.
pause
