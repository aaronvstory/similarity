@echo off
setlocal EnableExtensions

set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%" >nul

set "PYTHON_CMD="

python -c "import cv2, numpy" >nul 2>nul
if not errorlevel 1 set "PYTHON_CMD=python"

if not defined PYTHON_CMD (
  py -3 -c "import cv2, numpy" >nul 2>nul
  if not errorlevel 1 set "PYTHON_CMD=py -3"
)

if not defined PYTHON_CMD (
  echo Could not find a Python interpreter with both cv2 and numpy installed.
  echo Install the dependencies for your active Python and try again.
  goto DONE
)

rem Optional tuning flags. Edit this line if you want different defaults.
set "EXTRA_ARGS="
if defined EXTRACT_FACE_EXTRA_ARGS set "EXTRA_ARGS=%EXTRACT_FACE_EXTRA_ARGS%"

if "%~1"=="" goto PICK_FILES
goto PROCESS_ARGS

:PICK_FILES
echo Select one or more image files to process with extract_face.
set "SELECTION_FILE=%TEMP%\extract_face_selection_%RANDOM%%RANDOM%.txt"

powershell -NoProfile -STA -Command ^
  "Add-Type -AssemblyName System.Windows.Forms; " ^
  "$dialog = New-Object System.Windows.Forms.OpenFileDialog; " ^
  "$dialog.Multiselect = $true; " ^
  "$dialog.Filter = 'Image Files|*.jpg;*.jpeg;*.png;*.bmp;*.webp|All Files|*.*'; " ^
  "if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) { $dialog.FileNames | Set-Content -Path '%SELECTION_FILE%' }"

if not exist "%SELECTION_FILE%" (
  echo No files selected.
  goto DONE
)

for /f "usebackq delims=" %%F in ("%SELECTION_FILE%") do call :PROCESS_ONE "%%F"
del "%SELECTION_FILE%" >nul 2>nul
goto DONE

:PROCESS_ARGS
if "%~1"=="" goto DONE
call :PROCESS_ONE "%~1"
shift
goto PROCESS_ARGS

:PROCESS_ONE
echo(
echo ===============================================================
echo Processing: %~1
call %PYTHON_CMD% "%SCRIPT_DIR%extract_face.py" "%~1" %EXTRA_ARGS%
if errorlevel 1 (
  echo Failed: %~1
) else (
  echo Finished: %~1
)
exit /b 0

:DONE
echo(
echo All done.
if not defined EXTRACT_FACE_NO_PAUSE pause
popd >nul
endlocal
