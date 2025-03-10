@echo off
REM Activate the virtual environment located in Depth-Anything-V2\env
call Depth-Anything-V2\env\Scripts\activate
if errorlevel 1 goto error

REM Run the ChromoStereoizer.py script from the current folder
python ChromoStereoizer.py
if errorlevel 1 goto error

goto end

:error
echo An error occurred while running ChromoStereoizer.py.
pause

:end
