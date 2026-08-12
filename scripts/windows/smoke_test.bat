@echo off
setlocal EnableDelayedExpansion

rem Smoke test: run training_example.py once against every YAML config under configs/,
rem cut down to a few iterations so the whole sweep finishes quickly. A config whose
rem framework is not installed (mjlab, isaaclab) exits with code 2 and is reported as
rem skipped rather than failed, matching examples/run_all_examples.py.
rem
rem Isaac Sim's own shutdown code can force the process to exit 0 even after an unhandled
rem Python exception, so each run's output is also captured to OUTDIR and scanned for a
rem traceback, which overrides the reported exit code to a failure.
rem
rem Each run also has a wall-clock timeout. Isaac Sim's GPU renderer can, on rare occasions,
rem deadlock instead of erroring out - for example when two Kit/Isaac Sim processes contend
rem for the same GPU at once - and a hang like that would otherwise block the whole sweep
rem indefinitely with no visible output, since output is buffered to a log file until the
rem process exits. Batch alone can't kill a whole process tree on timeout, so this file
rem writes a small PowerShell helper to a temp file at runtime and shells out to it per
rem config run - keeping this .bat the only tracked file the smoke test needs.
rem
rem Usage:
rem     scripts\windows\smoke_test.bat [gymnasium|isaaclab|mjlab|all] [timeout-minutes]
rem
rem With no argument, or "all", every framework under configs/ is swept. Timeout defaults to 20
rem minutes per run, generous enough for a cold Isaac Sim shader-cache build.

set "ROOT=%~dp0..\.."
set "SCRIPT=%ROOT%\examples\training_example.py"
set "LOGDIR=%TEMP%\rlbotics_smoke_test"
set "OUTDIR=%TEMP%\rlbotics_smoke_test_output"
set "RUNNER_PS1=%TEMP%\rlbotics_smoke_run.ps1"
set "ITERATIONS=3"
set "SKIPPED_CODE=2"
set "TIMEOUT_CODE=124"

rem --- Generate the timeout/kill helper -------------------------------------------------
rem Start-Process + WaitForExit lets us poll for completion, print a heartbeat, and on
rem timeout kill the whole process tree via taskkill /T (Stop-Process alone would leave
rem Isaac Sim's own child processes, e.g. Kit, running and still holding the GPU).
> "%RUNNER_PS1%" echo param(
>>"%RUNNER_PS1%" echo     [string]$ScriptPath,
>>"%RUNNER_PS1%" echo     [string]$ConfigPath,
>>"%RUNNER_PS1%" echo     [string]$OutFile,
>>"%RUNNER_PS1%" echo     [int]$TimeoutSeconds,
>>"%RUNNER_PS1%" echo     [int]$Iterations,
>>"%RUNNER_PS1%" echo     [string]$LogDir,
>>"%RUNNER_PS1%" echo     [int]$NumEnvs = 0
>>"%RUNNER_PS1%" echo )
>>"%RUNNER_PS1%" echo $ErrFile = $OutFile + '.stderr'
>>"%RUNNER_PS1%" echo $pyArgs = @($ScriptPath, $ConfigPath, '-i', $Iterations, '--log-dir', $LogDir)
>>"%RUNNER_PS1%" echo if ($NumEnvs -gt 0) { $pyArgs += @('-n', $NumEnvs) }
>>"%RUNNER_PS1%" echo $proc = Start-Process -FilePath python -ArgumentList $pyArgs -NoNewWindow -PassThru -RedirectStandardOutput $OutFile -RedirectStandardError $ErrFile
>>"%RUNNER_PS1%" echo $timedOut = $false
>>"%RUNNER_PS1%" echo $elapsed = 0
>>"%RUNNER_PS1%" echo $heartbeat = 30
>>"%RUNNER_PS1%" echo while (-not $proc.WaitForExit($heartbeat * 1000)) {
>>"%RUNNER_PS1%" echo     $elapsed += $heartbeat
>>"%RUNNER_PS1%" echo     Write-Host ('   ... still running (' + $elapsed + 's elapsed, log: ' + $OutFile + ')')
>>"%RUNNER_PS1%" echo     if ($elapsed -ge $TimeoutSeconds) {
>>"%RUNNER_PS1%" echo         Write-Host ('   ... timed out after ' + $TimeoutSeconds + 's, killing PID ' + $proc.Id + ' and its children')
>>"%RUNNER_PS1%" echo         ^& taskkill /PID $proc.Id /T /F 2^>^&1 ^| Out-Null
>>"%RUNNER_PS1%" echo         $timedOut = $true
>>"%RUNNER_PS1%" echo         break
>>"%RUNNER_PS1%" echo     }
>>"%RUNNER_PS1%" echo }
>>"%RUNNER_PS1%" echo if (Test-Path $ErrFile) {
>>"%RUNNER_PS1%" echo     Get-Content $ErrFile -ErrorAction SilentlyContinue ^| Add-Content $OutFile
>>"%RUNNER_PS1%" echo     Remove-Item $ErrFile -ErrorAction SilentlyContinue
>>"%RUNNER_PS1%" echo }
>>"%RUNNER_PS1%" echo if ($timedOut) { exit 124 }
>>"%RUNNER_PS1%" echo exit $proc.ExitCode

set "FRAMEWORK=%~1"
if "%FRAMEWORK%"=="" set "FRAMEWORK=all"

if /I "%FRAMEWORK%"=="all" (
    set "CONFIGS_DIR=%ROOT%\configs"
) else if /I "%FRAMEWORK%"=="gymnasium" (
    set "CONFIGS_DIR=%ROOT%\configs\gymnasium"
) else if /I "%FRAMEWORK%"=="isaaclab" (
    set "CONFIGS_DIR=%ROOT%\configs\isaaclab"
) else if /I "%FRAMEWORK%"=="mjlab" (
    set "CONFIGS_DIR=%ROOT%\configs\mjlab"
) else (
    echo Unknown framework '%FRAMEWORK%'. Expected gymnasium, isaaclab, mjlab, or all.
    del "%RUNNER_PS1%" >nul 2>&1
    exit /b 1
)

set "TIMEOUT_MIN=%~2"
if "%TIMEOUT_MIN%"=="" set "TIMEOUT_MIN=20"
set /a "TIMEOUT_SEC=%TIMEOUT_MIN%*60"

rem Python falls back to the system codepage (cp1252 on most Windows installs) instead of
rem UTF-8 once stdout is redirected to a file rather than a real console, which crashes on
rem any library that prints a Unicode character (torch.onnx's exporter logs a checkmark).
set "PYTHONIOENCODING=utf-8"

rem The console's own codepage is a separate thing from Python's: even with UTF-8 bytes
rem written correctly to the log file above, `type`-ing them back out through a CP437/1252
rem console renders mojibake (a checkmark becomes "Γ£à"). Switch the console to UTF-8 for
rem the run and restore whatever it was before on exit.
for /f "tokens=2 delims=:" %%c in ('chcp') do set "ORIG_CHCP=%%c"
chcp 65001 >nul

if not exist "%OUTDIR%" mkdir "%OUTDIR%"

set /a PASS_COUNT=0
set /a FAIL_COUNT=0
set /a SKIP_COUNT=0
set "FAILED_LIST="

pushd "%ROOT%"

echo Framework: %FRAMEWORK%  (%CONFIGS_DIR%)
echo.

for /R "%CONFIGS_DIR%" %%F in (*.yaml) do (
    set "NAME=%%~nxF"
    set "OUT=%OUTDIR%\!NAME!.log"

    rem Gymnasium configs already use smoke-test-sized env counts (8-32), but every isaaclab
    rem config uses real training scale (2048-8192 parallel envs) and mjlab similarly
    rem (512-4096) - building that many envs from a cold shader/kernel cache before even the
    rem first iteration runs can take minutes on its own. Cut it down for a quick smoke run.
    set "NUM_ENVS=0"
    echo %%F | findstr /C:"\isaaclab\\" >nul 2>&1 && set "NUM_ENVS=64"
    echo %%F | findstr /C:"\mjlab\\" >nul 2>&1 && set "NUM_ENVS=8"

    echo ============================================================
    echo Running !NAME!  ^(timeout !TIMEOUT_MIN!m^)
    echo ============================================================
    powershell -NoProfile -ExecutionPolicy Bypass -File "%RUNNER_PS1%" ^
        -ScriptPath "%SCRIPT%" -ConfigPath "%%F" -OutFile "!OUT!" -TimeoutSeconds %TIMEOUT_SEC% ^
        -Iterations %ITERATIONS% -LogDir "%LOGDIR%" -NumEnvs !NUM_ENVS!
    set "RC=!ERRORLEVEL!"
    type "!OUT!"

    rem Isaac Sim's teardown can force exit code 0 after an unhandled exception, so a
    rem traceback in the output overrides whatever exit code was reported.
    findstr /C:"Traceback (most recent call last):" "!OUT!" >nul 2>&1
    if "!ERRORLEVEL!"=="0" if not "!RC!"=="%SKIPPED_CODE%" set "RC=1"

    if "!RC!"=="0" (
        echo PASS  !NAME!
        set /a PASS_COUNT+=1
    ) else if "!RC!"=="%SKIPPED_CODE%" (
        echo SKIP  !NAME!  ^(optional dependency missing^)
        set /a SKIP_COUNT+=1
    ) else if "!RC!"=="%TIMEOUT_CODE%" (
        echo FAIL  !NAME!  ^(timed out after !TIMEOUT_MIN!m - killed^)
        set /a FAIL_COUNT+=1
        set "FAILED_LIST=!FAILED_LIST! !NAME!"
    ) else (
        echo FAIL  !NAME!  ^(exit code !RC!^)
        set /a FAIL_COUNT+=1
        set "FAILED_LIST=!FAILED_LIST! !NAME!"
    )
    echo.
)

popd

del "%RUNNER_PS1%" >nul 2>&1

echo ============================================================
echo Summary: !PASS_COUNT! passed, !SKIP_COUNT! skipped, !FAIL_COUNT! failed
echo Per-config logs: %OUTDIR%

chcp %ORIG_CHCP% >nul

if not "!FAIL_COUNT!"=="0" (
    echo Failed configs: !FAILED_LIST!
    endlocal
    exit /b 1
)

endlocal
exit /b 0
