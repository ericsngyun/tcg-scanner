# PowerShell script to create Windows Task Scheduler task
# Run as Administrator: powershell -ExecutionPolicy Bypass -File setup_scheduled_task.ps1

$TaskName = "TCG-Scanner-PriceUpdate"
$TaskDescription = "Daily price update for TCG Scanner at 2 PM"

# Get the project directory (parent of tools/data_collection)
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $ScriptDir)
$BatchFile = Join-Path $ScriptDir "run_price_update.bat"

Write-Host "Setting up scheduled task: $TaskName"
Write-Host "Project root: $ProjectRoot"
Write-Host "Batch file: $BatchFile"

# Check if task already exists
$existingTask = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue

if ($existingTask) {
    Write-Host "Task already exists. Removing old task..."
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

# Create the action (run the batch file)
$Action = New-ScheduledTaskAction -Execute $BatchFile -WorkingDirectory $ProjectRoot

# Create the trigger (daily at 2:00 PM)
$Trigger = New-ScheduledTaskTrigger -Daily -At "14:00"

# Create settings
$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RunOnlyIfNetworkAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Hours 1)

# Register the task
try {
    Register-ScheduledTask `
        -TaskName $TaskName `
        -Description $TaskDescription `
        -Action $Action `
        -Trigger $Trigger `
        -Settings $Settings `
        -RunLevel Highest

    Write-Host ""
    Write-Host "SUCCESS: Task '$TaskName' created successfully!" -ForegroundColor Green
    Write-Host ""
    Write-Host "The task will run daily at 2:00 PM"
    Write-Host ""
    Write-Host "To verify, run:"
    Write-Host "  Get-ScheduledTask -TaskName '$TaskName'"
    Write-Host ""
    Write-Host "To run immediately for testing:"
    Write-Host "  Start-ScheduledTask -TaskName '$TaskName'"
    Write-Host ""
    Write-Host "To remove the task:"
    Write-Host "  Unregister-ScheduledTask -TaskName '$TaskName'"
}
catch {
    Write-Host "ERROR: Failed to create scheduled task" -ForegroundColor Red
    Write-Host $_.Exception.Message
    Write-Host ""
    Write-Host "Make sure you're running PowerShell as Administrator"
    exit 1
}
