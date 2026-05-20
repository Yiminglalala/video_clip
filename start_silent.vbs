Dim shell
Set shell = CreateObject("WScript.Shell")

shell.Run "powershell -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File ""D:\video_clip\tools\start_service.ps1"" -Port 8501 -OpenBrowser", 0, False

Set shell = Nothing
