Set fso = CreateObject("Scripting.FileSystemObject")
Set sh = CreateObject("WScript.Shell")
here = fso.GetParentFolderName(WScript.ScriptFullName)
root = fso.GetParentFolderName(here)
sh.CurrentDirectory = root
py = root & "\.venv\Scripts\pythonw.exe"
If fso.FileExists(py) Then
  sh.Run """" & py & """ -m signal_desk --no-browser --until 15:40", 0, False
Else
  sh.Run "uv run pythonw -m signal_desk --no-browser --until 15:40", 0, False
End If
