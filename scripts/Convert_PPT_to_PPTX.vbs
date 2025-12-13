Option Explicit

Dim fso, ppt, pres, folder, file
Dim sourcePath, newPath, count

' 获取当前文件夹
Set fso = CreateObject("Scripting.FileSystemObject")
sourcePath = fso.GetParentFolderName(WScript.ScriptFullName)

' 启动PowerPoint（可见但最小化）
Set ppt = CreateObject("PowerPoint.Application")
ppt.Visible = True
ppt.WindowState = 2

count = 0
Set folder = fso.GetFolder(sourcePath)

' 调用递归函数开始转换
Call ProcessFolder(folder)

' 递归函数：处理文件夹和子文件夹
Sub ProcessFolder(currentFolder)
    Dim subFolder, f
    
    ' 处理当前文件夹内文件
    For Each f In currentFolder.Files
        If LCase(fso.GetExtensionName(f.Name)) = "ppt" Then
            newPath = fso.BuildPath(fso.GetParentFolderName(f.Path), fso.GetBaseName(f.Path) & ".pptx")
            
            If Not fso.FileExists(newPath) Then
                On Error Resume Next
                Set pres = ppt.Presentations.Open(f.Path, True, False, False)
                If Err.Number = 0 Then
                    pres.SaveAs newPath, 24
                    pres.Close
                    count = count + 1
                    WScript.Echo "[成功] 转换: " & f.Name
                End If
                On Error Goto 0
            End If
        End If
    Next
    
    ' 递归处理子文件夹
    For Each subFolder In currentFolder.SubFolders
        ProcessFolder subFolder
    Next
End Sub

' 清理并显示结果
ppt.Quit
WScript.Echo "========== 完成 =========="
WScript.Echo "成功转换文件数: " & count
WScript.Sleep 3000