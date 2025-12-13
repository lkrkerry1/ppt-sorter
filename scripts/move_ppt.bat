@echo off
chcp 65001 >nul
title PPT文件整理工具
setlocal enabledelayedexpansion

:: ============================================
:: 批量移动PPT文件并删除空文件夹脚本
:: 功能：将当前目录下所有子文件夹中的PPT文件移动到当前目录
::       并删除移动后变为空的文件夹
:: ============================================

echo.
echo ============================================
echo          PPT文件整理工具
echo ============================================
echo.
echo 脚本功能：
echo 1. 遍历当前目录下所有子文件夹
echo 2. 将所有PPT文件（*.ppt, *.pptx, *.pptm）移动到当前目录
echo 3. 删除所有空文件夹
echo ============================================
echo.

:: 检查当前目录
set "currentDir=%cd%"
echo 当前工作目录：%currentDir%
echo.

:: 询问确认
set /p confirm=是否继续？(Y/N): 
if /i not "!confirm!"=="Y" (
    echo 操作已取消。
    pause
    exit /b
)

echo.
echo 开始扫描PPT文件...
echo.

:: 计数器初始化
set fileCount=0
set folderCount=0
set movedCount=0
set deletedFolderCount=0

:: 步骤1：扫描并统计PPT文件
echo [步骤1] 正在扫描PPT文件...
for /r . %%i in (*.ppt *.pptx *.pptm) do (
    set /a fileCount+=1
    echo 找到文件: %%~nxi
)
echo 共找到 !fileCount! 个PPT文件
echo.

:: 如果没有找到文件，则退出
if !fileCount! equ 0 (
    echo 未找到任何PPT文件，无需操作。
    pause
    exit /b
)

:: 步骤2：移动PPT文件
echo [步骤2] 正在移动PPT文件...
for /r . %%i in (*.ppt *.pptx *.pptm) do (
    if not "%%~dpi" == "%currentDir%\" (
        echo 正在移动：%%~nxi
        move "%%i" "%currentDir%\" >nul
        if !errorlevel! equ 0 (
            set /a movedCount+=1
            echo   成功移动到：%currentDir%\
        ) else (
            echo   移动失败：%%~nxi
        )
    )
)
echo 已完成移动 !movedCount! 个文件
echo.

:: 步骤3：扫描文件夹数量
echo [步骤3] 正在扫描文件夹...
for /f "delims=" %%d in ('dir /ad /b /s 2^>nul') do (
    set /a folderCount+=1
)
echo 共找到 !folderCount! 个文件夹
echo.

:: 步骤4：删除空文件夹（从最深层的文件夹开始）
echo [步骤4] 正在删除空文件夹...
:deleteEmptyFolders
set deletedThisPass=0

:: 使用dir命令获取所有文件夹，按深度排序（最深的前面）
for /f "delims=" %%d in ('dir /ad /b /s ^| sort /r 2^>nul') do (
    dir "%%d" /b 2>nul | findstr . >nul
    if errorlevel 1 (
        echo 删除空文件夹：%%d
        rd "%%d" 2>nul
        if !errorlevel! equ 0 (
            set /a deletedFolderCount+=1
            set /a deletedThisPass+=1
        )
    )
)

:: 如果本轮删除了文件夹，可能需要再次检查
if !deletedThisPass! gtr 0 (
    echo 本轮删除了 !deletedThisPass! 个空文件夹，继续检查...
    goto deleteEmptyFolders
)

:: 步骤5：显示统计结果
echo.
echo ============================================
echo               操作完成！
echo ============================================
echo 统计信息：
echo   扫描到的PPT文件数：!fileCount!
echo   成功移动的文件数：!movedCount!
echo   删除的空文件夹数：!deletedFolderCount!
echo ============================================
echo.

:: 显示剩余文件夹
echo 当前目录结构：
dir /ad /b 2>nul
echo.

pause