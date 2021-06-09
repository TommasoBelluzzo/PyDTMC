@ECHO OFF

PUSHD %~dp0

IF "%SPHINX_BUILD%" == "" (
	SET SPHINX_BUILD=sphinx-build.exe
)

%SPHINX_BUILD% >NUL 2>NUL

IF ERRORLEVEL 9009 (
	ECHO.
	ECHO.Install Sphinx from:
	ECHO.http://sphinx-doc.org/
	EXIT /b 1
)

IF "%1" == "" (
	SET SPHINX_TYPE=html
) ELSE (
	SET SPHINX_TYPE=%1
)

SET SOURCE_DIR=_source
SET BUILD_DIR=_build

DEL /S /Q %BUILD_DIR%\%SPHINX_TYPE%\* >NUL 2>NUL
%SPHINX_BUILD% -nW -b %SPHINX_TYPE% -c . %SOURCE_DIR% %BUILD_DIR%\%SPHINX_TYPE%
%SPHINX_BUILD% -nW -b coverage -c . %SOURCE_DIR% %BUILD_DIR%\%SPHINX_TYPE%

POPD