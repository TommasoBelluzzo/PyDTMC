@ECHO OFF

PUSHD %~dp0

IF "%SPHINX_BUILD%" == "" (
    SET SPHINX_BUILD=sphinx-build.exe
)

%SPHINX_BUILD% >NUL 2>NUL

IF ERRORLEVEL 9009 (
    ECHO.
    ECHO.Sphinx not found, install it from:
    ECHO.http://sphinx-doc.org/
    ECHO.

    EXIT /B 1
)

SET SPHINX_TYPE=%1

IF "%SPHINX_TYPE%" == "" (
    ECHO.
    %SPHINX_BUILD% --help
    ECHO.

    EXIT /B 1
)

SET SOURCE_DIR=source
SET BUILD_DIR=build
FOR /F "TOKENS=1,* DELIMS= " %%A IN ("%*") DO SET SPHINX_OPTS=%%B

DEL /S /Q %BUILD_DIR%\%SPHINX_TYPE%\* >NUL 2>NUL

ECHO.
%SPHINX_BUILD% -b %SPHINX_TYPE% %SPHINX_OPTS% %SOURCE_DIR% %BUILD_DIR%\%SPHINX_TYPE%
ECHO.

POPD