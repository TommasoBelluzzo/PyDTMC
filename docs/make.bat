@ECHO OFF

PUSHD %~dp0

IF "%SPHINX_BUILD%" == "" (
    SET SPHINX_BUILD=sphinx-build.exe
)

%SPHINX_BUILD% >NUL 2>NUL

IF ERRORLEVEL 9009 (
    ECHO.
    ECHO.Sphinx not found. Install it from:
    ECHO.http://sphinx-doc.org/
    ECHO.
    EXIT /B 1
)

IF "%1" == "" (
    %SPHINX_BUILD% --help
) ELSE (
    SET SPHINX_TYPE=%1
    SET SOURCE_DIR=source
    SET BUILD_DIR=build
    SET SPHINX_OPTS=-nW

    DEL /S /Q %BUILD_DIR%\doctest\* >NUL 2>NUL
    DEL /S /Q %BUILD_DIR%\linkcheck\* >NUL 2>NUL
    DEL /S /Q %BUILD_DIR%\coverage\* >NUL 2>NUL
    DEL /S /Q %BUILD_DIR%\%SPHINX_TYPE%\* >NUL 2>NUL

    ECHO.
    %SPHINX_BUILD% -b doctest %SOURCE_DIR% %BUILD_DIR%\doctest -W
    ECHO.
    %SPHINX_BUILD% -b linkcheck %SOURCE_DIR% %BUILD_DIR%\linkcheck -W
    ECHO.
    %SPHINX_BUILD% -b coverage %SOURCE_DIR% %BUILD_DIR%\coverage -W
    ECHO.
    %SPHINX_BUILD% -b %SPHINX_TYPE% %SOURCE_DIR% %BUILD_DIR%\%SPHINX_TYPE% %SPHINX_OPTS% %O%
    ECHO.
)

POPD