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
SET SOURCE_DIR=source
SET BUILD_DIR=build

IF "%CI%" == "YES" (

    IF NOT "%SPHINX_TYPE%" == "" (
        EXIT /B 1
    )

    SET SPHINX_OPTS=-n

    DEL /S /Q %BUILD_DIR%\doctest\* >NUL 2>NUL
    DEL /S /Q %BUILD_DIR%\linkcheck\* >NUL 2>NUL
    DEL /S /Q %BUILD_DIR%\coverage\* >NUL 2>NUL

    ECHO.
    %SPHINX_BUILD% -b doctest %SOURCE_DIR% %BUILD_DIR%\doctest %SPHINX_OPTS% %O%
    ECHO.
    %SPHINX_BUILD% -b linkcheck %SOURCE_DIR% %BUILD_DIR%\linkcheck %SPHINX_OPTS% %O%
    ECHO.
    %SPHINX_BUILD% -b coverage %SOURCE_DIR% %BUILD_DIR%\coverage %SPHINX_OPTS% %O%
    ECHO.

) ELSE (

    IF "%SPHINX_TYPE%" == "" (
        %SPHINX_BUILD% --help
        EXIT /B 1
    )

    SET SPHINX_OPTS=-an

    DEL /S /Q %BUILD_DIR%\%SPHINX_TYPE%\* >NUL 2>NUL

    ECHO.
    %SPHINX_BUILD% -b %SPHINX_TYPE% %SOURCE_DIR% %BUILD_DIR%\%SPHINX_TYPE% %SPHINX_OPTS% %O%
    ECHO.

)

POPD