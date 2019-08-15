@ECHO OFF
PUSHD %~dp0

IF "%SPHINXBUILD%" == "" (
	SET SPHINXBUILD=sphinx-build.exe
)

%SPHINXBUILD% >NUL 2>NUL

IF ERRORLEVEL 9009 (
	ECHO.
	ECHO.The 'sphinx-build' command was not found. Make sure you have Sphinx
	ECHO.installed, then set the SPHINXBUILD environment variable to point
	ECHO.to the full path of the 'sphinx-build' executable. Alternatively you
	ECHO.may add the Sphinx directory to PATH.
	ECHO.
	ECHO.If you don't have Sphinx installed, grab it from:
	ECHO.http://sphinx-doc.org/
	EXIT /b 1
)

IF "%1" == "" (
	%SPHINXBUILD% --help
	EXIT /b 1
)

SET SOURCEDIR=_source
SET BUILDDIR=_build

%SPHINXBUILD% -b %1 -c . %SOURCEDIR% %BUILDDIR% %SPHINXOPTS%

POPD