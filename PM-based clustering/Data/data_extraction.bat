:: %1: n_samples_per_opmode (Use -1 to load all samples)

set n_samples_per_opmode=%1

:: Create directories if they don't exist yet
if not exist Output\Train mkdir Output\Train
if not exist Output\Test mkdir Output\Test
if not exist Output\Global mkdir Output\Global

:: Clean up previous files
del /F /Q Output\Train\*
del /F /Q Output\Test\*
del /F /Q Output\Global\*

:: Execute script with just the sample size parameter
python data_extraction.py %n_samples_per_opmode%