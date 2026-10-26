setlocal enabledelayedexpansion

echo ==============================================================
echo   Process Mining-Enabled Federated Anomaly Detection Pipeline
echo ==============================================================
echo.

:: Ensure the root Results folder exists
if not exist "Results" mkdir "Results"

:: ==========================================
:: Step 1: Create the Dataset
:: ==========================================
echo [1/4] Running Data Extraction...
cd "PM-based clustering/Data"
:: Ensure this matches the exact name of your extraction script
python data_extraction.py 1000
cd ..

echo Routing generated datasets to input folders...
:: 1a. Route to PM-based clustering Inputs
xcopy /Y /S /I "Data\Output\Global\*.*" "Input\Global"
xcopy /Y /S /I "Data\Output\Train\*.*" "Input\Train"

:: 1b. Route to Federated Anomaly Detection Inputs
xcopy /Y /S /I "Data\Output\Global\*.*" "..\Federated anomaly detection\Input\Data\Global"
xcopy /Y /S /I "Data\Output\Train\*.*" "..\Federated anomaly detection\Input\Data\Train"
xcopy /Y /S /I "Data\Output\Test\*.*"  "..\Federated anomaly detection\Input\Data\Test"
echo.

:: ==========================================
:: Step 2: Perform PM-Based Clustering
:: ==========================================
echo [2/4] Running Process Mining and Similarity Extraction...
:: Configure your arguments here: <n_clusters> <clustering_technique> <process_discovery_variant>
set N_CLUSTERS=3
set CLUSTER_TECH=kmeans
set DISCOVERY_VAR=im

python extract_similarity_matrix.py %N_CLUSTERS% %CLUSTER_TECH% %DISCOVERY_VAR%

echo Routing Similarity Matrix to Federated learning folder...
xcopy /Y "Output\SimilarityMatrix\SimilarityMatrix.csv" "..\Federated anomaly detection\Input"
cd ..
echo.

:: ==========================================
:: Step 3: Perform Federated Anomaly Detection
:: ==========================================
echo [3/4] Running Federated Anomaly Detection...
cd "Federated anomaly detection"
python federated_anomaly_detection.py
cd ..
echo.

:: ==========================================
:: Step 4: Collect Results
:: ==========================================
echo [4/4] Aggregating Results...
:: Copy all contents from the Federated output folder to the root Results folder
xcopy /Y /S /I "Federated anomaly detection\Output\*.*" "Results"

echo.
echo ==============================================================
echo Pipeline execution complete! All outputs are saved in 'Results'.
echo ==============================================================
pause