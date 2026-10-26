:: %1: n_clusters, %2: clustering_technique, %3: process_discovery_variant

:: Options:
:: n_clusters = <integer>
:: clustering_technique = [kmeans, gmm]
:: process_discovery_variant = [im, imf25, imf50, imf75, imf99, ilp, ilp25, ilp50, ilp75, ilp99, alpha]

set n_clusters=%1
set clustering_technique=%2
set process_discovery_variant=%3

:: 1. Ensure directories exist to prevent "Path not found" errors
if not exist Input\Global mkdir Input\Global
if not exist Input\Train mkdir Input\Train
if not exist Output\PetriNets\GlobalNode mkdir Output\PetriNets\GlobalNode
if not exist Output\PetriNets\LocalNodes mkdir Output\PetriNets\LocalNodes
if not exist Output\SimilarityMatrix mkdir Output\SimilarityMatrix

:: 2. Clean up previous files using the NEW directory names
del /F /Q Input\Global\*
del /F /Q Input\Train\*
del /F /Q Output\PetriNets\GlobalNode\*
del /F /Q Output\PetriNets\LocalNodes\*
del /F /Q Output\SimilarityMatrix\*

:: 3. Copy new data from the extraction step 
:: (Assuming your 'Data\Messina' output folders were also updated to the new naming convention)
copy Data\Messina\Output\Global\* Input\Global
copy Data\Messina\Output\Train\* Input\Train

:: 4. Run the python script
python extract_similarity_matrix.py %n_clusters% %clustering_technique% %process_discovery_variant%