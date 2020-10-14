#!/bin/bash
path=$1
iterations=$2

green=`tput setaf 2`
reset=`tput sgr0`

rm -rf "/home/gpurepair/tools/results"

mkdir "/home/gpurepair/tools/results"
mkdir "/home/gpurepair/tools/results/gpuverify"
mkdir "/home/gpurepair/tools/results/autosync"
mkdir "/home/gpurepair/tools/results/autosync/gpuverify"
mkdir "/home/gpurepair/tools/results/gpurepair"
mkdir "/home/gpurepair/tools/results/gpurepair_maxsat"
mkdir "/home/gpurepair/tools/results/gpurepair_grid"
mkdir "/home/gpurepair/tools/results/gpurepair_inspection"
mkdir "/home/gpurepair/tools/results/gpurepair_grid_inspection"

path="./tests/${path}"
solution_path="${path}/../solutions"

for ((i = 1; i <= iterations; i++)); do
	clear
	echo "${green}Running GPUVerify - Iteration ${i}${reset}"

	cd "/home/gpurepair/tools/gpuverify"
	rm -rf "./tests"
	cp -r "./tests-bkp" "./tests"

	python3 ./gvtester.py ${path} --threads=1 --time-as-csv --csv-file=gpuverify-$i.csv
	mv "gpuverify-$i.csv" "/home/gpurepair/tools/results/gpuverify/gpuverify-$i.csv"

	clear
	echo "${green}Running AutoSync - Iteration ${i}${reset}"

	cd "/home/gpurepair/tools/autosync"
	rm -rf "./tests"
	cp -r "./tests-bkp" "./tests"

	python3 ./testrunner.py ${path} | tee autosync-$i.csv
	mv "autosync-$i.csv" "/home/gpurepair/tools/results/autosync/autosync-$i.csv"

	clear
	echo "${green}Verifying AutoSync solutions for false positives - Iteration ${i}${reset}"

	python3 ../gpuverify/gvtester.py ${solution_path} --threads=1 --time-as-csv --csv-file=gpuverify-$i.csv
	mv "gpuverify-$i.csv" "/home/gpurepair/tools/results/autosync/gpuverify/gpuverify-$i.csv"
	rm -rf ${solution_path}

	clear
	echo "${green}Running GPURepair (default) - Iteration ${i}${reset}"

	cd "/home/gpurepair/tools/gpurepair"
	rm -rf "./tests"
	cp -r "./tests-bkp" "./tests"

	python3 ./grtester.py ${path} --threads=1 --time-as-csv --csv-file=gpurepair-$i.csv --gropt=--detailed-logging
	mv "gpurepair-$i.csv" "/home/gpurepair/tools/results/gpurepair/gpurepair-$i.csv"
	mv "gpurepair-$i.metrics.csv" "/home/gpurepair/tools/results/gpurepair/gpurepair-$i.metrics.csv"
	
	clear
	echo "${green}Running GPURepair (--maxsat) - Iteration ${i}${reset}"

	cd "/home/gpurepair/tools/gpurepair"
	rm -rf "./tests"
	cp -r "./tests-bkp" "./tests"

	python3 ./grtester.py ${path} --threads=1 --time-as-csv --csv-file=gpurepair-$i.csv --gropt=--detailed-logging --gropt=--maxsat
	mv "gpurepair-$i.csv" "/home/gpurepair/tools/results/gpurepair_maxsat/gpurepair-$i.csv"
	mv "gpurepair-$i.metrics.csv" "/home/gpurepair/tools/results/gpurepair_maxsat/gpurepair-$i.metrics.csv"
	
	clear
	echo "${green}Running GPURepair (--disable-grid) - Iteration ${i}${reset}"

	cd "/home/gpurepair/tools/gpurepair"
	rm -rf "./tests"
	cp -r "./tests-bkp" "./tests"

	python3 ./grtester.py ${path} --threads=1 --time-as-csv --csv-file=gpurepair-$i.csv --gropt=--detailed-logging --gropt=--disable-grid
	mv "gpurepair-$i.csv" "/home/gpurepair/tools/results/gpurepair_grid/gpurepair-$i.csv"
	mv "gpurepair-$i.metrics.csv" "/home/gpurepair/tools/results/gpurepair_grid/gpurepair-$i.metrics.csv"
	
	clear
	echo "${green}Running GPURepair (--disable-inspect) - Iteration ${i}${reset}"

	cd "/home/gpurepair/tools/gpurepair"
	rm -rf "./tests"
	cp -r "./tests-bkp" "./tests"

	python3 ./grtester.py ${path} --threads=1 --time-as-csv --csv-file=gpurepair-$i.csv --gropt=--detailed-logging --gropt=--disable-inspect
	mv "gpurepair-$i.csv" "/home/gpurepair/tools/results/gpurepair_inspection/gpurepair-$i.csv"
	mv "gpurepair-$i.metrics.csv" "/home/gpurepair/tools/results/gpurepair_inspection/gpurepair-$i.metrics.csv"
	
	clear
	echo "${green}Running GPURepair (--disable-grid --disable-inspect) - Iteration ${i}${reset}"

	cd "/home/gpurepair/tools/gpurepair"
	rm -rf "./tests"
	cp -r "./tests-bkp" "./tests"

	python3 ./grtester.py ${path} --threads=1 --time-as-csv --csv-file=gpurepair-$i.csv --gropt=--detailed-logging --gropt=--disable-grid --gropt=--disable-inspect
	mv "gpurepair-$i.csv" "/home/gpurepair/tools/results/gpurepair_grid_inspection/gpurepair-$i.csv"
	mv "gpurepair-$i.metrics.csv" "/home/gpurepair/tools/results/gpurepair_grid_inspection/gpurepair-$i.metrics.csv"
done

clear
echo "${green}Generating the reports${reset}"

cd "/home/gpurepair/tools/gpurepair/bin"
mono ./GPURepair.ReportGenerator.exe summary /home/gpurepair/tools/results

clear
echo "${green}All tests successfully completed, the results and the report are available at /home/gpurepair/tools/results${reset}"
