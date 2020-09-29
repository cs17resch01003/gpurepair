#!/bin/bash
rm -rf /datadrive/reports

for i in 0 1 2 3
do
	cd /datadrive/gpuverify/
	mv ./gvfindtools.py ../gvfindtools.py
	git reset --hard
	git clean -fd
	git pull https://cs17resch01003:MtG171-48@github.com/cs17resch01003/gpuverify.git master
	
	clear
	rm -rf Binaries/
	xbuild /p:Configuration=Release GPUVerify.sln
	
	mv ../gvfindtools.py ./gvfindtools.py
	clear
	cd /datadrive/gpuverify
	./gvtester.py ./testsuite --threads=1 --time-as-csv --csv-file=gpuverify-$i.csv 2>&1 | tee gpuverify-run-$i.log
	
	mkdir -p /datadrive/reports
	mv /datadrive/gpuverify/gpuverify-$i.csv /datadrive/reports/gpuverify-$i.csv
	mv /datadrive/gpuverify/gpuverify-run-$i.log /datadrive/reports/gpuverify-run-$i.log
done
