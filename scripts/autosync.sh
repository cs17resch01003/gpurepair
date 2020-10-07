#!/bin/bash
rm -rf /datadrive/reports

for i in 0 1 2 3
do
	cd /datadrive/autosync/
	git reset --hard
	git clean -fd
	git pull https://cs17resch01003:MtG171-48@github.com/cs17resch01003/autosync.git master

	clear
	cd /datadrive/autosync/src/AutoSync
	python3 testrunner.py /datadrive/autosync/tests/testsuite | tee autosync-run-$i.csv

	cd /datadrive/autosync/tests
	zip -r autosync-snapshot-$i.zip .
	
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
	./gvtester.py /datadrive/autosync/tests/solutions --threads=1 --time-as-csv --csv-file=gpuverify-$i.csv 2>&1 | tee gpuverify-run-$i.log

	mkdir -p /datadrive/reports
	mv /datadrive/autosync/src/AutoSync/autosync-run-$i.csv /datadrive/reports/autosync-run-$i.csv
	mv /datadrive/autosync/tests/autosync-snapshot-$i.zip /datadrive/reports/autosync-snapshot-$i.zip
	mv /datadrive/gpuverify/gpuverify-$i.csv /datadrive/reports/gpuverify-$i.csv
	mv /datadrive/gpuverify/gpuverify-run-$i.log /datadrive/reports/gpuverify-run-$i.log
done
