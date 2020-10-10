#!/bin/bash
rm -rf /datadrive/reports

for i in 0 1 2 3
do
	cd /datadrive/gpurepair/
	git reset --hard
	git clean -fd
	git pull https://cs17resch01003:MtG171-48@github.com/cs17resch01003/gpurepair.git master

	clear
	rm -rf Binaries/
	cd src
	xbuild /p:Configuration=Release /p:Platform=x86 GPURepair.sln

	clear
	cd /datadrive/gpurepair/src/Toolchain
	./grtester.py /datadrive/gpurepair/tests/testsuite --threads=1 --time-as-csv --csv-file=gpurepair-$i.csv --gropt=--detailed-logging --gropt=--disable-inspect 2>&1 | tee gpurepair-run-$i.log

	cd /datadrive/gpurepair/tests/testsuite
	zip -r gpurepair-snapshot-$i.zip .

	clear
	mkdir -p /datadrive/reports
	mv /datadrive/gpurepair/src/Toolchain/gpurepair-$i.csv /datadrive/reports/gpurepair-$i.csv
	mv /datadrive/gpurepair/src/Toolchain/gpurepair-$i.metrics.csv /datadrive/reports/gpurepair-$i.metrics.csv
	mv /datadrive/gpurepair/src/Toolchain/gpurepair-run-$i.log /datadrive/reports/gpurepair-run-$i.log
	mv /datadrive/gpurepair/tests/testsuite/gpurepair-snapshot-$i.zip /datadrive/reports/gpurepair-snapshot-$i.zip
done
