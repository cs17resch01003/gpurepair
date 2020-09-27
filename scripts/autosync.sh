#!/bin/bash
for i in 0 1 2 3
do
	cd /datadrive/autosync/
	git reset --hard
	git clean -fd
	git pull https://cs17resch01003:MtG171-48@github.com/cs17resch01003/autosync.git master

	clear
	cd /datadrive/autosync/src/AutoSync
	python3 testrunner.py /datadrive/autosync/tests/testsuite | tee autosync-run-$i.csv

	cd /datadrive/autosync/tests/testsuite
	zip -r autosync-snapshot-$i.zip .

	mkdir -p /datadrive/reports
	mv /datadrive/autosync/src/AutoSync/autosync-run-$i.csv /datadrive/reports/autosync-run-$i.csv
	mv /datadrive/autosync/tests/testsuite/autosync-snapshot-$i.zip /datadrive/reports/autosync-snapshot-$i.zip
done
