#!/bin/bash

# get all execution files in ./bin
files=./bin/*
# split file names into arr
arr=$(echo $files | tr " " "\n")
max_ver_num="$"
exe_file=${arr[0]}
# iterate over all file names to get the largest version number
for x in $arr
do
    output=$(grep -o "[0-9]\.[0-9]" <<< "$x")
    if [ "$output" \> "$max_ver_num" ]; then
        exe_file=$x
    fi
done

# put OS and Device type here
SUFFIX="ubuntu16.04_TitanV"

mkdir -p eval/$SUFFIX

for i in delaunay_n10
do
    echo $exe_file market ../../dataset/small/tri_sm.mtx ../../dataset/large/$i/$i.mtx"> ./eval/$SUFFIX/${NAME[$i]}.$SUFFIX${MARKS}.txt"
         $exe_file market ../../dataset/small/tri_sm.mtx ../../dataset/large/$i/$i.mtx > ./eval/$SUFFIX/${NAME[$i]}.$SUFFIX${MARKS}.txt
    sleep 1
done
