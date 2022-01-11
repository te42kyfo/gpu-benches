#!/bin/sh

range=256



    
make N=0 PREFIX=./build 1>&2 &
make N=1 PREFIX=./build 1>&2 &
make N=2 PREFIX=./build 1>&2 &
make N=4 PREFIX=./build 1>&2 &

for (( d=8 ; d<=$range; d+=8 ))
do
    make N=$d PREFIX=./build 1>&2 &
    while test $(jobs -p | wc -w) -ge 100; do sleep 1; done
done


wait

echo "-- Finished Building --"


./build/cu-roof0
./build/cu-roof1
./build/cu-roof2
./build/cu-roof4


for (( d=8 ; d<=$range; d+=8 ))
do
    ./build/cu-roof$d
done

