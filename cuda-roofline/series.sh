#!/bin/sh

range=256

for (( d=0 ; d<=$range; d+=8 ))
do
    make N=$d PREFIX=./build 1>&2 &
    while test $(jobs -p | wc -w) -ge 100; do sleep 1; done
done


wait

echo "-- Finished Building --"

for (( d=0 ; d<=$range; d+=8 ))
do
    ./build/cu-roof$d
done

