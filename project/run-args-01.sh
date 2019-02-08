#!/bin/bash

while true; do
	python3 WDCGAN-tpu-variable.py --tpu node3 --lr 0.01 --dir gan-tpu-lr-01 
done
