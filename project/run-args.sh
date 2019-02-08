#!/bin/bash

while true; do
	python3 WDCGAN-tpu-variable.py --tpu node2 --lr 0.001 --dir gan-tpu-lr-001 
done
