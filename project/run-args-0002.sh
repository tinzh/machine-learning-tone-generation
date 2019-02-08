#!/bin/bash

while true; do
	python3 WDCGAN-tpu-variable.py --tpu node1 --lr 0.0002 --dir gan-tpu-nhwc-opt-test
done
