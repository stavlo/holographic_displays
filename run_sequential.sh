#!/bin/bash

# chunk 1 z=1.0[m]
python main.py --model classic --z 1.0 && \
python main.py --epochs 200 --model skip_connection --z 1.0 --filter_size 0 && \
python main.py --epochs 200 --model conv --z 1.0 --lr 5e-4 --filter_size 0 && \

# chunk 1 z=0.5[m]
python main.py --model classic --z 0.5 && \
python main.py --epochs 200 --model skip_connection --z 0.5 --filter_size 0 && \
python main.py --epochs 200 --model conv --z 0.5 --lr 1e-4 --filter_size 0