#!/bin/bash

# chunk 1 z=0.01[m]
python main.py --model classic --z 0.01 && \
python main.py --epochs 300 --model skip_connection --z 0.01 && \
python main.py --epochs 300 --model conv --z 0.01 && \
# chunk 1 z=0.05[m]
python main.py --model classic --z 0.05 && \
python main.py --epochs 300 --model skip_connection --z 0.05 && \
python main.py --epochs 300 --model conv --z 0.05 && \
# chunk 1 z=0.1[m]
python main.py --model classic --z 0.1 && \
python main.py --epochs 300 --model skip_connection --z 0.1 && \
python main.py --epochs 300 --model conv --z 0.1 && \
# chunk 1 z=0.25[m]
python main.py --model classic --z 0.25 && \
python main.py --epochs 300 --model skip_connection --z 0.25 && \
python main.py --epochs 300 --model conv --z 0.25 && \
# chunk 1 z=0.5[m]
python main.py --model classic --z 0.5 && \
python main.py --epochs 300 --model skip_connection --z 0.5 && \
python main.py --epochs 300 --model conv --z 0.5
# chunk 1 z=1.0[m]
python main.py --model classic --z 1.0 && \
python main.py --epochs 300 --model skip_connection --z 1.0 && \
python main.py --epochs 300 --model conv --z 1.0
# chunk 1 z=1.5[m]
python main.py --model classic --z 1.5 && \
python main.py --epochs 300 --model skip_connection --z 1.5 && \
python main.py --epochs 300 --model conv --z 1.5