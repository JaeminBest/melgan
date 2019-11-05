#!/bin/bash
set -e
# Run
case "$1" in
    start)
        export IP_ADDR=$(curl -s checkip.dyndns.org | sed -e 's/.*Current IP Address: //' -e 's/<.*$//')
        echo "Dataset preparation"
        python data_preparation.py --folder /data -s -d 
        echo "Training start"
        nohup python trainer.py -n $NAME &
        tensorboard --logdir /app/logs/$NAME --port=$EXPORT --host=0.0.0.0
        echo "done making training block"
esac
