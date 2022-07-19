#!/bin/bash

logdir="../data/logs"

action_repeat=2
early_stopping=True
vec_normalize=True
algo="PPO"
model="SystemSimplePID"

case_study=${1:-"cs1"}

python ${case_study}.py --model ${model} --algo ${algo} --logdir ${logdir} --early_stopping ${early_stopping} --action_repeat ${action_repeat} --vec_normalize ${vec_normalize} --mode "test"
