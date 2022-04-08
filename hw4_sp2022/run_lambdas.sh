#!/bin/bash
for val in easy hard bio finance speech vision
do
    python3 main_lambda_means.py --mode train --train-data datasets/lambda_means/${val}.train --model-file models/${val}.model --cluster-lambda 0.0 --clustering-training-iterations 10;
    python3 main_lambda_means.py --mode test --test-data datasets/lambda_means/${val}.train --model-file models/${val}.model --predictions-file predictions/${val}.train.predictions;
    python3 main_lambda_means.py --mode test --test-data datasets/lambda_means/${val}.dev --model-file models/${val}.model --predictions-file predictions/${val}.dev.predictions;
    python3 utils/cluster_accuracy.py datasets/lambda_means/${val}.train predictions/${val}.train.predictions > outputs/${val}.txt;
    python3 utils/cluster_accuracy.py datasets/lambda_means/${val}.dev predictions/${val}.dev.predictions >> outputs/${val}.txt;
    python3 utils/number_clusters.py predictions/${val}.train.predictions >> outputs/${val}.txt;
    python3 utils/number_clusters.py predictions/${val}.dev.predictions >> outputs/${val}.txt;
done