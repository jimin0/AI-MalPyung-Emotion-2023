wandb login 20f894088a42a42e5eef02b48b1e6cce6805fdfe
export CUDA_VISIBLE_DEVICES=0
python run.py --special_token True \
    --model_class 'RealAttention5' \
    --epochs 20 --model_path nlp04/korean_sentiment_analysis_kcelectra --tokenizer nlp04/korean_sentiment_analysis_kcelectra \
    --batch-size 64 --learning-rate 3e-5 \
    --classifier_dropout_prob 0.05 \
    --classifier_hidden_size 768 \
    --kfold 5  --kfold_data_path ./data/tt5.jsonl \
