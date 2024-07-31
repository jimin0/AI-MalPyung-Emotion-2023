cd EA
python inference_logits.py \
    --model_ckpt_path ./results/20230926_223646_프렌치불독/ --tokenizer ./results/20230926_223646_프렌치불독/fold_1 --output_jsonl '../Ensemble/프렌치불독' \
    --special_token True --model_class EnhancedPoolingModel2 --batch-size 32 --classifier_dropout_prob 0.05 --classifier_hidden_size 768

python inference_logits.py \
    --model_ckpt_path ./results/20230930_202016_시바개/ --tokenizer ./results/20230930_202016_시바개/fold_1 --output_jsonl '../Ensemble//시바개' \
    --special_token True --model_class EnhancedPoolingModel2 --batch-size 32 --classifier_dropout_prob 0.05 --classifier_hidden_size 768 

python inference_logits.py \
    --model_ckpt_path ./results/20231019_005638_매멋/ --tokenizer ./results/20231019_005638_매멋/fold_1 --output_jsonl '../Ensemble/매멋' \
    --special_token True --model_class RealAttention5 --batch-size 32 --classifier_dropout_prob 0.05 --classifier_hidden_size 768 --hidden 128 --prompt True --clean True

python inference_logits.py \
    --model_ckpt_path ./results/20231019_234739_코끼리/ --tokenizer ./results/20231019_234739_코끼리/fold_1 --output_jsonl '../Ensemble/코끼리' \
    --special_token True --model_class RealAttention5 --batch-size 32 --classifier_dropout_prob 0.05 --classifier_hidden_size 768 --hidden 32 --max-seq-len 128 --prompt True --clean True

python inference_logits.py \
    --model_ckpt_path ./results/20231017_172653_천둥오리/ --tokenizer ./results/20231017_172653_천둥오리/fold_1 --output_jsonl '../Ensemble/천둥오리' \
    --special_token True --model_class RealAttention5 --batch-size 32 --classifier_dropout_prob 0.1 --classifier_hidden_size 768 --hidden 128 --prompt True 

python inference_logitslora.py \
    --model_path EleutherAI/polyglot-ko-12.8b --model_ckpt_path ./results/은색오리/checkpoint-5300/ --tokenizer ./results/은색오리/checkpoint-5300/ --output_jsonl '../Ensemble/은색오리logits5300' \
    --special_token True --batch-size 8 --prompt True

python inference_logitslora.py \
    --model_path EleutherAI/polyglot-ko-12.8b --model_ckpt_path ./results/은색오리/checkpoint-5400/ --tokenizer ./results/은색오리/checkpoint-5400/ --output_jsonl '../Ensemble/은색오리logits5400' \
    --special_token True --batch-size 8 --prompt True

python inference_logitslora.py \
    --model_path nlpai-lab/kullm-polyglot-12.8b-v2 --model_ckpt_path ./results/동색오리/checkpoint-4500/ --tokenizer ./results/동색오리/checkpoint-4500/ --output_jsonl '../Ensemble/동색오리logits4500' \
    --special_token True --batch-size 8 --prompt True

python inference_logitslora.py \
    --model_path nlpai-lab/kullm-polyglot-12.8b-v2 --model_ckpt_path ./results/동색오리/checkpoint-5000/ --tokenizer ./results/동색오리/checkpoint-5000/ --output_jsonl '../Ensemble/동색오리logits5000' \
    --special_token True --batch-size 8 --prompt True

python inference_logitslora.py \
    --model_path nlpai-lab/kullm-polyglot-12.8b-v2 --model_ckpt_path ./results/황색오리/checkpoint-4200/ --tokenizer ./results/황색오리/checkpoint-4200/ --output_jsonl '../Ensemble/황색오리logits4200' \
    --special_token True --batch-size 8 

python inference_logitslora.py \
    --model_path nlpai-lab/kullm-polyglot-12.8b-v2 --model_ckpt_path ./results/황색오리/checkpoint-4300/ --tokenizer ./results/황색오리/checkpoint-4300/ --output_jsonl '../Ensemble/황색오리logits4300' \
    --special_token True --batch-size 8 

python ensemble.py