from torch import nn
from transformers import AutoModel
import torch
import torch.nn.functional as F

class RealAttention5(nn.Module):
    def __init__(self, args, label, tokenizer):
        super(RealAttention5, self).__init__()
        self.model = AutoModel.from_pretrained(args.model_path)
        self.num_label = label
        self.labels_classifier = EnhancedClassifier(args, label)
        self.model.resize_token_embeddings(len(tokenizer))
        self.tokenizer = tokenizer
        self.args = args
        self.ngrams = args.n_gram
        
        # Linear layers for Query, Key, Value for each representation
        self.linear_query_ngram = nn.Linear(args.classifier_hidden_size, args.classifier_hidden_size)
        self.linear_query_word = nn.Linear(args.classifier_hidden_size, args.classifier_hidden_size)
        self.linear_key = nn.Linear(args.classifier_hidden_size, args.classifier_hidden_size)
        self.linear_value = nn.Linear(args.classifier_hidden_size, args.classifier_hidden_size)
        
    def forward(self, input_ids, attention_mask, labels=None):
        e_start_id = self.tokenizer.convert_tokens_to_ids('<e>')
        e_end_id = self.tokenizer.convert_tokens_to_ids('</e>')
        sep_id = self.tokenizer.convert_tokens_to_ids('[SEP]')  # SEP 토큰의 ID를 찾습니다.

        # token_type_ids 생성
        sep_positions = (input_ids == sep_id).nonzero(as_tuple=True)
        token_type_ids = torch.zeros_like(input_ids)
        for idx in range(input_ids.size(0)):
            sep_position = sep_positions[1][sep_positions[0] == idx]
            if sep_position.size(0) > 0:
                token_type_ids[idx, sep_position[0]:] = 1

        outputs = self.model(input_ids, attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
    
        e_start_positions = (input_ids == e_start_id).nonzero(as_tuple=True)
        e_end_positions = (input_ids == e_end_id).nonzero(as_tuple=True)

        attention_outputs_ngram = []
        word_mean_pooled_outputs = []
        post_sep_outputs = []
        
        for idx in range(input_ids.size(0)):
            start_pos = e_start_positions[1][e_start_positions[0] == idx]
            end_pos = e_end_positions[1][e_end_positions[0] == idx]
            sep_position = sep_positions[1][sep_positions[0] == idx]
            
            if start_pos.size(0) > 0 and end_pos.size(0) > 0:
                
                
                # 1. [SEP] 이후의 문장에 대한 임베딩
                post_sep_embedding = sequence_output[idx, sep_position[0]+1:]
                post_sep_output, _ = torch.max(post_sep_embedding, dim=0)
                post_sep_outputs.append(post_sep_output)
                
                # 2. Ngram 토큰 정보 수정
                ngram_before = sequence_output[idx, max(sep_position[0] + 1, start_pos[0] - 3):start_pos[0], :]
                ngram_after = sequence_output[idx, end_pos[0] + 1:end_pos[0] + 1 + self.ngrams*2, :]
                ngram_combined = torch.cat([ngram_before, ngram_after], dim=0)
                ngram_output, _ = torch.max(ngram_combined, dim=0)
            
                # 3. 단어 그 자체의 정보
                word_output = sequence_output[idx, start_pos[0] + 1:end_pos[0], :]
                word_output, _ = torch.max(word_output, dim=0)
                word_mean_pooled_outputs.append(word_output)
                
                # Generate Query, Key, Value from the embeddings for each representation
                Q_ngram = self.linear_query_ngram(ngram_output).unsqueeze(0)
                # Q_word = self.linear_query_word(word_output).unsqueeze(0)
                
                K = self.linear_key(sequence_output[idx])
                V = self.linear_value(sequence_output[idx])

                # Attention mechanism for each representation
                attention_weights_ngram = torch.matmul(Q_ngram, K.transpose(-2, -1))
                attention_weights_ngram = attention_weights_ngram / (self.args.classifier_hidden_size ** 0.5)
                attention_weights_ngram = F.softmax(attention_weights_ngram, dim=-1)
                attention_output_ngram = torch.matmul(attention_weights_ngram, V)
                attention_outputs_ngram.append(attention_output_ngram)
                
            else:
                zero_output = torch.zeros_like(sequence_output[0, 0, :]).unsqueeze(0)
                attention_outputs_ngram.append(zero_output)
                word_mean_pooled_outputs.append(torch.zeros_like(sequence_output[0, 0, :]))
                post_sep_outputs.append(torch.zeros_like(sequence_output[0, 0, :]))

        post_sep_outputs = torch.stack(post_sep_outputs)
        attention_outputs_ngram = torch.stack(attention_outputs_ngram).squeeze(1)
        word_mean_pooled_outputs = torch.stack(word_mean_pooled_outputs)
        
        # Directly use cls token representation without attention
        cls_output = sequence_output[:, 0, :]
        
        combined_output = torch.cat([cls_output, attention_outputs_ngram, word_mean_pooled_outputs, post_sep_outputs], dim=1)
        
        logits = self.labels_classifier(combined_output)

        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)

        return loss, logits

        # self.dense = nn.Linear(self.classifier_hidden_size * 4, self.classifier_hidden_size * 128) 
        # self.final_dense = nn.Linear(self.classifier_hidden_size * 128, num_label)

class EnhancedClassifier(nn.Module):
    def __init__(self, args, num_label):
        super().__init__()
        # 입력 차원을 args.classifier_hidden_size * 3으로 변경
        self.classifier_hidden_size = args.classifier_hidden_size
        self.dense = nn.Linear(self.classifier_hidden_size * 4, self.classifier_hidden_size * args.hidden) 
        self.final_dense = nn.Linear(self.classifier_hidden_size * args.hidden, num_label)
        self.dropout = nn.Dropout(args.classifier_dropout_prob)

    def forward(self, combined_output):
        combined_output = self.dropout(combined_output)
        combined_output = self.dense(combined_output)
        combined_output = F.gelu(combined_output)
        logits = self.final_dense(combined_output)
        return logits


