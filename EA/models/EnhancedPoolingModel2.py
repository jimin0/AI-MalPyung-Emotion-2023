from torch import nn
from transformers import AutoModel
import torch
import torch.nn.functional as F

class EnhancedPoolingModel2(nn.Module):
    def __init__(self, args, label, tokenizer):
        super(EnhancedPoolingModel2, self).__init__()
        self.model = AutoModel.from_pretrained(args.model_path)
        self.num_label = label
        self.labels_classifier = EnhancedClassifier(args, label)
        self.model.resize_token_embeddings(len(tokenizer))
        self.tokenizer = tokenizer
        
        # Attention weights for attention pooling
        self.attention_weights = nn.Linear(args.classifier_hidden_size, 1)
        
    def forward(self, input_ids, attention_mask, labels=None):
        e_start_id = self.tokenizer.convert_tokens_to_ids('<e>')
        e_end_id = self.tokenizer.convert_tokens_to_ids('</e>')

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=None
        )
        sequence_output = outputs[0]

        mean_squared_pooled_outputs = []
        max_pooled_outputs = []
        attention_pooled_outputs = []
        for idx, (input_seq, att_mask) in enumerate(zip(input_ids, attention_mask)):
            e_start_pos = (input_seq == e_start_id).nonzero(as_tuple=True)
            e_end_pos = (input_seq == e_end_id).nonzero(as_tuple=True)
            
            if e_start_pos[0].size(0) > 0 and e_end_pos[0].size(0) > 0:
                start = e_start_pos[0][0]
                end = e_end_pos[0][0]
                if start + 1 < end:  # Ensure there are tokens between <e> and </e>
                    mean_squared_pooled_output = torch.sqrt(torch.mean(sequence_output[idx, start+1:end]**2, dim=0))
                    max_pooled_output, _ = torch.max(sequence_output[idx, start+1:end], dim=0)
                    
                    # Attention pooling
                    att_weights = torch.tanh(self.attention_weights(sequence_output[idx, start+1:end]))
                    att_weights = F.softmax(att_weights, dim=0)
                    attention_pooled_output = torch.sum(att_weights * sequence_output[idx, start+1:end], dim=0)
                    
                    mean_squared_pooled_outputs.append(mean_squared_pooled_output)
                    max_pooled_outputs.append(max_pooled_output)
                    attention_pooled_outputs.append(attention_pooled_output)
                else:  # If no tokens between <e> and </e>
                    zero_output = torch.zeros_like(sequence_output[0, 0, :])
                    mean_squared_pooled_outputs.append(zero_output)
                    max_pooled_outputs.append(zero_output)
                    attention_pooled_outputs.append(zero_output)
            else:
                zero_output = torch.zeros_like(sequence_output[0, 0, :])
                mean_squared_pooled_outputs.append(zero_output)
                max_pooled_outputs.append(zero_output)
                attention_pooled_outputs.append(zero_output)

        mean_squared_pooled_outputs = torch.stack(mean_squared_pooled_outputs)
        max_pooled_outputs = torch.stack(max_pooled_outputs)
        attention_pooled_outputs = torch.stack(attention_pooled_outputs)
        
        cls_output = sequence_output[:, 0, :]
        
        logits = self.labels_classifier(cls_output, mean_squared_pooled_outputs, max_pooled_outputs, attention_pooled_outputs)

        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)
            
        # FocalLoss 테스트용 -> 성능높게 안나옴
        
        # loss = None
        # if labels is not None:
        #     loss_fct = FocalLoss()
        #     loss = loss_fct(logits, labels)


        return loss, logits

class EnhancedClassifier(nn.Module):
    def __init__(self, args, num_label):
        super(EnhancedClassifier, self).__init__()
        self.dense1 = nn.Linear(args.classifier_hidden_size, args.classifier_hidden_size)
        self.dense2 = nn.Linear(args.classifier_hidden_size, args.classifier_hidden_size)
        self.dense3 = nn.Linear(args.classifier_hidden_size, args.classifier_hidden_size)
        self.dense4 = nn.Linear(args.classifier_hidden_size, args.classifier_hidden_size)
        self.final_dense = nn.Linear(args.classifier_hidden_size*4, num_label)
        self.dropout = nn.Dropout(args.classifier_dropout_prob)

    def forward(self, cls_output, mean_squared_pooled_output, max_pooled_output, attention_pooled_output):
        cls_output = self.dropout(cls_output)
        cls_output = self.dense1(cls_output)
        cls_output = F.gelu(cls_output)

        mean_squared_pooled_output = self.dropout(mean_squared_pooled_output)
        mean_squared_pooled_output = self.dense2(mean_squared_pooled_output)
        mean_squared_pooled_output = F.gelu(mean_squared_pooled_output)

        max_pooled_output = self.dropout(max_pooled_output)
        max_pooled_output = self.dense3(max_pooled_output)
        max_pooled_output = F.gelu(max_pooled_output)

        attention_pooled_output = self.dropout(attention_pooled_output)
        attention_pooled_output = self.dense4(attention_pooled_output)
        attention_pooled_output = F.gelu(attention_pooled_output)

        combined_output = torch.cat([cls_output, mean_squared_pooled_output, max_pooled_output, attention_pooled_output], dim=1)
        logits = self.final_dense(combined_output)

        return logits


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=True, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduction = reduction

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss
