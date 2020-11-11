from torch import nn
from transformers import AutoModelForMaskedLM, AutoTokenizer

model = AutoModelForMaskedLM.from_pretrained('squeezebert/squeezebert-mnli')
tokenizer = AutoTokenizer.from_pretrained('squeezebert/squeezebert-mnli')

#model.tie_weights()
input_txt = ["[MASK] was an American [MASK]  and lawyer who served as the 16th president  of the United States from 1861 to 1865. [MASK] led the nation through the American Civil War, the country's greatest [MASK], [MASK], and [MASK] crisis. ", \
             "George [MASK], who served as the first  president of the United States from [MASK] to 1797, was an American political leader, [MASK] [MASK], statesman, and Founding Father. Previously, he led Patriot forces to [MASK] in the nation's War for Independence. ", \
             "[MASK], the first African-American [MASK] of the [MASK] [MASK], is an American politician and attorney who served as the 44th [MASK] of the United States from [MASK] to 2017.  [MASK] was a member of the [MASK] [MASK]. "]
#input_txt =
input_txt= [i.replace("[MASK]", tokenizer.mask_token) for i in input_txt] #
inputs = tokenizer(input_txt, return_tensors='pt', add_special_tokens=True, padding=True)
inputs['output_attentions'] = True
inputs['output_hidden_states'] = True
inputs['return_dict'] = True
outputs = model(**inputs)
if True:
  predictions = outputs.logits
  for pred in predictions:
    print ("**")
    sorted_preds, sorted_idx = pred.sort(dim=-1, descending=True)
    for k in range(2):
        predicted_index = [sorted_idx[i, k].item() for i in range(0,len(predictions[0]))]
        predicted_token = ' '.join([tokenizer.convert_ids_to_tokens([predicted_index[x]])[0] for x in range(1,len(predictions[0]))]).replace('Ġ', ' ').replace('  ', ' ').replace('##', '')
        print(predicted_token)

