import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

BASE = "cardiffnlp/twitter-roberta-base-sentiment"
ADAPTER = "/tmp/twitter_v39_out/sentiment_analyzer/artifacts/models/twitter"

print("loading_tokenizer.....")
tokenizer=AutoTokenizer.from_pretrained(ADAPTER)

print("loading base model....")
base=AutoModelForSequenceClassification.from_pretrained(BASE,num_labels=3,ignore_mismatched_sizes=True)


model=PeftModel.from_pretrained(base,ADAPTER)
model.eval()


labels={0:'negative',1:'neutral',2:'positive'}

tests=['This product is amazing! I love it!',
    'This is the worst thing I have ever bought.',
    'It is okay, nothing special.']

for text in tests:
    inputs=tokenizer(text,return_tensors='pt',truncation=True,padding=True,max_length=128)    

    with torch.no_grad():
        outputs=model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)[0]    
    pred=torch.argmax(probs).item()    
    print(f'{text:<50} {labels[pred]:<10} {probs[pred]:.3f}')