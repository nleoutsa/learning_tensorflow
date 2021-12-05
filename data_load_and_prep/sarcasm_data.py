import json


with open("datasets/sarcasm/Sarcasm_Headlines_Dataset.json", 'r') as f:
    datastore = json.load(f)

data = {
    'sentences': [],
    'labels': [],
    'urls': []
}

for item in datastore:
    data['sentences'].append(item['headline'])
    data['labels'].append(item['is_sarcastic'])
    data['urls'].append(item['article_link'])
