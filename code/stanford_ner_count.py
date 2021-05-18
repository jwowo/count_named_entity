from nltk.tag import StanfordNERTagger
import pandas as pd
from tqdm import tqdm
import os

path = '/Users/jongwoo/Desktop/Project/stanford_ner/'

# Set CLASSPATH
os.environ["CLASSPATH"] = path + 'stanford-ner.jar'

# Load dataset
df = pd.read_csv(path + '/Random_Art_story_0.tsv', sep='\t')

# Make dataframe to save the NER counts
df_ner_count = pd.DataFrame(columns={'ID', 'LOCATION', 'PERSON', 'ORGANIZATION', 
                               'MONEY', 'PERCENT', 'DATE', 'TIME'})

# Set stanford Name Entitiy Tagger
st = StanfordNERTagger('/home/jwhan/project/stanford_NER/stanford-ner-2020-11-17/classifiers/english.muc.7class.distsim.crf.ser.gz')


for idx, text in enumerate(tqdm(df['review'])):
    ner_tag_list = st.tag(text.split())
    
    index_id = df['ID'][idx]
    tag_count_dict = {'ID':index_id, 'LOCATION':0, 'PERSON':0, 'ORGANIZATION':0, 'MONEY':0, 
                  'PERCENT':0, 'DATE':0, 'TIME':0}
     
    
    for word, tag in ner_tag_list:
        if tag in tag_count_dict.keys():
            tag_count_dict[tag] += 1
    
    df_ner_count = df_ner_count.append(tag_count_dict, ignore_index=True)

df_ner_count