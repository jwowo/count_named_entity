import pandas as pd
from nltk.tag import StanfordNERTagger
import os
from tqdm import tqdm

# Alternatively to setting the CLASSPATH add the jar via their path:
os.environ["CLASSPATH"] = "./stanford-ner.jar"

# Set Path
path = './'
output_path = './output_files/'

# Set StanfordNERTagger
st = StanfordNERTagger(path + 'classifiers/english.muc.7class.distsim.crf.ser.gz')

# Set data name list
data_list = ['b_comment_full', 'c_comment_full', 'full_comment_full', 'full_text_full', 'story_full', 'update_full']

for data in data_list:
	# Load Dataset
	df = pd.read_csv(path + 'data/' + data + '.tsv', sep='\t')

	# Make dataframe to save the NER counts
	df_ner_count = pd.DataFrame(columns={'ID','LOCATION', 'PERSON', 'ORGANIZATION', 
		'MONEY', 'PERCENT', 'DATE', 'TIME'})

	print('Start counting NER...')

	for idx, text in enumerate(tqdm(df['content'])):
		ner_tag_list = st.tag(text.split())
        
		index_id = df['ID'][idx]
        
		tag_count_dict = {'ID':index_id, 'LOCATION':0, 'PERSON':0, 'ORGANIZATION':0, 'MONEY':0, 
				'PERCENT':0, 'DATE':0, 'TIME':0}
        
		for word, tag in ner_tag_list:
			if tag in tag_count_dict.keys():
				tag_count_dict[tag] += 1
        
	df_ner_count = df_ner_count.append(tag_count_dict, ignore_index=True)

	df_new = pd.merge(df, df_ner_count)
	df_new.to_csv(output_path + data + '.tsv', sep='\t')

	print('Finished counting... : ' + data + '.tsv')

print('Finished counting Named Entity...')
