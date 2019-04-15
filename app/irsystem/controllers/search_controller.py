from . import *
from app.irsystem.models.helpers import *
from app.irsystem.models.helpers import NumpyEncoder as NumpyEncoder

import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize
from nltk import word_tokenize
from nltk.util import ngrams
import re

project_name = "S"
net_id = "...."

@irsystem.route('/', methods=['GET'])
def search():
	query = request.args.get('search')
	WineType = request.args.get('flavor')
	MinPrice = request.args.get('MinPrice')
	MaxPrice = request.args.get('MaxPrice')
	Wine  = request.args.get('color')
	if not Wine:
		data = []
		output_message = ''
	else:

	## comment to update code
		wines = pd.read_csv("~/app/app/data/winemag-data-130k-v2.csv")
		wines_color = pd.read_csv("~/app/app/data/Wine_color.csv")
		wines_color.rename(columns={'Variety':'variety'}, inplace=True)
		master_data = pd.merge(wines, wines_color, on='variety', how='left')
		master_data = master_data.groupby(master_data.title).first()


		master_data['index'] = np.arange(len(master_data))
		master_data_cpy = master_data.copy(deep = True)

		vectorizer = TfidfVectorizer(decode_error = 'ignore', ngram_range=(1,1), stop_words = 'english', max_df = 1.0,min_df = 10)
		vin_tfidf = vectorizer.fit_transform([x for x in master_data['description']])

		def query_vec(s):
			s = s.lower()
			s = re.sub(r'[^a-zA-Z0-9\s]', ' ', s)
			tokens = [token for token in s.split(" ") if token != ""]
			bigram = [x[0]+" "+ x[1] for x in list(ngrams(tokens, 2))]
			query_token = tokens + bigram
			word_to_index = vectorizer.vocabulary_
			query_vec = np.zeros((1,len(word_to_index)))
			for i,v in enumerate(query_token):
				if v in word_to_index:
					query_vec[0,word_to_index[v]] = 1
			return query_vec

		def similar_vin(query_vec, vin_tfidf):
			sims = np.dot(vin_tfidf.toarray(),query_vec.T)
			asort = np.argsort(-sims,axis = 0)
			return asort

		def shortlist_vin(query_vec, color = None, price_min = None, price_max = None):
			reccommendation = similar_vin(query_vec,vin_tfidf)
			chk = pd.DataFrame(reccommendation)
			chk.columns = ['index']
			chk1 = pd.merge(chk, master_data_cpy, on='index', how='left')

			if color != None:
				chk1 =  chk1[chk1.Color == color]


			return chk1

		winetype = WineType
		query_vec = query_vec(winetype)
		shortlist = shortlist_vin(query_vec, color = Wine, price_min = MinPrice, price_max = MaxPrice)

		output_message = "Searched"
		print (output_message)
		#data = "You want " + query + " Wine " + " in " + price
		data = shortlist['title'][:10].values.tolist()

		print (type(data))
	return render_template('search.html', name=project_name, netid=net_id, output_message=output_message, data = data)
