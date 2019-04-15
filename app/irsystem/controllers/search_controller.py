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
	#query = request.args.get('search')
	WineType = request.args.get('flavor')
	MinPrice = request.args.get('MinPrice')
	MaxPrice = request.args.get('MaxPrice')
	color  = request.args.get('color')
	if not color:
		data = []
		output_message = ''
	else:

	## comment to update code
		master_data = pd.read_pickle("~/app/data/master_data_lean.pkl")
		vin_tfidf = np.load("~/app/data/vin_tfidf.npy").item()
		word_to_index = np.load("~/app/data/word_to_index.npy").item()


		def query_vec(s):
			s = s.lower()
			s = re.sub(r'[^a-zA-Z0-9\s]', ' ', s)
			tokens = [token for token in s.split(" ") if token != ""]
			bigram = [x[0]+" "+ x[1] for x in list(ngrams(tokens, 2))]
			query_token = tokens + bigram
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
			master_data_cpy = master_data.copy(deep = True)
			reccommendation = similar_vin(query_vec,vin_tfidf)
			chk = pd.DataFrame(reccommendation)
			chk.columns = ['index']
			chk1 = pd.merge(chk, master_data_cpy, on='index', how='left')

			if color != None:
				chk1 =  chk1[chk1.Color == color]

			if price_min != None and price_max != None:
				chk1 = chk1[chk1.price < price_max]
				chk1 = chk1[chk1.price > price_min]
			return chk1





		output_message = "Searched"

		#data = "You want " + query + " Wine " + " in " + price

		query = query_vec(WineType)
		shortlist = shortlist_vin(query, color = color, price_min = int(MinPrice), price_max = int(MaxPrice))
		data= shortlist['title'][:10].values.tolist()

	return render_template('search.html', name=project_name, netid=net_id, output_message=output_message, data = data)
