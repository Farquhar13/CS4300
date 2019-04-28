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
#from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
#import matplotlib


# relative path stuff
import os
dirname = os.path.dirname(__file__)
master_data_path = os.path.join(dirname, '..', '..', 'data', 'master_data_lean.pkl')
vin_tfidf_path = os.path.join(dirname, '..', '..', 'data', 'vin_tfidf.npy')
word_to_index_path = os.path.join(dirname, '..', '..', 'data', 'word_to_index.npy')
variety_tfidf_path = os.path.join(dirname, '..', '..', 'data', 'variety_tfidf.npy')

project_name = "S"
net_id = "...."

@irsystem.route('/', methods=['GET'])
def search():
	#query = request.args.get('search')
	WineType = request.args.get('flavor')
	MinPrice = request.args.get('MinPrice')
	MaxPrice = request.args.get('MaxPrice')
	color  = request.args.get('color')
	Flavor_Additional= request.args.get('Flavor_1')
	#print(MinPrice)
	#print(WineType)
	if not MinPrice:
		MinPrice = 0
	if not MaxPrice:
		MaxPrice = 3300
	if not color:
		color = None
	#else:
	print(MinPrice)
	print(MaxPrice)

	price = 0
	sim = 0
	if not WineType and not Flavor_Additional:
		data = []
		output_message = "Please enter a flavour"
	else:
		if not WineType:
			WineType = ""
		if not Flavor_Additional:
			Flavor_Additional = ""
		WineType = WineType + " " +Flavor_Additional
		print(WineType)

	## comment to update code
		master_data = pd.read_pickle(master_data_path)
		vin_tfidf = np.load(vin_tfidf_path).item()
		word_to_index = np.load(word_to_index_path).item()
		#wine_mask = np.load('D:\\Cornell_Acads\\Second_Semester\\CS4300_Flask_template-master\\app\\irsystem\\controllers\\wine_mask.npy')


		def query_vec(s):

			s = s.lower()
			s = re.sub(r'[^a-zA-Z0-9\s]', ' ', s)
			tokens = [token for token in s.split(" ") if token != ""]
			query_token = tokens
			query_vec = np.zeros((1,len(word_to_index)))
			for i,v in enumerate(query_token):
				if v in word_to_index:
					query_vec[0,word_to_index[v]] = 1

			return query_vec


		def similar_vin(query_vec, vin_tfidf):

			subset = []
			for i,v in enumerate(query_vec[0,:]):
				if v != 0:
					subset.append(i)

			tfidf = vin_tfidf[:,subset]
			query_vec = query_vec[0,subset]

			sims = np.dot(tfidf.toarray(),query_vec.T)
			sim_mat = pd.DataFrame(sims)
			sim_mat.columns = ['sims']
			sim_mat['index'] = np.arange(len(sim_mat))

			return sim_mat


		def shortlist_vin(query_vec, color = None, price_min = None, price_max = None):
			sim_mat = similar_vin(query_vec,vin_tfidf)
			recco_mat = pd.merge(sim_mat, master_data, on='index', how='left')

			if color != None:
				recco_mat =  recco_mat[recco_mat.Color == color]

			if price_min != None and price_max != None:
				recco_mat = recco_mat[recco_mat.price < price_max]
				recco_mat = recco_mat[recco_mat.price > price_min]

			recco_mat.sort_values("sims", axis = 0, ascending = False,
						 inplace = True)

			return recco_mat[0:10]



		if color  ==None:
			output_message = "Showing Results for "+ WineType + " Wine"
		else:
			output_message = "Showing Results for " + color + " " + WineType + " Wine"

	#data = "You want " + query + " Wine " + " in " + price

	#print(WineType)



		query = query_vec(WineType)
		shortlist = shortlist_vin(query, color = color, price_min = int(MinPrice), price_max = int(MaxPrice))


		data= shortlist[['title','price','sims','features']][:10]
		lst = []
		#for i in range(10):
	#		lst.append(data.features[i])
	#	list_of_path = WineCloud(lst)
	#	data['path'] = list_of_path
		data = data.values.tolist()
		price = shortlist['price'][:10].values.tolist()
		sim = shortlist['sims'][:10].values.tolist()


	return render_template('search.html', name=project_name, netid=net_id, output_message=output_message, data = data)
