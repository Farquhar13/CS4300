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
from numpy import linalg as la

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
	WineType = request.args.getlist('flavor[]')
	WineType = ' '.join(WineType)
	MaxPrice = request.args.get('MaxPrice')
	color  = request.args.get('color')
	Flavor_Additional= request.args.get('Flavor_1')
	Variety = request.args.get('variety')
	rating = request.args.get('rating')
	varietyfilter = request.args.get('varietyfilter')
	Country = request.args.get('Country')
	#print(MinPrice)
	#print(WineType)
	flag = []
	flag3 =[]

	if not color:
		color = None
	#else:


	if not varietyfilter:
		varietyfilter = 0
	if not Country:
		Country = None
	price = 0
	sim = 0
	if not WineType and not Flavor_Additional and not Variety and not color and not Country and not MaxPrice and not rating:
		flag3 = [1]
		data = []
		output_message = "Please enter a flavor"

	else:
		if not WineType:
			WineType = ""
		if not Flavor_Additional:
			Flavor_Additional = ""
		WineType = WineType + " " +Flavor_Additional
		if not Variety:
			Variety = ""
		#print(WineType)
		if not rating:
			rating = 0

		if not MaxPrice:
			MaxPrice = 3300
		master_data = pd.read_pickle(master_data_path)
		vin_tfidf = np.load(vin_tfidf_path ).item()
		word_to_index = np.load(word_to_index_path).item()
		variety_tfidf = np.load(variety_tfidf_path).item()

	## comment to update code
	#	master_data = pd.read_pickle("D:\\Cornell_Acads\\Second_Semester\\CS4300_Flask_template-master\\app\\irsystem\\controllers\\master_data_lean.pkl")
	#	vin_tfidf = np.load('D:\\Cornell_Acads\\Second_Semester\\CS4300_Flask_template-master\\app\\irsystem\\controllers\\vin_tdidf.npy').item()
	#	word_to_index = np.load('D:\\Cornell_Acads\\Second_Semester\\CS4300_Flask_template-master\\app\\irsystem\\controllers\\word_to_index.npy').item()
	#	variety_tfidf = np.load('D:\\Cornell_Acads\\Second_Semester\\CS4300_Flask_template-master\\app\\irsystem\\controllers\\variety_tfidf.npy').item()


		def query_vec(s):

			s = s.lower()
			s = re.sub(r'[^a-zA-Z0-9\s]', ' ', s)
			tokens = [token for token in s.split(" ") if token != ""]
			query_token = tokens
			query_vec = np.zeros((1,len(word_to_index)))
			for i,v in enumerate(query_token):
				if v in word_to_index:
					query_vec[0,word_to_index[v]] = 0.5

			return query_vec


		def similar_vin(query_vec, vin_tfidf):

			subset = []
			for i,v in enumerate(query_vec[0,:]):
				if v != 0:
					subset.append(i)

			tfidf = vin_tfidf[:,subset]
			query_vec = query_vec[0,subset]
			query_vec = query_vec/la.norm(query_vec,ord=2)

			sims = np.dot(tfidf.toarray(),query_vec.T)
			sim_mat = pd.DataFrame(sims)
			sim_mat.columns = ['sims']
			sim_mat['index'] = np.arange(len(sim_mat))

			return sim_mat


		def shortlist_vin(query_vec, color = None, price_min = None, price_max = None, rating = None,
                  variety = None, variety_flag =0, country = None):
					sim_mat = similar_vin(query_vec,vin_tfidf)
					recco_mat = pd.merge(sim_mat, master_data, on='index', how='left')

					if color != None:
						recco_mat =  recco_mat[recco_mat.Color == color]

					if price_max != None:
						recco_mat = recco_mat[recco_mat.price < price_max]

					if rating != None:
						   recco_mat = recco_mat[recco_mat.points > rating]

					if variety_flag == 1 and (variety in list(variety_tfidf.keys())):
						 recco_mat = recco_mat[recco_mat['variety']==variety]

					if country in ['Spain', 'US', 'New Zealand', 'France', 'Israel', 'Australia',
					   'Portugal', 'Argentina', 'South Africa', 'Italy', 'Germany',
					   'Greece', 'Moldova', 'Croatia', 'Serbia', 'Chile', 'Austria',
					   'Georgia', 'Uruguay', 'Hungary', 'Ukraine', 'Brazil',
					   'Slovenia', 'Canada', 'Morocco', 'Bulgaria', 'England',
					   'Macedonia', 'Romania', 'Mexico', 'Turkey', 'China', 'Cyprus',
					   'Slovakia', 'Lebanon', 'Switzerland', 'Luxembourg','Peru',
					   'Egypt', 'Czech Republic', 'India', 'Armenia',
					   'Bosnia and Herzegovina']:
						recco_mat = recco_mat[recco_mat['country']==country]

					recco_mat.sort_values("sims", axis = 0, ascending = False,
								 inplace = True)
					if np.sum(query_vec) == 0:
						recco_mat.sort_values("points", axis = 0, ascending = False,inplace = True)

					return recco_mat[0:10]

	#	def WineCloud (df):
		output_message  =  "Showing Wine"
		if color is not None:
					output_message = output_message + " with " + color + " Color" + " "
		if WineType != " ":
			output_message = output_message + " with " + WineType + " flavour "
		if Variety != "":
			output_message = output_message + " like " + Variety
		if Country is not None:
			output_message = output_message + " from " + Country
		if rating != 0 :
			output_message = output_message + " with rating greater than " + str(rating)
		if MaxPrice!= 3300:
			output_message =  output_message + " with price less than $" + MaxPrice




		#if color  ==None:
		#	output_message = "Showing Results for "+ WineType + " Wine"
		#else:
		#	output_message = "Showing Results for " + color + " " + WineType + " Wine"
		#if WineType == " " :
		#	output_message = "Showing Results for " + Variety + " Wine"
		#elif color is not None and Variety is not None:
		#	output_message = "Showing Results for " + color + " " + Variety + " "+ WineType + " Wine"
		#elif Variety is not None:
		#	output_message = "Showing Results for "  + Variety + " "+ WineType + " Wine"

		#print ("WineType is " + WineType)


	#data = "You want " + query + " Wine " + " in " + price

	#print(WineType)
		#print(varietyfilter)
		#print(Variety)

		if Variety in list(variety_tfidf.keys()):
			query = query_vec(WineType) + variety_tfidf[Variety].toarray()
		else:
			query = query_vec(WineType)

		shortlist = shortlist_vin(query, color = color, price_max = int(MaxPrice), rating = int(rating),variety = Variety,variety_flag = int(varietyfilter), country = Country)

		data= shortlist[['title','price','sims','description','points','country','variety']][:10]
		data['sims'] = data['sims'].apply(lambda x:round(x,3))


	#	list_of_path = WineCloud(data)
	#	data['path'] = list_of_path
		data = data.values.tolist()
		price = shortlist['price'][:10].values.tolist()
		sim = shortlist['sims'][:10].values.tolist()

		if (len(data)== 0):
			flag = [1]


	return render_template('search.html', name=project_name, netid=net_id, output_message=output_message, data = data, flag = flag,flag3 = flag3)
