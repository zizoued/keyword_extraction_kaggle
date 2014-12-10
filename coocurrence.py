import csv
from itertools import product
import operator
import cPickle as pickle

def make_coocurr_pairs():
	with open(r"Data/train_nostop_20k2.csv") as r, open(r"Data/comb_titletag_20k2.csv", "w") as w:
		rdr = csv.reader(r)
		for row in rdr:
        		a=row[1]
        		b=row[3]
			#print a,b
			for x, y in product(a.split(), b.split()):
                		w.write("{},{}\n".format(x, y))


freq = {}
'''
def make_freq_dict():
	with open(r"Data/train_nostop_20k2.csv") as r:
		rdr = csv.reader(r)
		for row in rdr:
			a = row[1] # title of post
			words = a.split()
			for w in words:
				if w in freq:
					freq[w]+=1
				else:
					freq[w]=1

'''
counter = {}
def make_count_dict():

	with open("Data/comb_titletag_20k2.csv",'rb') as file_name:
		reader=csv.reader(file_name)
		for row in reader:
        		pair=row[0]+' '+row[1]
			word = row[0]
        		if pair in counter:
                		counter[pair]+=1
          		else:
                		counter[pair]=1

			if word in freq:
				freq[word]+=1
			else:
				freq[word]=1

def make_conf_supp_data():
	data = []
	for item in counter.keys():
		newlist = []
		word_tag = item.split()
		
		if len(word_tag)>1 and len(word_tag[0])>1 and word_tag[0] in freq :
			word = word_tag[0]
			tag = word_tag[1]
			support = counter[item]
			confidence = float(support)/freq[word]
			newlist.append(word)
			newlist.append(tag)
			newlist.append(support)
			newlist.append(confidence)
			data.append(newlist)
	
	with open("Data/conf_supp_20k2.csv","wb") as w:
		writer = csv.writer(w, delimiter=',')
		for row in data:
			writer.writerow(row)

def pickle_conf_supp():
	conf_supp = {}
	with open("Data/conf_supp_20k2.csv",'rb') as file_name:
                reader=csv.reader(file_name)
                for row in reader:
			#print row	
			word = row[0]
			wlist = [row[1],row[2],row[3]]
			if word in conf_supp:
				conf_supp[word].append(wlist)
			else:
				conf_supp[word]=[]
				conf_supp[word].append(wlist)			
	pickle.dump(conf_supp, open("Data/conf_supp_20k2.p", "wb"))
	print conf_supp	

def get_top_10_words(tag):
	""" get top 10 words associated with a tag """
	tag_dict = {}
	for item in counter.keys():
		word_tag = item.split()
		#print word_tag
		if len(word_tag)>1 and tag in word_tag[1] and len(word_tag[0])>1:
			if word_tag[0] in tag_dict:
				tag_dict[word_tag[0]]+=1
			else:
				tag_dict[word_tag[0]]=1 
	sorted_dict = sorted(tag_dict.items(), key=operator.itemgetter(1), reverse=True)
	
	print "----------- ",tag," -----------"
	for i in range(0,10):
		print sorted_dict[i]
	print "--------------------------------"

def main():
	#make_coocurr_pairs()
	make_count_dict()
	#get_top_10_words("php")
	#get_top_10_words("sql")
	#get_top_10_words("java")
	#get_top_10_words("c++")
	#make_freq_dict()
	make_conf_supp_data()
	pickle_conf_supp()	

if __name__=="__main__" :
        main()

