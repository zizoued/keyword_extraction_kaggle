import csv
from itertools import product
import operator
import cPickle as pickle
import random
import time

def make_coocurr_pairs(data):
	with open(r"Data/comb_titletag_20k2.csv", "w") as w:
		for row in data:
        		a=row[1]
        		b=row[3]
			#print a,b
			for x, y in product(a.split(), b.split()):
                		w.write("{},{}\n".format(x, y))


def make_train_test(test_percent):
	t1 = time.time()
	data = []
	with open(r"Data/train_nostop_20k2.csv") as r:
		rdr = csv.reader(r)
		for row in rdr:
			data.append(row)
	
	random.shuffle(data)
	total = len(data)
	print total, "data points total"
	test = float(test_percent)/100*total
	print test, "test data points"
	test_data = data[:int(test)]
	train_data = data[int(test):]
	print "length of test data", len(test_data)
	print "length of train data", len(train_data)
	t2 = time.time()
	print "time taken to make train/test split randomly", t2-t1
	return test_data, train_data

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
	#print conf_supp	

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

def make_predictions(alpha, beta):
	""" Assuming that the confidence and support data has been calculated from the training set, and pickled, and the testing data has been pickled, this function now makes tag predictions and writes them out into a csv file"""

	conf_supp = pickle.load(open("Data/conf_supp_20k2.p", "rb"))
	test_data = pickle.load(open("Data/test_data_20k2.p", "rb"))

	results = []
	
	for row in test_data :
		title = row[1]
		corr_tags = row[3]
		pred_tags = ''
		for word in title.split():
			if word in conf_supp:
				rules = conf_supp[word] 
				# each rule is a list of tag,supp,conf
				for r in rules:
					if r[1]>beta and r[2]>alpha and r[0] not in pred_tags.split():
						pred_tags+=' '
						pred_tags+=r[0]

		thisrow = [title, corr_tags, pred_tags]
		results.append(thisrow)

	with open("Data/results_20k2.csv","wb") as w:
                writer = csv.writer(w, delimiter=',')
                for row in results:
                        writer.writerow(row)

def make_predictions_2():
        """ Assuming that the confidence and support data has been calculated from the training set, and pickled, and the testing data has been pickled, this function now makes tag predictions and writes them out into a csv file"""
	"""This method only uses confidence i.e. conditional probability, and simply outputs the tags having the highest conditional probabilities, compared for all the words in the title"""

        conf_supp = pickle.load(open("Data/conf_supp_20k2.p", "rb"))
        test_data = pickle.load(open("Data/test_data_20k2.p", "rb"))

        results = []

        for row in test_data :
                title = row[1]
                corr_tags = row[3]
                pred_tags = ''
		all_rules = []
                for word in title.split():
                        if word in conf_supp:
                                rules = conf_supp[word]
                                # each rule is a list of tag,supp,conf
				for r in rules :
                                	all_rules.append(r)
		
		#print all_rules
		#print "---"
		
		sorted_tags = sorted(all_rules, key=operator.itemgetter(2), reverse=True)
		#print sorted_tags
		
		'''
		if len(sorted_tags)>2:
                	pred_tags+=sorted_tags[0][0]+' '+sorted_tags[1][0]+' '+sorted_tags[2][0]
		elif len(sorted_tags)>1:
			pred_tags+=sorted_tags[0][0]+' '+sorted_tags[1][0]

		elif len(sorted_tags)>0:
			pred_tags+=sorted_tags[0][0]

		'''
		
		taglist = []
		lmax = min(len(sorted_tags),11)
		for i in range(0,lmax-1):
			
			if sorted_tags[i][0] not in taglist:
				taglist.append(sorted_tags[i][0])
			if len(taglist)>4 :
				break
		
		pred_tags = ' '.join(taglist)
		
                thisrow = [title, corr_tags, pred_tags]
                results.append(thisrow)

        with open("Data/results_20k2_onlycon.csv","wb") as w:
                writer = csv.writer(w, delimiter=',')
                for row in results:
                        writer.writerow(row)


def main():
	
	#make_freq_dict()
	'''	
	test_data, train_data = make_train_test(25)
	make_coocurr_pairs(train_data)
	make_count_dict()
	get_top_10_words("php")
        get_top_10_words("sql")
        get_top_10_words("java")
        get_top_10_words("c++")
	make_conf_supp_data()
        pickle_conf_supp() 
	pickle.dump(test_data, open("Data/test_data_20k2.p", "wb"))	
	'''
	#make_predictions(0.5,5)
	make_predictions_2()

if __name__=="__main__" :
        main()

