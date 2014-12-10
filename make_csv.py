import csv
from itertools import islice
from bs4 import BeautifulSoup
from nltk.corpus import stopwords

cachedStopWords = stopwords.words("english")
cachedStopWords.append('what')
cachedStopWords.append('why')
cachedStopWords.append('how')

def line_stream(filename, stop=None):
	"""Streams `filename` one line at a time, optionally stopping at `stop`"""
	with open(filename) as csvfile:
		next(csvfile, None)  # skip header
		for line in islice(csv.reader(csvfile, delimiter=',', quotechar='"'), stop):
			yield line

def main():
	"""Make a CSV file out of training data of 1GB"""

	filename = "Data/train-aa"
	title_dict = {} # will use this to remove duplicates
	csv_data = []
	id_counter = 1
	for line in line_stream(filename):
        	ID, title, body, tags = line
		if id_counter>20000 : # change this for data points
			break	
		title = title.lower()
		# check if title is duplicate, if yes, ignore
		if title in title_dict.keys() :
			continue
		title_dict[title]=1

		body = body.lower()
		# remove all code, and extract text from body (i.e. remove html tags)
		soup = BeautifulSoup(body)
		[s.extract() for s in soup('code','a')]	
		# also remove all links
		
		body = soup.get_text()

		body = body.encode('ascii', 'ignore')
		title = title.decode('utf-8').encode('ascii','ignore')
		
		title = ' '.join([word for word in title.split() if word not in cachedStopWords])
		body = ' '.join([word for word in body.split() if word not in cachedStopWords])		

		this_line = [id_counter, title, body, tags]
		id_counter += 1
		csv_data.append(this_line)

	with open("Data/train_nostop_20k2.csv", "wb") as out_file:
		writer = csv.writer(out_file, delimiter=',')
		for row in csv_data:
			writer.writerow(row)

if __name__=="__main__" :
	main()
