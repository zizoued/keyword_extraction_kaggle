# assume csv fil with tags and predicted
# make overall f1 score the average of all f1 scores
import csv

counter = 0
total_f1 = 0.0

with open("Data/results_20k2_onlycon.csv", "rb") as r:
	rdr = csv.reader(r)
	for row in rdr:
		tags = row[1]
		predicted = row[2]

		tags = set(tags)
		predicted = set(predicted)
		tp = len(tags & predicted)
		fp = len(predicted) - tp
		fn = len(tags) - tp
		'''
		print tags
		print predicted
		print "tp", tp
		print "fp", fp
		print "fn", fn
		'''
		if tp>0:
    			precision=float(tp)/(tp+fp)
    			recall=float(tp)/(tp+fn)
    			#print "precision", precision
    			#print "recall", recall
    			f1 = 2*((precision*recall)/(precision+recall))
		else:
    			f1 = 0.0
		
		total_f1 += f1
		counter += 1	

print "F1-score on average is", total_f1/counter
