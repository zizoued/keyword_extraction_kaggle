# assume csv fil with tags and predicted
# make overall f1 score the average of all f1 scores

tags = ["c", "c++", "java"]
predicted = ["java", "python"]

tags = set(tags)
predicted = set(predicted)
tp = len(tags & predicted)
fp = len(predicted) - tp
fn = len(tags) - tp

print tags
print predicted
print "tp", tp
print "fp", fp
print "fn", fn

if tp>0:
    precision=float(tp)/(tp+fp)
    recall=float(tp)/(tp+fn)
    print "precision", precision
    print "recall", recall
    print 2*((precision*recall)/(precision+recall))
else:
    print 0
