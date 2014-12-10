from nltk.corpus import stopwords
import time
import timeit

cachedStopWords = stopwords.words("english")

def testFuncOld():
	text = 'hello bye the the hi'
	text = ' '.join([word for word in text.split() if word not in stopwords.words("english")])

def testFuncNew():
        text = 'hello bye the the hi'
        text = ' '.join([word for word in text.split() if word not in cachedStopWords])

def testFuncNew2():
        text = 'hello bye the the hi'
        text = ' '.join(word for word in text.split() if word not in cachedStopWords)

if __name__ == "__main__":
	'''
	t0 = time.time()
        for i in xrange(10000):
        	testFuncOld()
	t1 = time.time()
	print "old", t1-t0
	t0 = time.time()
        for i in xrange(10000):
                testFuncNew()
        t1 = time.time()
        print "new", t1-t0
        '''
	t = timeit.timeit('testFuncOld()','from __main__ import testFuncOld' ,number=10000)
	print t
	t = timeit.timeit('testFuncNew()','from __main__ import testFuncNew' ,number=10000)
        print t
	t = timeit.timeit('testFuncNew2()','from __main__ import testFuncNew2' ,number=10000)
        print t
