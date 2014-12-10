import csv
from itertools import * 

with open(r"Train_Sample.csv") as r, open(r"output_file.csv", "w") as w:
     rdr = csv.reader(r)
     for row in rdr:
          a=row[1]
          b=row[3]
          for x, y in product(a.split(), b.split()):
               w.write("{},{}\n".format(x, y))

counter={}
with open("output_file.csv",'rb') as file_name:
     reader=csv.reader(file_name)
     for row in reader:
          pair=row[0]+' '+row[1]
          if pair in counter:
               counter[pair]+=1
          else:
               counter[pair]=1

