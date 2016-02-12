import csv
boundary = list(csv.reader(open("../KDD/thesaurus_boundary_test.txt"), delimiter="\t"))
category= list(csv.reader(open("../KDD/thesaurus_category_test.txt"), delimiter="\t"))
hybrid = []
i = 0
for c in category[:-1]:
    for b in boundary[:-1]:
        hybrid.append(("%s-%s" % (b[0], c[0]), i))
        i+=1
b, c = boundary[-1], category[-1]
hybrid.append(("%s-%s" % (b[0], c[0]), i))
with open("thesaurus_hybrid.txt", "wb+") as fp:
    for row in hybrid:
        print >> fp, "%s\t%s" % row
