

def tfidf(document):
    pass
if __name__ == "__main__":
    handler = open("C:\\IntelligentDataAnalysis\\Document_mining\\vocab.txt")
    document = handler.read()
    document = str.split(document,"\n")
    print (type(document))
    print (document)