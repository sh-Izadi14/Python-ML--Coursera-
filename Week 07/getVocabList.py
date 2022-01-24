def getVocabList():
#GETVOCABLIST reads the fixed vocabulary list in vocab.txt and returns a
#cell array of the words
#   vocabList = GETVOCABLIST() reads the fixed vocabulary list in vocab.txt 
#   and returns a cell array of the words in vocabList.


      ## Read the fixed vocabulary list
      fid = open("vocab.txt","r").read()

      # Store all dictionary words in cell array vocab{}
      n = 1899;  # Total number of words in the dictionary

      # For ease of implementation, we use a struct to map the strings => integers
      # In practice, you'll want to use some form of hashmap
      fid = fid.split("\n")

      vocabList={}
      for i in fid:
            value,key = i.split("\t")
            vocabList[key] = int(value)-1
      
      return vocabList