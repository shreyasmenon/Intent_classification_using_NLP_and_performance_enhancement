def tokenize_lemmatize(df_queries, use_stopwords = True , extract_entities = True):
    
    list_preprocessed = []
    stop_words = stopwords.words('english')
    
    #Add relevant words
    stop_words.append('https')
    stop_words.append('http')
    stop_words.append('thank')
    stop_words.append('thanks')
    stop_words.append('optimumhelp')
    stop_words.append('optimum')
    stop_words.append('hi')
    stop_words.append('dm')
    stop_words.append('ok')
    stop_words.append('okay')
    stop_words.append('twigmg')
    stop_words.append('pbs')
    stop_words.append('jpg')
    stop_words.append('com')
    
    
    #Remove words from list
    stop_words.remove('not')
    stop_words.remove('nor')
    stop_words.remove('no')
    stop_words.remove('ain')
    stop_words.remove("aren't")
    stop_words.remove("couldn't")    
    #stop_words.remove("couldn'")             
    #stop_words.remove("didn'")
    #stop_words.remove("doesn'")
    stop_words.remove("doesn't")
    #stop_words.remove("don'")
    stop_words.remove("don't")
    stop_words.remove("hadn")
    stop_words.remove("hadn't")
    stop_words.remove("hasn")
    stop_words.remove("hasn't")
    stop_words.remove("haven")
    stop_words.remove("haven't")
    #stop_words.remove("isn'")
    stop_words.remove("isn't")    
    stop_words.remove("mightn")
    stop_words.remove("mightn't")
    stop_words.remove("mustn")
    stop_words.remove("mustn't")
    stop_words.remove("needn't")
    stop_words.remove("off")
    stop_words.remove("shan't")
    #stop_words.remove("shan'")
    stop_words.remove("should've")
    #stop_words.remove("shouldn'")
    stop_words.remove("shouldn't")
    #stop_words.remove("wasn''")    
    stop_words.remove("wasn't")
    #stop_words.remove("weren'")   
    stop_words.remove("weren't")    
    stop_words.remove("will")
    #stop_words.remove("wouldn'")
    #stop_words.remove("won'")
    stop_words.remove("won't")
    #stop_words.remove("wouldn'")
    stop_words.remove("wouldn't")
    stop_words.remove("down")
    
    printable = set(string.printable)

    
    for response in df_queries:
        #sample_response = response
        #keep only unicode characters    
        response = ''.join(filter(lambda x: x in printable, response))
        
        if extract_entities:
            response = extract_metainfo(response)
         
        tokeinzed_response = nltk.word_tokenize(response)
        tokeinzed_response_lower = [token.lower() for token in tokeinzed_response if len(token) > 1]
        tokenized_without_punctuations =[token.strip(string.punctuation) for token in tokeinzed_response_lower]
        tokenized_without_punctuations_and_spaces = [token.strip() for token in tokenized_without_punctuations if token.strip()!='']
        tagged_tokens= nltk.pos_tag(tokenized_without_punctuations_and_spaces)
        
        wordnet_lemmatizer = WordNetLemmatizer()
        lemmatized_words = [wordnet_lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for (word, tag) in tagged_tokens if word not in string.punctuation]       
        lemmatized_string = " ".join(lemmatized_words)
        
        if len(lemmatized_string) > 0:
            list_preprocessed.append(lemmatized_string)
        else:
            list_preprocessed.append('Empty String')
