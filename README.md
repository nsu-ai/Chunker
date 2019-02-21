# Chunker

This is the module for using the pre-trained syntax analysis system for the russian language. 
It is based on Conditional Random Fields and morpho-tagger from [deeppavlov](http://docs.deeppavlov.ai/en/master/components/morphotagger.html) 

This module provides two ways of chunking: for txt and for the sentences
You can download the model [here](https://drive.google.com/open?id=1OJwqK4wu-ZoDvnTWYCov7Q1VveuI3T6Y)
```
f = Chunker('path to the crf model')
```
For txt use the function:
```
chunked_text = f.predict_file("myfile.txt")
```
For the sentence use:
```
chunked_sentence = f.predict_sentence("Какие красивые деревья!")
```
