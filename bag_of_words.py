import nltk

paragraph = """Reed College at that time offered perhaps the best calligraphy instruction in the country. Throughout the campus every poster, every label on every drawer, was beautifully hand calligraphed. Because I had dropped out and didn’t have to take the normal classes, I decided to take a calligraphy class to learn how to do this. I learned about serif and san serif typefaces, about varying the amount of space between different letter combinations, about what makes great typography great. It was beautiful, historical, artistically subtle in a way that science can’t capture, and I found it fascinating.
None of this had even a hope of any practical application in my life. But ten years later, when we were designing the first Macintosh computer, it all came back to me. And we designed it all into the Mac. It was the first computer with beautiful typography. If I had never dropped in on that single course in college, the Mac would have never had multiple typefaces or proportionally spaced fonts. And since Windows just copied the Mac, it’s likely that no personal computer would have them. If I had never dropped out, I would have never dropped in on this calligraphy class, and personal computers might not have the wonderful typography that they do. Of course it was impossible to connect the dots looking forward when I was in college. But it was very, very clear looking backwards ten years later.
Again, you can’t connect the dots looking forward; you can only connect them looking backwards. So you have to trust that the dots will somehow connect in your future. You have to trust in something — your gut, destiny, life, karma, whatever. This approach has never let me down, and it has made all the difference in my life."""

#step 1: cleaning the text (stopwords, stemming, lemmatization, converting to lowercase)
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

ps = PorterStemmer()
wordnet = WordNetLemmatizer()
sentences = nltk.sent_tokenize(paragraph)

corpus = []
for i in range(len(sentences)):
    
    #replacing all characters expect alphabets with space in all the sentences
    review = re.sub('[^a-zA-Z]', ' ', sentences[i])
    #converting to lowercase
    review = review.lower()
    #splitting of a sentence will give a list of words
    review = review.split()
    #remove all the stop words, lemmatize each word and add the words to the list review
    review = [wordnet.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

#Creating Bag of Words model
#sklearn is the library in which we have data preprocessing modules
#Count Vectorizer is a data preprocessing library, responsible for creating bag of word document matrix
#importing & creating object for necessary library
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
#fit_transform helps to create the matrix
X = cv.fit_transform(corpus).toarray()