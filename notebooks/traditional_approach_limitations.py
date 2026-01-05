# Traditional approach: TF-IDF + Linear Regression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

# Problems:
# 1. Bag-of-words loses word order
# 2. Can't capture complex semantic relationships
# 3. Fixed vocabulary, no generalization
# 4. No attention to important words

