import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


nltk.download('stopwords')
from nltk.corpus import stopwords

emails = [
    "Congratulations! You've won a $1000 gift card. Click to claim now.",
    "Team meeting at 10 AM tomorrow. Don't be late.",
    "Your account is suspended. Login immediately to restore access.",
    "Let's catch up over coffee this weekend.",
    "Win a free vacation! Limited time offer, click here!",
    "Project deadline has been extended to next Friday.",
    "You’ve been selected for a prize. Claim your reward now!"
]

labels = [
    1,  # spam
    0,  # ham
    1,
    0,
    1,
    0,
    1
]


def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)


emails_cleaned = [preprocess(email) for email in emails]


X_train, X_test, y_train, y_test = train_test_split(emails_cleaned, labels, test_size=0.3, random_state=42)

model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', MultinomialNB())
])


model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
