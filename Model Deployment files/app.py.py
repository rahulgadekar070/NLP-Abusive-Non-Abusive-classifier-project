from flask import Flask,render_template,request,send_file
import pandas as pd
import re 
import seaborn as sns; sns.set_style('whitegrid')
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
import seaborn as sns; sns.set_style('whitegrid')
from nltk.tokenize import RegexpTokenizer
import nltk

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
stop_words = set(stopwords.words("english"))
default_stemmer = PorterStemmer()
default_stopwords = stopwords.words('english')
default_tokenizer=RegexpTokenizer(r"\w+")

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    df = pd.read_csv('file:///C:/Users/admin/rg_deploy/train.csv')
    df = df.drop(['Unnamed: 0', 'filename', 'Message-ID'], axis = 1)
    df['label'] = df['Class'].map({'Non Abusive': 0, 'Abusive': 1})
    
    stopword = nltk.corpus.stopwords.words('english')
    unword = ['http','https','html','excelr','re','cc','FW', 'ga','ha','www','will','final' ,'cc', 'aa', 'aaa', 'aaaa','hou', 'cc', 'etc', 'subject', 'pm']
    stopword.extend(unword)
    def clean_text(text, ):
        if text is not None:
        #exclusions = ['RE:', 'Re:', 're:']
        #exclusions = '|'.join(exclusions)
                text = re.sub(r'[0-9]+','',str(text))
                text =  text.lower()
                text = re.sub('re:', '', text)
                text = re.sub('-', '', text)
                text = re.sub('_', '', text)
                text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
                text = re.sub(r'\S*@\S*\s?', '', text, flags=re.MULTILINE)
        # Remove text between square brackets
                text =re.sub('\[[^]]*\]', '', text)
        # removes punctuation
                text = re.sub(r'[^\w\s]','',text)
                text = re.sub(r'\n',' ',text)
                text = re.sub(r'[0-9]+','',text)
                #text = re.sub(r'[0-9]+','',text)
        # strip html 
                p = re.compile(r'<.*?>')
                text = re.sub(r"\'ve", " have ", text)
                text = re.sub(r"can't", "cannot ", text)
                text = re.sub(r"n't", " not ", text)
                text = re.sub(r"I'm", "I am", text)
                text = re.sub(r" m ", " am ", text)
                text = re.sub(r"\'re", " are ", text)
                text = re.sub(r"\'d", " would ", text)
                text = re.sub(r"\'ll", " will ", text)
        
                text = p.sub('', text)

        def tokenize_text(text,tokenizer=default_tokenizer):
            token = default_tokenizer.tokenize(text)
            return token
        
        def remove_stopwords(text, stop_words=stopword):
            tokens = [w for w in tokenize_text(text) if w not in stop_words]
            return ' '.join(tokens)

        def stem_text(text, stemmer=default_stemmer):
            tokens = tokenize_text(text)
            return ' '.join([stemmer.stem(t) for t in tokens])

        text = stem_text(text) # stemming
        text = remove_stopwords(text) # remove stopwords
        #text.strip(' ') # strip whitespaces again?

        return text
    
    rfc = open('rfc.pickle','rb')
    cv = open('cv.pickle','rb')
    rfc = joblib.load(rfc)
    cv = joblib.load(cv)
   
    '''X = df['content']
    y = df['label']
    cv = TfidfVectorizer(max_features = 10000)
    X = cv.fit_transform(X) # Fit the Data
    #Naive Bayes Classifier
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
    rfc = RandomForestClassifier(n_estimators=100,random_state=252)
    rfc.fit(X_train,y_train)
    rfc.score(X_test,y_test)
    joblib.dump(rfc, 'emails_model.pickle')'''
    
    
    if request.method == 'POST':
        message = request.form['message']
        if message:
            
            data = pd.DataFrame([message],columns=["content"])
            df_content = data['content'].apply(clean_text)
            vect = cv.transform(df_content).toarray()
            prediction = rfc.predict(vect)
            data['predicted']=prediction
            data["predicted"] = data["predicted"].map({0:'Non Abusive', 1:'Abusive'})
            return render_template('result.html',tables=[data.to_html()], titles=data.columns.values)
        
        
        else:
                csv_file = request.files['csv-file']
                data = pd.read_csv(csv_file,usecols=["content"])
                df_content = data["content"].apply(clean_text)
                vect = cv.transform(df_content).toarray()
                model = rfc.predict(vect)
                data["predicted"]=model
                data["predicted"] = data["predicted"].map({0:'Non Abusive', 1:'Abusive'})
                data.to_csv('test.csv')
                return send_file('test.csv')
                


if __name__ == '__main__':
    app.run(debug=True)