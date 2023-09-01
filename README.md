# FakeNewsDetection

1. **Importing Libraries**:
   - The code begins by importing necessary Python libraries, including Pandas for data manipulation, NumPy for numerical operations, regular expressions (re) for text processing, and various libraries for data visualization (Matplotlib and Seaborn).

2. **Loading Data**:
   - The `data.csv` file is read into a Pandas DataFrame called `data`.

3. **Exploratory Data Analysis (EDA)**:
   - Several EDA operations are performed:
     - `data.describe()`: Provides summary statistics of the dataset.
     - `data.shape`: Displays the shape of the dataset (number of rows and columns).
     - `data.info()`: Gives information about the dataset's data types and missing values.
     - `data.isna().sum()`: Counts the number of missing values in each column.
     - `data.dropna(inplace=True, axis=0)`: Removes rows with any missing values.

4. **Data Preprocessing**:
   - The code extracts the website name from the URLs in the 'URLs' column and stores it in a new 'Website' column. The 'URLs' column is then dropped.
   
5. **Data Visualization**:
   - Several data visualizations are created using Seaborn and Matplotlib. These visualizations include:
     - A countplot showing the distribution of fake and real news.
     - Bar plots displaying the top websites posting real and fake news.

6. **Text Cleaning**:
   - NLTK is used for text cleaning. The following text preprocessing steps are applied to the 'Text' column:
     - Convert text to lowercase.
     - Remove non-alphabetical characters.
     - Tokenization: Split text into words.
     - Lemmatization: Reduce words to their base form (excluding English stopwords).
     - The cleaned text is stored in the 'Text' column of the DataFrame.

7. **Word Clouds**:
   - Word clouds are generated separately for fake and real news using the WordCloud library. These word clouds visually represent the most common words in each category of news.

8. **Text Classification Model**:
   - A deep learning model for text classification is created using TensorFlow and Keras. The model consists of:
     - An embedding layer (word embeddings).
     - An LSTM layer (Long Short-Term Memory) for sequence processing.
     - A dense output layer with a sigmoid activation function.
   - The model is compiled with binary cross-entropy loss and the Adam optimizer.
   
9. **Text Data Preparation for Model**:
   - The text data is preprocessed for model input:
     - One-hot representation of words is generated (limited to 5000 words).
     - Sentences are padded or truncated to a fixed length of 400.
   
10. **Model Training**:
    - The model is trained using the preprocessed text data. It is fitted to the training data with 10 epochs and a batch size of 64.

11. **Model Evaluation**:
    - The model's performance is evaluated using various metrics:
      - A confusion matrix is plotted to visualize the model's classification results.
      - The accuracy of the model is calculated using `accuracy_score`.
    
