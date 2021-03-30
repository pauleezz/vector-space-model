# WSM Project: Ranking by Vector Space Models

A retrieval program that is able to retrieve the relevant news to the given query from a set of 7,034 English News collected from reuters.com according to different weighting schemes and similarity metrics. In the given dataset, each file is named by its News ID and contains the corresponding news title and content.

## Implementation

`$ python main.py --query {query}`

Here's an example:

`$ python main.py --query="Trump Biden Taiwan China"`

After entering the query, it will output five different combinations which retrieve the top 5 results and score, each combination is listed below:

1. Term Frequency (TF) Weighting + Cosine Similarity
2. Term Frequency (TF) Weighting + Euclidean Distance
3. TF-IDF Weighting + Cosine Similarity
4. TF-IDF Weighting + Euclidean Distance
5. Relevance Feedback + TF-IDF Weighting + Cosine Similarity

The output would be:

## File Structure

```
    vector-space-model
    |-- EnglishNews
    |-- english.stop
    |-- main.py
    |-- Parser.py
    |-- PorterStemmer.py
    |-- README.md
    |-- util.py
    |-- VectorSpace.py
```
