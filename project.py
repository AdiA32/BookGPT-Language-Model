# project.py


import pandas as pd
import numpy as np
import os
import re
import requests
import time


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def get_book(url):
    response = requests.get(url)

    time.sleep(5)
    book_text = response.text.replace("\r\n", "\n")

    start_comment = "*** START OF THIS PROJECT GUTENBERG EBOOK BEOWULF ***"
    end_comment = "*** END OF THIS PROJECT GUTENBERG EBOOK BEOWULF ***"

    start_index = book_text.find(start_comment) + len(start_comment)
    end_index = book_text.find(end_comment)

    book_content = book_text[start_index:end_index]

    return book_content


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def tokenize(book_string):
    book_string = book_string.strip()
    book_string = re.sub("\n{2,}", " \x03 \x02 ", book_string)
    tokens = re.findall("\w+|[^\w\s]", book_string)
    tokens = ["\x02"] + tokens + ["\x03"]

    return tokens


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


class UniformLM(object):
    def __init__(self, tokens):
        self.mdl = self.train(tokens)

    def train(self, tokens):
        unique = set(tokens)
        num_unique = len(unique)
        token_probs = pd.Series(1 / num_unique, index=unique)
        return token_probs

    def probability(self, words):
        prob = 1
        for word in words:
            if word in self.mdl:
                prob *= self.mdl[word]
            else:
                return 0
        return prob

    def sample(self, M):
        return " ".join(np.random.choice(self.mdl.index, size=M, p=self.mdl.values))


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


class UnigramLM(object):
    def __init__(self, tokens):
        self.mdl = self.train(tokens)

    def train(self, tokens):
        ser = pd.Series(tokens)
        res = ser.value_counts(normalize=True)
        return res

    def probability(self, words):
        prob = 1
        for word in words:
            if word in self.mdl:
                prob *= self.mdl[word]
            else:
                return 0
        return prob

    def sample(self, M):
        return " ".join(np.random.choice(self.mdl.index, size=M, p=self.mdl.values))


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


class NGramLM(object):

    def __init__(self, N, tokens):
        # You don't need to edit the constructor,
        # but you should understand how it works!
        
        self.N = N

        ngrams = self.create_ngrams(tokens)

        self.ngrams = ngrams
        self.mdl = self.train(ngrams)

        if N < 2:
            raise Exception("N must be greater than 1")
        elif N == 2:
            self.prev_mdl = UnigramLM(tokens)
        else:
            self.prev_mdl = NGramLM(N - 1, tokens)

    def create_ngrams(self, tokens):
        ngrams = []
        for i in range(len(tokens) - self.N + 1):
            ngram = tuple(tokens[i:i+self.N])
            ngrams.append(ngram)
        return ngrams


    def train(self, ngrams):
        ngram_series = pd.Series(ngrams)
        ngram_counts = ngram_series.value_counts()

        n1gram_counts = pd.Series([gram[:-1] for gram in ngrams]).value_counts()

        df = pd.DataFrame({
            'ngram': ngram_counts.index,
            'n1gram': [gram[:-1] for gram in ngram_counts.index],
            'counts': ngram_counts.values
        })

        df['n1gram_counts'] = df['n1gram'].apply(lambda x: n1gram_counts[x])
        df['prob'] = df['counts'] / df['n1gram_counts']

        df = df[['ngram', 'n1gram', 'prob']]

        return df

    

    def probability(self, words):
        if self.N == 1:
            prob = self.mdl.probability(words[:1])
        else:
            prob = self.prev_mdl.probability(words[:self.N-1])

        for i in range(self.N-1, len(words)):
            ngram = tuple(words[i-self.N+1:i+1])
            ngram_rows = self.mdl[self.mdl['ngram'] == ngram]
            
            if not ngram_rows.empty:
                prob *= ngram_rows['prob'].values[0]
            else:
                return 0 

        return prob



    def sample(self, M):
        """
        Generate a random sentence of length M
        """
        # Initialize output with the START token
        sentence = ['\x02']

        while len(sentence) < M + 1:  # we don't count the first '\x02' token in the M tokens
            # Get the most recent (n-1) tokens
            n_1gram = tuple(sentence[-(self.N-1):])

            # Debugging print statements
            print("self.mdl:", self.mdl)
            print("n_1gram:", n_1gram)

            # Get the options for the next token
            options_df = self.mdl[self.mdl['n1gram'] == n_1gram]

            if options_df.empty:
                # If there are no options, add a STOP token
                sentence.append('\x03')
            else:
                # Select a token based on its probability
                next_token = np.random.choice(options_df['ngram'].str[-1], p=options_df['prob'])
                sentence.append(next_token)

        # Join the tokens with spaces and return the string
        return ' '.join(sentence)

    def sample(self, M):
        """
        Generate a random sentence of length M
        """
        # Initialize output with the START token
        sentence = ['\x02']

        while len(sentence) < M:
            # Get the most recent (n-1) tokens
            n_1gram = tuple(sentence[-(self.N-1):])

            # Get the options for the next token
            options_df = self.mdl[self.mdl['n1gram'] == n_1gram]

            if options_df.empty:
                # If there are no options, break the loop
                break
            else:
                # Select a token based on its probability
                next_token = np.random.choice(options_df['ngram'].str[-1], p=options_df['prob'])
                sentence.append(next_token)

        # Join the tokens with spaces and return the string
        return ' '.join(sentence)


