import simSCESentenceIterator
import numpy as np
import umap.umap_ as umap
import torch
import pandas as pd
from transformers import AutoModel, AutoTokenizer
import joblib, re
import plotly.express as px
from datetime import datetime

class model_handler:
    def __init__(self,
                 classifier_path,
                 doc_vec_path,
                 reference_df_path,
                 doc_vec_3D_path,
                 doc_vec_2D_path,
                 umap2D_path,
                 umap3D_path,
                 genre_ids_path,
                 country_ids_path,
                 time_ids_path,
                 genocide_ids_path,
                 tokenizer_path=None,
                 model_path=None,
                ):
        usecols = ['title', 'year', 'director', 'producer', 'duration', 'genocide', 'iso','decade', 'data_source',
                   'documentary','drama_fiction_humor', 'propaganda', 'archive', 'video', 'tv', 'news',
                   'animation', 'short', 'audio', 'museum', 'featurefilm', 'play','advertising', 'other','text', 'weights'] # 'summary',
        self.genre_types = ['documentary','drama_fiction_humor','propaganda','archive','video','tv', 'news','animation',
                    'short','audio','museum','featurefilm','play','advertising','other']
        if model_path is None: self.model = AutoModel.from_pretrained('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        else:
            try: self.model = torch.jit.load(model_path, map_location=torch.device('cpu'))
            except: self.model = torch.load(model_path, map_location=torch.device('cpu'))
        self.umap2D = joblib.load(umap2D_path)
        self.umap3D = joblib.load(umap3D_path)
        self.classifier = joblib.load(classifier_path)
        self.model.eval()
        if not tokenizer_path: self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        else: self.tokenizer = torch.load(tokenizer_path) # remove?!
        self.docvecs = self._make_numpy(torch.load(doc_vec_path)) # full document/sentence vectors ~ (num_vectors, embedding_size)
        self.docvecs3D = np.load(doc_vec_3D_path) # UMAP REDUCED document/sentence vectors ~(num_vectors, 3)
        self.docvecs2D = np.load(doc_vec_2D_path) # UMAP REDUCED document/sentence vectors ~(num_vectors, 2)
        self.refdf = pd.read_parquet(reference_df_path, sep='\t', usecols=usecols) # usecols holds human readable columns
        self.refdf.iso = self.refdf.iso.fillna('')
        self.max_length = 20 # user input limited to 20 tokens
        it = simSCESentenceIterator(self.refdf, 8, self.tokenizer,
                                    joblib.load(genre_ids_path),
                                    joblib.load(country_ids_path),
                                    joblib.load(time_ids_path),
                                    joblib.load(genocide_ids_path))
        self.idx_to_genocide = it.idx_to_genocide
        self.flattened_sentences, self.doc_ids, self.title, self.year, self.iso = it.flattened_sentences,it.doc_ids,it.title,it.year,it.iso
        self.genre = (pd.DataFrame(it.genre) * it.genre_cols).apply(lambda x: ', '.join(str(i) for i in x if str(i).strip()), axis=1).values
        self.genocide = pd.Series(it.genocide).map(it.idx_to_genocide)
        self.refdf.drop(columns=['weights'], inplace=True)

    def reduce_user_input(self, user_input_embedding, umap):
        '''uses loaded umap model to reduce user input'''
        user_input_embedding = self._make_numpy(user_input_embedding)
        return umap.transform(user_input_embedding)

    def make_scatter_df(self, user_input, user_input_embedding, dimension='2D'):
        '''
        creates and returns a pandas.DataFrame() with the reduced, pre-computed vectors
        and the user input for downstream purposes of visualization.
        the returned df has these columns:
                x,y (if dimension==2D), or x,y,z (if dimension==3D)
                sentence
                title
                year
                genocide
                iso
                genre
        the last row is the user input
        '''
        if dimension not in ['2D', '3D']:
            raise ValueError('dimension must be 2D or 3D')
        predstr, genocideprobs = self.classify(user_input_embedding)
        red_user_input = self.reduce_user_input(genocideprobs, self.umap2D if dimension == '2D' else self.umap3D)
        if dimension == '2D':
            df = pd.DataFrame(self.docvecs2D, columns=['x', 'y'])
            dfuser = pd.DataFrame(red_user_input, columns=['x', 'y'])
        else:
            df = pd.DataFrame(self.docvecs3D, columns=['x', 'y', 'z'])
            dfuser = pd.DataFrame(red_user_input, columns=['x', 'y', 'z'])
        df = pd.concat([df, dfuser], ignore_index=True)
        df['sentence'] = np.hstack([self.flattened_sentences, [user_input]])
        df['title'] = np.hstack([self.title, ['USER INPUT']])
        df['year'] = np.hstack([self.year, [datetime.now().year]])
        df['genocide'] = np.hstack([self.genocide, ['genocide prediction of user input: ' + predstr]])
        df['iso'] = np.hstack([self.iso, ['no production country']])
        df['genre'] = np.hstack([self.genre, ['no genre']])
        return df

    def create_scatter_fig(self, user_input, user_input_embedding, dimension='2D', **kwargs):
        scatterdf = self.make_scatter_df(user_input, user_input_embedding, dimension)
        if dimension=='2D':
            fig = px.scatter(scatterdf, x='x', y='y', color='genocide',
                            hover_data=['sentence','title', 'year', 'genocide', 'iso','x', 'y'],
                            opacity=0.5, width=1200, height=1200)
        elif dimension=='3D':
            fig = px.scatter_3d(scatterdf, x='x', y='y', z='z',color='genocide',
                                hover_data=['sentence','title', 'year', 'genocide', 'iso','x', 'y', 'z'],
                                opacity=0.5, width=1200, height=1200)

        fig.update_traces(marker=dict(symbol='circle-open'))
        if not kwargs:
            kwargs = {'size': 50, 'symbol': 'cross-thin-open', 'opacity': 1, 'color':'black'}
        fig.data[-1].marker.size = kwargs.get('size')
        fig.data[-1].marker.symbol = kwargs.get('symbol')
        fig.data[-1].marker.opacity = kwargs.get('opacity')
        fig.data[-1].marker.color = kwargs.get('color')
        return fig

    def classify(self, user_input):
        '''
        user_input: torch.FloatTensor | np.ndarray : main model embedding output
        returns: tuple[str, np.ndarray] :
                        predicted label : string with predicted genocide class
                        predicted probabilities of shape(len(user_input), num_genocides)
        '''
        user_input = np.tanh(self._make_numpy(user_input))
        return (self.idx_to_genocide.get(self.classifier.predict(user_input).item()),
                self.classifier.predict_proba(user_input))

    def get_user_input_embedding(self, input, **kwargs):
        '''
        Args:
            input: str : user string input
            kwargs: dict[str, list[int]] : keyword arguments for model
        Returns:
            torch.FloatTensor : mean-pooled model output
        '''
        if len(input) > 20:
            print(f'Warning, search limited to 20 chars, dropping {len(input)-20} chars from search')
        input = input[:20]
        tokenized_input = self.tokenizer(input, return_tensors="pt", truncation=True, padding=True, max_length=self.max_length)
        with torch.no_grad():
            out = self.model(input_ids=tokenized_input['input_ids'], attention_mask=tokenized_input['attention_mask'])
        return self._mean_pooling(out, tokenized_input['attention_mask'])

    def _make_numpy(self, x):
        '''helper to turn torch tensors into numpy'''
        if torch.is_tensor(x):
            x = x.detach()
            if x.is_quantized:
                x = x.dequantize()
            return x.cpu().numpy()
        elif isinstance(x, np.ndarray):
            return x
        else:
            print(f'WARNING: function _make_numpy() was given unknown type {type(x)}')

    def _cos_sim(self, a, b):
        '''
        vector cosine similarity
        Args:
            a : np.ndarray | torch.FloatTensor : shape (num_vectors, embedding_dim)
            b : np.ndarray | torch.FloatTensor : shape (num_vectors, embedding_dim)
        Returns:
            cos sim : np.ndarray : shape (num_vectors, )
        '''
        a = self._make_numpy(a)
        b = self._make_numpy(b)
        # normalize with smoothing in case norm is 0
        a /= (np.linalg.norm(a, axis=1, keepdims=True)  + 1e-5)
        b /= (np.linalg.norm(b, axis=1, keepdims=True)  + 1e-5)
        return (a @ b.T).squeeze()

    def _mean_pooling(self, model_output, attention_mask):
        '''model_output: tuple[torch.FloatTensor] : SentenceTransformer output'''
        token_embeddings = model_output[0] # all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def get_cos_sim(self, embeddings, n_similar, indices):
        embeddings = self._make_numpy(embeddings)

        # avoid indexing errors if user-specified metadata reduces indices to less than n_similar
        if len(indices) < n_similar:
            print(f'warning, n_similar {n_similar} > documents meeting criteria: {len(indices)}, returning only {len(indices)}')
            n_similar = len(indices)

        if len(self.docvecs) > 15000: # if docvecs are array of sentencevecs
            cossims = self._cos_sim(self.docvecs, embeddings)
            len_unique = len(np.unique(self.doc_ids))

            # create array to store the maximums
            max_cos_sim_per_doc = np.full(len_unique, -np.inf)
            np.maximum.at(max_cos_sim_per_doc, self.doc_ids, cossims) # doc_ids look like this [0,0,0,1,1,1,1 .... ] where each value corresponds to a document ix

            # aggregate over documents
            avg_cos_sim_per_doc = np.zeros(len_unique)
            num_sentences_per_doc = avg_cos_sim_per_doc.copy()
            np.add.at(avg_cos_sim_per_doc, self.doc_ids, cossims)
            np.add.at(num_sentences_per_doc, self.doc_ids, 1)
            avg_cos_sim_per_doc /= num_sentences_per_doc

            # take the mean of max and avg?
            mean_avg_max = np.mean(np.vstack([avg_cos_sim_per_doc, max_cos_sim_per_doc]), axis=0).squeeze()
            # mean_avg_max = max_cos_sim_per_doc
            # mean_avg_max = avg_cos_sim_per_doc # just use the summed similarities?

            # sort
            doc_argsorted = np.argsort(mean_avg_max)[::-1] # highest to lowest
            idx_max_sim_sentences = np.argsort(cossims)[::-1]
            most_sim_sentence_per_doc = pd.DataFrame({'doc_id': self.doc_ids[idx_max_sim_sentences[:n_similar]].tolist(),
                                                      'sentences': self.flattened_sentences[idx_max_sim_sentences[:n_similar]].tolist(),
                                                     'similarity': cossims[idx_max_sim_sentences[:n_similar]]})
            # most_sim_sentence_per_doc.drop_duplicates(columns='doc_id', keep='first', inplace=True) # avoid multiple sentences per doc?
            return mean_avg_max, doc_argsorted,\
                   cossims[idx_max_sim_sentences[:n_similar]], idx_max_sim_sentences[:n_similar], most_sim_sentence_per_doc

    def get_similar_vecs(self, user_input, **kwargs):
        '''
        main function to obtain the reduced vectors after similar documents
        to user input have been found.
        Args:
            user_input: str : user string input
            kwargs: dict[str, list[str]] : keyword arguments for model
        Returns:
            tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]
                 - first pd.DataFrame: df similar docs by aggregation criterion (mean, sum, max)
                 - first np.ndarray: n_similar docs argsorted by aggregation criterion (mean, sum, max)
                 - second np.ndarray: n_similar-many sentence cosine similarities
                 - third np.ndarray: n_similar-many argsorted indices of max-cosine similar sentences
                 - second pd.DataFrame: df with n_similar-many

        '''
        user_embedding = self.get_user_input_embedding(user_input, **kwargs)
        sliced_df = self._kwargs_parser(**kwargs)
        if kwargs.get('n_similar'):
            n_similar = kwargs.get('n_similar')
        else:
            n_similar = 10
        if n_similar > len(sliced_df):
            n_similar = len(sliced_df)

        doc_cossim, doc_argsorted, sentence_cossim, idx_max_sim_sentences, most_sim_sentence_per_doc = self.get_cos_sim(user_embedding, n_similar=n_similar, indices=sliced_df.index.values)

        most_sim_sentence_per_doc = pd.merge(most_sim_sentence_per_doc, self.refdf, how='left', left_on=['doc_id'], right_index=True)
        return sliced_df.loc[doc_argsorted[np.isin(doc_argsorted, sliced_df.index.values)]].iloc[np.arange(n_similar)], \
               doc_cossim[doc_argsorted[:n_similar]], doc_argsorted[:n_similar],\
               self.flattened_sentences[idx_max_sim_sentences], sentence_cossim, idx_max_sim_sentences, most_sim_sentence_per_doc

    def _kwargs_parser(self, **kwargs):
        '''
        kwargs : dict[str, list[str]] : with keys for genre, country, decade, genocide and corresponding values

        returns: pd.DataFrame
        '''
        outputdict = {'genre':[], 'country':[], 'decade':[], 'genocide':[]}
        for k, v in kwargs.items():
            if k not in outputdict.keys() and k != 'n_similar':
                print(f'warning kwargs input with key {k} not known, ignoring')
            else: assert type(v) == list, 'values of kwargs dict should be a list of at least one string'

        sliced_df = self.refdf.copy()
        if kwargs.get('genre'):
            sliced_df = sliced_df[np.all((sliced_df.loc[:, [v for v in kwargs.get('genre') if v in self.genre_types]].values>0), axis=1)]
        if kwargs.get('country'):
            sliced_df = sliced_df[sliced_df.iso.str.contains('|'.join([v for v in kwargs.get('country') if v in self.country_ids.keys()]), regex=True, case=False, na=False)]
        if kwargs.get('decade'):
            sliced_df = sliced_df[sliced_df.decade.isin(kwargs.get('decade'))]
        if kwargs.get('genocide'):
            sliced_df = sliced_df[sliced_df.genocide.isin(kwargs.get('genocide'))]
        return sliced_df