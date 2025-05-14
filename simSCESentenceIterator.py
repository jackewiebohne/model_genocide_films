import numpy as np

class simSCESentenceIterator():
    def __init__(self, df, batch_size, tokenizer,
                 genre_cols,
                 country_to_idx,
                 decade_to_idx,
                 genocide_to_idx
                 ):

        self.country_ids = np.zeros([len(df),len(country_to_idx)])
        self.country_to_idx = country_to_idx
        dfcountryix = df.iso.apply(lambda x: [country_to_idx.get(ele) for ele in x.split(',')]).tolist()
        for i, indices in enumerate(dfcountryix):
            self.country_ids[i, indices] = 1 # multi-hot encoding for country
        self.country_ids = self.country_ids.tolist()
        self.flattened_sentences = []
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.genocide_to_idx = genocide_to_idx
        self.decade_to_idx = decade_to_idx
        self.genre_cols = genre_cols
        self.idx_to_decade = {i: d for d,i in decade_to_idx.items()}
        self.idx_to_genocide = {i: ele for ele,i in genocide_to_idx.items()}
        self.idx_to_country = {i: iso for iso,i in country_to_idx.items()}
        self.regex_pattern = r'(?<=[a-zA-Z]{2})\.\s*|\?\s*|\!\s*|\;\s*' # split at period IFF it's not an abbreviation (more than one letter precedes it)
        to_keep = r'(Holodomor|Title|Holocaust)'
        self.flattened_sentences, self.doc_ids, self.title, self.year, self.iso, self.weights, self.genre, self.decade, self.genocide, self.country = zip(*[
                                                                                                (sentence.strip() + '.', i, title, year, iso, weight, genre, decade, genocide, country)
                                                                                                for i, (doctext, title, year, iso, weight, genre, decade, genocide, country) in
                                                                                                    enumerate(zip(
                                                                                                                df.text.tolist(),
                                                                                                                df.title.tolist(),
                                                                                                                df.year.tolist(),
                                                                                                                df.iso.tolist(),
                                                                                                                df.weights.tolist(),
                                                                                                                df[genre_cols].values.tolist(),
                                                                                                                df.decade.apply(lambda x: decade_to_idx.get(x, 12)).tolist(), # 12 == 'nan'
                                                                                                                df.genocide.apply(lambda x: genocide_to_idx.get(x)).tolist(),
                                                                                                                self.country_ids
                                                                                                            ))
                                                                                if isinstance(doctext, str)
                                                                                for sentence in re.split(self.regex_pattern, doctext)
                                                                                if sentence and (len(re.sub(to_keep,
                                                                                    'long placeholder string to keep short sentences with these words from being dropped',
                                                                                    sentence).split()) > 2)
                                                                ])
        self.flattened_sentences = np.array(self.flattened_sentences).squeeze()
        self.doc_ids = np.array(self.doc_ids, dtype=np.uint16).squeeze()
        self.iso = np.array(self.iso).squeeze()
        self.title = np.array(self.title).squeeze()
        self.year = np.array(self.year, dtype=np.float16).squeeze()
        self.weights = np.array(self.weights, dtype=np.float32).squeeze()
        self.genre = np.array(self.genre, dtype=np.uint8).squeeze()
        self.decade = pd.Series(self.decade).fillna(12).values # 12 == 'nan'
        self.decade = np.array(self.decade, dtype=np.uint8).squeeze()
        self.genocide = np.array(self.genocide, dtype=np.uint8).squeeze()
        self.country = np.array(self.country, dtype=np.uint8).squeeze()
        # count occurrences of doc_ids, maps counts onto doc_ids, returns array of counts of len(doc_ids)
        # self.lens thus is an indicator of the number of sentences per doc
        self.lens = pd.Series(self.doc_ids).map(Counter(self.doc_ids)).values.astype(np.uint16)
        assert len(self.flattened_sentences) == len(self.doc_ids) == len(self.lens) == len(self.weights)
        assert len(df) == len(np.unique(self.doc_ids)), 'df and doc_ids do not match. possibly documents containing only short sentence were tacitly dropped?'

    def __len__(self):
        return len(self.flattened_sentences)

    def get_class_distributions(self):
        self.genocide_counts = pd.DataFrame({'genocide':self.genocide})['genocide'].map(self.idx_to_genocide).value_counts().reset_index()
        self.genocide_counts['percentage'] = self.genocide_counts['count']/self.genocide_counts['count'].sum()

        self.decade_counts = pd.DataFrame({'decade':self.decade})['decade'].map(self.idx_to_decade).value_counts().reset_index()
        self.decade_counts['percentage'] = self.decade_counts['count']/self.decade_counts['count'].sum()

        self.genre_counts = pd.DataFrame({'genre': self.genre_cols, 'count':self.genre.sum(axis=0).tolist()})
        self.genre_counts['percentage'] = self.genre_counts['count']/self.genre_counts['count'].sum()

        self.country_counts = pd.DataFrame({'country': self.country_to_idx.keys(), 'count': self.country.sum(axis=0).tolist()})
        self.country_counts['percentage'] = self.country_counts['count']/self.country_counts['count'].sum()
        return self.genocide_counts, self.decade_counts, self.genre_counts, self.country_counts

    def balanced_train_test_split(self, split_on='genocide', test_size=0.2, holdout=None, percent_majority=None):
        '''
        Samples so that the majority class (in split_on feature) is either the size of all minority
        classes combined or a custom-set proportion

        Args:
            split_on : str : which of ['genocide', 'decade', 'genre', 'country'] to split on
            test_size : float : percent for test set
            holdout : None or np.ndarray of indices : these indices are ignored. Holdout can
                        be deliberately selected indices that aren't to be included in train or test,
                        or, more commonly, they can be test indices returned from a previous run of balanced_train_test_split.
            percent_majority : None or float : set the percent-proportion of majority class, if None the majority class
                        will approximate the size of all minority classes combined
        returns:
            boolean np.ndarray of len(flattened_sentences) to be used to slice data (flattened_sentences, doc_ids etc.)
        '''
        # holdout = if not None expects boolean array of len(sentences) which will not be considered for train test split
        # returns np.ndarray index of: train, test
        assert split_on in ['genocide', 'decade', 'genre', 'country']
        self.get_class_distributions()
        get_count_map = {'genocide':self.genocide_counts, 'decade':self.decade_counts, 'genre':self.genre_counts, 'country':self.country_counts}
        get_var_map = {'genocide': self.genocide, 'decade': self.decade, 'genre': self.genre, 'country': self.country}
        dist = get_count_map.get(split_on).sort_values(by='count')
        if percent_majority is None:
            ## add all count values of classes smaller than majority
            percent_minority = dist['count'].values[:-1].sum()/dist['count'].values.sum()
        else:
            percent_minority = percent_majority
        maj_index = dist.index[-1]
        var = get_var_map.get(split_on)
        maj_rand = np.random.rand(var.shape[0])
        if split_on in ('genocide', 'decade'):
            if holdout is not None:
                minority_slice = (var!=maj_index) & (holdout==False)
                majority_slice = (var==maj_index) & (maj_rand < percent_minority) & (holdout==False)
            else:
                minority_slice = var!=maj_index
                majority_slice = (var==maj_index) & (maj_rand < percent_minority)
            selected_indices = minority_slice | majority_slice
            train_split = np.random.rand(len(selected_indices)) > test_size
            return (selected_indices==True) & train_split, (selected_indices==True) & (train_split==False)
        if split_on in ('genre', 'country'): # multi-hot
            # shape (len(sentences), num_split_on_cols)
            if holdout is not None:
                minority_slice = (var[:, maj_index] == 0) & (holdout==False)
                majority_slice = (var[:, maj_index] == 1) & (maj_rand < percent_minority) & (holdout==False)
            else:
                minority_slice = var[:, maj_index] == 0
                majority_slice = (var[:, maj_index] == 1) & (maj_rand < percent_minority)
            selected_indices = minority_slice | majority_slice
            train_split = np.random.rand(len(selected_indices)) < test_size
            return (selected_indices==True) & train_split, (selected_indices==True) & (train_split==False)

    def sort_by_length(self, reverse=True):
        argsorted = np.argsort(self.lens)[::-1] # sort by descending lengths; if issues with token len & memory we want to find out at start, not end, of training
        self.flattened_sentences = self.flattened_sentences[argsorted]
        self.doc_ids = self.doc_ids[argsorted]
        self.lens = self.lens[argsorted]
        self.weights = self.weights[argsorted]
        self.genre = self.genre[argsorted]
        self.decade = self.decade[argsorted]
        self.genocide = self.genocide[argsorted]
        self.country = self.country[argsorted]

    def save(self, save_path):
        pd.Series(self.flattened_sentences).to_csv(save_path + 'flattened_sentences.tsv', sep='\t', index=False)
        np.save(save_path + 'doc_ids.npy', self.doc_ids)
        np.save(save_path + 'lens.npy', self.lens)

    def iterate_once(self, shuffle=True):
        if shuffle:
            shuffle_idx = np.random.permutation(np.arange(len(self.flattened_sentences)))
            self.flattened_sentences = self.flattened_sentences[shuffle_idx]
            self.doc_ids = self.doc_ids[shuffle_idx]
            self.lens = self.lens[shuffle_idx]
            self.weights = self.weights[shuffle_idx]
            self.genre = self.genre[shuffle_idx]
            self.decade = self.decade[shuffle_idx]
            self.genocide = self.genocide[shuffle_idx]
            self.country = self.country[shuffle_idx]
        num_batches = int(np.ceil(len(self.flattened_sentences) / self.batch_size))
        for i in range(num_batches):
            minimum = i*self.batch_size
            maximum = min((i+1)*self.batch_size, len(self.flattened_sentences))
            batch_sentences = self.flattened_sentences[minimum:maximum]
            sentence_lengths = [len(self.tokenizer.encode(sentence)) for sentence in batch_sentences]
            if sentence_lengths:
                max_length = min(max(sentence_lengths), 512)
                encoded_text = self.tokenizer(
                                                [s for s in batch_sentences],
                                                truncation=True,
                                                padding='max_length',
                                                max_length=max_length,
                                                return_tensors='pt'
                                            )
                yield {
                    'input_ids': encoded_text['input_ids'],
                    'attention_mask': encoded_text['attention_mask'],
                    'weights': torch.tensor(self.weights[minimum:maximum], dtype=torch.float32),
                    'genre': torch.tensor(self.genre[minimum:maximum], dtype=torch.uint8),
                    'decade': torch.tensor(self.decade[minimum:maximum], dtype=torch.uint8),
                    'genocide': torch.tensor(self.genocide[minimum:maximum], dtype=torch.uint8),
                    'country': torch.tensor(self.country[minimum:maximum], dtype=torch.uint8)
                }
            else: print('WARNING: sentences are empty')


    def withhold_data(self, holdouts):
        '''
        function that permanently resets the class attributes/vars so that during any calls of self.iterate_once()
        sentences, genre, etc with index in the provided holdout array are not iterated over and yielded.
        where the class attributes/vars match with index in holdout array they are stored under new variables
        that can be accessed by holdout_ + variable_name which each stores a list of len(holdouts) of np.ndarray[str or float or int]
        that was sliced by the indices in holdouts.

        for example:
            holdouts holds two lists of validation and test indices, where each is a boolean array of len(flattened_sentences)
            withhold_data will iterate over the two lists and store under `class_instance.holdout_flattened_sentences` two string arrays (of possibly
            different lengths) where the first array will be validation and the second test sentences

        Args:
            holdouts : list[np.ndarray[bool]] : list of one or more boolean arrays that are to be excluded
        '''
        assert isinstance(holdouts, list), 'holdouts must be a list of np.ndarray with bool'
        assert len(holdouts[0]) == len(self.flattened_sentences), 'arrays in holdouts list must be of same length as data stored in iterator (e.g flattened_sentences)'
        include_slice = np.ones(len(holdouts[0])).astype(bool)
        self.holdout_flattened_sentences = []
        self.holdout_doc_ids = []
        self.holdout_lens = []
        self.holdout_weights = []
        self.holdout_genre = []
        self.holdout_decade = []
        self.holdout_genocide = []
        self.holdout_country = []
        # withheld data
        for holdout in holdouts:
            include_slice = (include_slice & (holdout==False))
            self.holdout_flattened_sentences += [self.flattened_sentences[holdout==True]]
            self.holdout_doc_ids += [self.doc_ids[holdout==True]]
            self.holdout_lens += [self.lens[holdout==True]]
            self.holdout_weights += [self.weights[holdout==True]]
            self.holdout_genre += [self.genre[holdout==True]]
            self.holdout_decade += [self.decade[holdout==True]]
            self.holdout_genocide += [self.genocide[holdout==True]]
            self.holdout_country += [self.country[holdout==True]]
        # keep data for iteration only iff not in holdouts
        self.flattened_sentences = self.flattened_sentences[include_slice]
        self.doc_ids = self.doc_ids[include_slice]
        self.lens = self.lens[include_slice]
        self.weights = self.weights[include_slice]
        self.genre = self.genre[include_slice]
        self.decade = self.decade[include_slice]
        self.genocide = self.genocide[include_slice]
        self.country = self.country[include_slice]

    def jointly_iterate_once_with_new_data(self, more_sentences, more_genocide_labels, shuffle=True, holdouts=None):
        '''
        helper function that allows for joint iteration over data stored natively in the iterator
        and new data that is not stored in the iterator.
        note that when holdout is provided class attributes/vars are changed
        in accordance to withhold_data function (more_sentences and more_genocide_labels won't be affected
        by this and will always be iterated over completely)

        Args:
            more_sentences : list[str] or np.ndarray[str] : list of sentences
            more_genocide_labels : list[int] or np.ndarray[str]: list of labels (indices in genocide_to_idx)
            holdout : list[np.ndarray[bool]] : list of one or more boolean np.ndarrays (e.g. a validation and test slice index)
                            that are to be excluded from iteration

        yields:
            dict[str, torch.FloatTensor] :
                batch of data with keys 'input_ids', 'attention_mask', 'weights', 'genre','decade', 'genocide', 'country', 'data_source'.
                'data_source' matches the first dimension of input_ids, attention_mask, and genocide and thus enables tracking the positions of the
                original batches that were concatenated
                Note that the lenghts of the values will differ and only the values of keys 'input_ids', 'attention_mask', 'genocide'
                will match and that in case len(more_sentences) > len(self.flattened_sentences) the values for keys 'weights', 'genre','decade', 'country'
                will be None

        '''
        print('NOTE: two batches of data will be concatenated, effectively doubling batch_size. you may have to manually adjust batch_size to avoid memory issues')
        print('WARNING: values for \'weights\', \'genre\',\'decade\', \'country\' CAN be None')
        if holdouts is not None:
            try:
                self.holdout_flattened_sentences
                print('data are already withheld. ignoring holdouts')
            except:
                self.withhold_data(holdouts)
                print('NOTE: data is withheld and can be accessed with class_instance.holdout_ + some_variable_name')
        once1 = self.iterate_once(shuffle=shuffle)
        once2 = self.ingest_n_batch_once(more_sentences, more_genocide_labels, shuffle=shuffle)
        for (batch1, batch2) in itertools.zip_longest(once1, once2, fillvalue={}):
            if batch2 and batch1:
                joint_batch = batch1.copy()
                # each batch was generated with flexible max_length padded only to the max length of sentences within each batch
                # to join the batches, we must pad again to the max_length across BOTH batches
                if batch1['input_ids'].size(1) < batch2['input_ids'].size(1): # if sentences in batch2 are longer
                    add_zero_pad = torch.zeros(batch2['input_ids'].size(1) - batch1['input_ids'].size(1), dtype=torch.int64).repeat(batch1['input_ids'].size(0),1)
                    batch1['input_ids'] = torch.cat([batch1['input_ids'], add_zero_pad], dim=-1)
                    batch1['attention_mask'] = torch.cat([batch1['attention_mask'], add_zero_pad], dim=-1)
                elif batch1['input_ids'].size(1) > batch2['input_ids'].size(1): # if sentences in batch1 are longer
                    add_zero_pad = torch.zeros(batch1['input_ids'].size(1) - batch2['input_ids'].size(1), dtype=torch.int64).repeat(batch2['input_ids'].size(0),1)
                    batch2['input_ids'] = torch.cat([batch2['input_ids'], add_zero_pad], dim=-1)
                    batch2['attention_mask'] = torch.cat([batch2['attention_mask'], add_zero_pad], dim=-1)
                joint_batch['input_ids'] = torch.cat([batch1['input_ids'], batch2['input_ids']], dim=0)
                joint_batch['attention_mask'] = torch.cat([batch1['attention_mask'], batch2['attention_mask']], dim=0)
                joint_batch['genocide'] = torch.cat([batch1['genocide'], batch2['genocide']], dim=0)
                joint_batch['data_source'] = np.array(['film'] * len(batch1['genocide']) + ['nonfilm'] * len(batch2['genocide']))
                yield joint_batch
            elif batch1 and not batch2:
                batch1['data_source'] = np.array(['film'] * len(batch1['genocide']))
                yield batch1
            elif batch2 and not batch1:
                batch2['country'] = None
                batch2['genre'] = None
                batch2['decade'] = None
                batch2['data_source'] = np.array(['nonfilm'] * len(batch2['genocide']))
                yield batch2

    def ingest_n_batch_once(self, more_sentences, more_genocide_labels, shuffle=True, **kwargs):
        ''''
        helper generator function that takes additional sentences and additional genocide labels (indices in
        genocide_to_idx) for further training. function is agnostic about the other labels (decade, genre, ...).
        the new data is not incorporated into the class attributes/vars.

        Args:
            more_sentences : list[str] or np.ndarray[str] : list of sentences
            more_genocide_labels : list[int] or np.ndarray[str] : list of labels (indices in genocide_to_idx)
            shuffle : bool
            kwargs : dict[str, np.ndarray] or dict[str, torch.FloatTensor] :
        yields:
            dict[str, torch.FloatTensor] : batch of data with keys 'input_ids', 'attention_mask', 'genocide'
        '''
        assert len(more_sentences) == len(more_genocide_labels)
        if isinstance(more_sentences, list): more_sentences = np.array(more_sentences)
        if isinstance(more_genocide_labels, list): more_genocide_labels = np.array(more_genocide_labels)
        if shuffle:
            shuffle_idx = np.random.permutation(np.arange(len(more_sentences)))
            more_sentences = more_sentences[shuffle_idx]
            more_genocide_labels = more_genocide_labels[shuffle_idx]
        num_batches = int(np.ceil(len(more_sentences) / self.batch_size))
        for i in range(num_batches):
            minimum = i*self.batch_size
            maximum = min((i+1) * self.batch_size, len(more_sentences))
            batch_sentences = more_sentences[minimum:maximum]
            sentence_lengths = [len(self.tokenizer.encode(sentence)) for sentence in batch_sentences]
            if sentence_lengths:
                max_length = min(max(sentence_lengths), 512)
                encoded_text = self.tokenizer(
                                                [s for s in batch_sentences],
                                                truncation=True,
                                                padding='max_length',
                                                max_length=max_length,
                                                return_tensors='pt'
                                            )
                if kwargs:
                    yield {'input_ids': encoded_text['input_ids'],
                        'attention_mask': encoded_text['attention_mask'],
                        'genocide': torch.tensor(more_genocide_labels[minimum:maximum], dtype=torch.uint8),
                        **{k: torch.tensor(v[minimum:maximum], dtype=torch.uint8) for k,v in kwargs.items()}}

                else:
                    yield {'input_ids': encoded_text['input_ids'],
                        'attention_mask': encoded_text['attention_mask'],
                        'genocide': torch.tensor(more_genocide_labels[minimum:maximum], dtype=torch.uint8)}
            else: print("WARNING: sentences are empty")


    def adjust_genocide_weights(self, train_indices, more_labels=None, clip_min=1/3, clip_max=16, verbose=False):
            '''
            function to get weights per genocide based on distribution of genocides in training set and any additional labels provided.
            NOTE that if not all genocides are in train_indices and/or more_labels weights for these will default to clip_max

            Args:
                train_indices : np.ndarray : train indices from a run of balanced_train_test_split
                                with split_on='genocide'.
                clip_min : float : minimum to clip weights
                clip_max : float : max to clip weights
                verbose : bool : if there should be a printout matching genocide to weights or not
                more_labels : np.ndarray[int] : additional labels with integers in range of genocide_to_idx values

            returns:
                torch.FloatTensor : of len(genocide_to_idx) where keys are index representation of genocides (that
                                can be looked up in idx_to_genocide) and values are the new weights

            '''
            try:
                genocide_weights = np.zeros(len(self.idx_to_genocide.keys()))
                unique, counts = np.unique(self.genocide[train_indices], return_counts=True)
                genocide_weights[unique] = counts
                if more_labels is not None:
                    unique2, counts2 = np.unique(more_labels, return_counts=True)
                    genocide_weights[unique2] += counts2
                    unique = np.unique(np.hstack([unique, unique2]))
                total = np.sum(genocide_weights)
                new_weights = np.clip(np.sqrt(total/(genocide_weights * len(unique))), clip_min, clip_max)
                if verbose:
                    print({self.idx_to_genocide.get(i):v for i,v in enumerate(new_weights)})
                return torch.tensor(new_weights, dtype=torch.float32)
            except:
                raise IndexError('index out of bounds, adjust_genocide_weights needs to be called BEFORE withhold_data method')


def add_tokens_and_initialize_embeddings(tokenizer, model, new_tokens, words_to_average):
    """
    Adds new tokens to the tokenizer and initializes their embeddings by averaging embeddings of provided words.

    Args:
        tokenizer (BertTokenizer): Pretrained BERT tokenizer.
        model (nn.Module): BERT-based model, with `model.bert` attribute to access the BERT model.
        new_tokens (list[str]): List of new tokens to be added to the tokenizer.
        words_to_average (list[list[str]]): List of lists containing words whose embeddings will be averaged
                                            for each corresponding new token.
    """
    assert len(new_tokens) == len(words_to_average)
    tokenizer.add_tokens(new_tokens)
    try:
        model.bert.resize_token_embeddings(len(tokenizer))
    except:
        model.resize_token_embeddings(len(tokenizer))
    bert_embeddings = model.bert.embeddings.word_embeddings
    for i, token in enumerate(new_tokens):
        words_for_average = words_to_average[i]
        token_ids = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word)) for word in words_for_average]
        token_ids = [item for sublist in token_ids for item in sublist]
        token_embeddings = bert_embeddings(torch.tensor(token_ids))
        avg_embedding = torch.mean(token_embeddings, dim=0)
        new_token_id = tokenizer.convert_tokens_to_ids(token)
        with torch.no_grad():
            bert_embeddings.weight[new_token_id] = avg_embedding
    print(f"Added {len(new_tokens)} tokens and initialized their embeddings.")