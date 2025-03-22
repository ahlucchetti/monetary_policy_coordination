import os

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

import nltk
nltk.download('stopwords')

from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

import networkx as nx


class speech_similarity(object):
  def __init__(self,
               speech_threshold = 50,
               first_year = 2010,
               word_speech_min = 5,
               word_corpus_min = 10,
               alpha = 0.1, # (0,1]
               beta = 3,    # [1,inf)
               project_path = '/content/drive/MyDrive/' \
                            + 'Monetary Policy Coordination',
               data_dir = '1_raw_txt',
               ):
    """
    speech_threshold: Minimum number of speeches for a central bank to be
    included in the analysis.

    first_year: First year to include in the study.

    word_speech_min: Rare word removel parameter. Removes all words that arise
    less than this much in a speech.

    word_corpus_min: Rare word removel parameter. Removes all words that arise
    in less than this much speeches.
    """
    self.project_path = project_path
    self.data_dir = data_dir
    self.data_path = os.path.join(project_path, data_dir)

    self.speech_threshold = speech_threshold
    self.first_year = first_year
    self.word_speech_min = word_speech_min
    self.word_corpus_min = word_corpus_min
    self.alpha = alpha
    self.beta = beta

    self.speech_count_year = None
    self.speech_count = speech_count = None
    self.selected_central_banks = None
    self.word_index = None
    self.central_bank_index = None
    self.term_document = None
    self.term_central_bank_year = None
    self.period_results = {}

  # Aux functions to data preprocessing
  def tokenize(self, text, *args):
    """
    Tokenization

    *args are used to specify regex paterns. The default is '\w+'
    """
    if not args:
      pattern = r'\w+'
    else:
      pattern = r'|'.join(args)
    regex_tokenizer = RegexpTokenizer(pattern)
    return regex_tokenizer.tokenize(text)

  def remove_stop_words(self, tokens):
    """
    Stop word removal
    """
    stop_words = stopwords.words('english')
    return [w for w in tokens if not w.lower() in stop_words]

  def stem_tokens(self, tokens):
    """
    Stemming
    """
    stemmer = PorterStemmer()
    return [stemmer.stem(w) for w in tokens]

  # Rare word removal functions
  def rare_word_rem_speech(self, x):
    return x >= self.word_speech_min

  def rare_word_rem_corpus(self, x):
    return x.count(axis=1) >= self.word_corpus_min

  # Data functions
  def explore_speech_data(self):
    """
    Explore speeches made available by Hansson (2021).
    This step calculates the total number of speeches per central bank per
    year.

    https://github.com/HanssonMagnus/scrape_bis

    Hansson, M. (2021). Evolution of topics in central bank speech
    communication. arXiv preprint arXiv:2109.10058.
    """
    speech_count_year = {}

    for central_bank in os.listdir(self.data_path):
      central_bank_dict = {}
      for file_name in os.listdir(os.path.join(self.data_path, central_bank)):
        year = int(file_name[:4])
        if year not in central_bank_dict:
          central_bank_dict[year] = 1
        else:
          central_bank_dict[year] += 1
      speech_count_year[central_bank] = central_bank_dict

    speech_count_year = pd.DataFrame(speech_count_year).fillna(0).sort_index()
    speech_count = speech_count_year.sum(axis=0).sort_values(ascending=False)
    self.speech_count_year = speech_count_year
    self.speech_count = speech_count

  def calc_term_central_bank_year(self):
    """
    Calculates the term-central-bank-year matrix. This is a version of a
    term-document matrix, with documents (speeches) aggregated by central bank
    and year.
    """
    term_central_bank_year = self.term_document.T.groupby(level=(0,1)).sum().T
    self.term_central_bank_year = term_central_bank_year

  def load_speech_data_to_tdm(self):
    """
    Loads speeches made available by Hansson (2021) and prepares the initial
    version of the term-document matrix.

    https://github.com/HanssonMagnus/scrape_bis

    Hansson, M. (2021). Evolution of topics in central bank speech
    communication. arXiv preprint arXiv:2109.10058.
    """
    selected_central_banks = self.speech_count.loc[lambda x:
                                                   x >= self.speech_threshold
                                                   ].index.to_list()
    self.selected_central_banks = selected_central_banks

    speech_list = []
    central_bank_index = {}
    i = 0

    for central_bank in self.selected_central_banks:
      central_bank_index[i] = central_bank
      for file_name in os.listdir(os.path.join(self.data_path, central_bank)):
        year = int(file_name[:4])
        if year >= self.first_year:
          with open(os.path.join(self.data_path,
                                 central_bank,
                                 file_name),
                                 'r',
                                 errors='ignore') as file:
            speech = self.stem_tokens(
                self.remove_stop_words(
                    self.tokenize(file.read())
                    )
                )
          speech_list.append(pd.Series(
              speech,
              dtype=str,
              ).value_counts().rename((i, year)).loc[self.rare_word_rem_speech])
      i += 1

    term_document = pd.concat(speech_list,
                              axis=1).loc[self.rare_word_rem_corpus].fillna(0)
    word_index = term_document.index.to_series().reset_index(drop=True)
    central_bank_index = pd.Series(central_bank_index)
    term_document = term_document.reset_index(drop=True)
    self.word_index = word_index
    self.central_bank_index = central_bank_index
    self.term_document = term_document
    self.calc_term_central_bank_year()

  def save_tdm(self,
               directory='Processed_Data',
               term_document_file = 'Term_Document.csv',
               word_index_file = 'Word_Index.csv',
               central_bank_index_file = 'Central_Bank_Index.csv',
               ):
    """
    Saves the term-document matrix in the provided directory
    """
    processed_data_path = os.path.join(self.project_path,
                                       directory)
    term_document_path = os.path.join(processed_data_path,
                                      term_document_file)
    word_index_path = os.path.join(processed_data_path,
                                   word_index_file)
    central_bank_index_path = os.path.join(processed_data_path,
                                           central_bank_index_file)

    self.term_document.to_csv(term_document_path)
    self.word_index.to_csv(word_index_path)
    self.central_bank_index.to_csv(central_bank_index_path)

  def load_tdm(self, load_file = 'Term_Document.csv',
               word_file='Word_Index.csv',
               central_bank_file='Central_Bank_Index.csv',
               ):
    """
    Loads a previously saved term-document matrix from self.data_dir.
    Also looks for word_index and central_bank_index in self.data_dir.
    """
    term_document = pd.read_csv(os.path.join(self.project_path,
                                             self.data_dir,
                                             load_file),
                                header=[0,1],
                                index_col=0)
    term_document = term_document.rename(columns=lambda x: int(x[:4]))
    word_index = pd.read_csv(os.path.join(self.project_path,
                                          self.data_dir,
                                          word_file),
                             header=0,
                             index_col=0)
    central_bank_index = pd.read_csv(os.path.join(self.project_path,
                                                  self.data_dir,
                                                  central_bank_file),
                                     header=0,
                                     index_col=0)
    # if eventually needed to drop some central bank
    # hard coded in this version
    cb_drop = [#48, # bank indonesia
               #53, # bank of ghana
               # G10
              #  5, # bank of japan
              #  0, # european central bank
              #  7, # reserve bank of australia
              #  24, # reserve bank of new zealand
              #  6, # bank of canada
              #  10, # sveriges riksbank
              #  15, # swiss national bank
              #  20, # central bank of norway
              #  4, # bank of england
              #  1, # board of governors of the federal reserve system
               # bancos centrais contidos em outros
              #  3, # 100 -> deutsche bundesbank
              #  11, # federal reserve bank of new york
              #  12, # 100 -> bank of france
              #  14, # 100 -> bank of italy
              #  16, # 100 -> bank of spain
              #  17, # 100 -> central bank of ireland
              #  22, # 100 -> netherlands bank
              #  28, # 100 -> bank of greece
              #  45, # central bank of the republic of austria
              #  46, # national bank of romania
              #  49, # czech national bank
              #  52, # bank of portugal
               # demais para restar G10
              #  2, # reserve bank of india
              #  8, # central bank of the philippines
              #  9, # central bank of malaysia
              #  13, # south african reserve bank
              #  18, # monetary authority of singapore
              #  19, # bank of albania
              #  21, # hong kong monetary authority
              #  23, # bank of thailand
              #  25, # central bank of kenya
              #  26, # bank of finland
              #  27, # bank of mauritius
              #  29, # bank of zambia
              #  30, # bank of uganda
              #  31, # people's bank of china
              #  32, # state bank of pakistan
              #  33, # reserve bank of fiji
              #  34, # national bank of serbia
              #  35, # central bank of chile
              #  36, # bank of israel
              #  37, # central bank of the republic of turkey
              #  38, # national bank of denmark
              #  39, # central bank of barbados
              #  40, # central bank of trinidad and tobago
              #  41, # bank of mexico
              #  42, # national bank of north macedonia
              #  43, # bank of korea
              #  44, # central bank of iceland
              #  47, # central bank of sri lanka
              #  50, # bank of papua new guinea
              #  51, # central bank of malta
              #  54, # bulgarian national bank
               ]
    rename_cb = {v: k for k, v in (central_bank_index
                                   .drop(cb_drop)
                                   .reset_index()['index']
                                   .to_dict()
                                   ).items()}
    central_bank_index = (central_bank_index
                          .drop(cb_drop)
                          .rename(index=rename_cb)
                          )
    # again a hard coded list of terms to drop
    w_drop = [3027, # etc
              322, # u
              ]
    rename_w = {v: k for k, v in (word_index
                                  .drop(w_drop)
                                  .reset_index()['index']
                                  .to_dict()
                                  ).items()}
    word_index = (word_index
                  .drop(w_drop)
                  .rename(index=rename_w)
                  )
    term_document = (term_document
                     .drop(cb_drop, axis=1)
                     .drop(w_drop)
                     .rename(index=rename_w,
                             columns=rename_cb,
                             )
                     )
    self.term_document = term_document
    self.word_index = word_index.iloc[:,0]
    self.central_bank_index = central_bank_index.iloc[:,0]
    self.calc_term_central_bank_year()

  # Model aux functions
  def calc_N(self, term_central_bank_year=None):
    """
    Calculates the matrix N with the term count for each central bank in the
    analysis period
    """
    if term_central_bank_year is None:
      term_central_bank_year = self.term_central_bank_year
    N = term_central_bank_year.T.groupby(level=0).sum().T
    # this reinforces rare word exclusion, now for the filtered period:
    N = N.replace(0, np.nan).loc[self.rare_word_rem_corpus].fillna(0)
    return N

  def calc_F(self, N):
    """
    Calculates the matrix F, representing the importance of each term in each
    document for a given matrix N.

    F is calculated according to the Entropy Model, a model that provides a
    similarity evaluation closer to the one provided by actual people in an
    experiment that compares the similarities generated by an algorithm to the
    ones evaluated by people (Pincombe, 2004).

    Pincombe, B. (2004) Comparison of Human and Latent Semantic Analysis (LSA)
    Judgements of pairwise document similarities for a news corpus.
    Technical Report. Defence Science and Technology Organisation Department of
    Defense – Australian Government.
    """
    n = N.div(N.sum(axis=0), axis=1)
    omega_local = np.log2(n + 1)
    g = N.sum(axis=1).div(N.sum().sum())
    P = N.div(N.sum(axis=1), axis=0)
    with np.errstate(divide='ignore'):
      omega_global = 1 + ( (P * np.log2(P)).fillna(0).sum(axis=1)
                          /(1 + np.log2(N.shape[1]))
                         )
    F = omega_local.multiply(omega_global, axis=0)
    return F, omega_local, omega_global, P

  def calc_Q(self, F):
    """
    Calculates the similarity matrix Q.
    """
    Q_numerator = F.T @ F
    Q_norms = np.sqrt(Q_numerator.to_numpy().diagonal()[:,np.newaxis])
    Q_denominator = Q_norms @ Q_norms.T
    Q = Q_numerator / Q_denominator
    return Q, Q_numerator, Q_denominator, Q_norms

  def calc_M(self, Q):
    """
    Calculates de similarity perception matrix M.
    """
    M = self.alpha * Q ** self.beta
    # zeroing the diagonal will make our lives easier ahead
    M = M - np.diag(M.to_numpy().diagonal())
    return M

  def calc_G(self, M, draw=False):
    """
    Calculates the graph representation for the matrix M
    """
    G = nx.from_numpy_array(M.to_numpy())
    if draw:
      nx.draw(G)
    return G

  def Pi(self, Pi_, M):
    """
    Matrix Pi as a function of itself for fixed point iteration calculation
    """
    return (1 - (1 - M.multiply(Pi_, axis=1)).prod()).to_numpy()

  def fixed_point_iter(self, fun, x0, args=[],
                       tol=1e-5, max_iter=1e3, disp=False):
    """
    Fixed point iteration, used to calculate the infection probabilities Pi
    """
    i = 0
    e = 1
    while e > tol and i < max_iter:
      x = fun(x0, *args)
      e = (abs(x - x0)).sum()
      x0 = x
      i += 1
    if disp:
      print('success', e <= tol, sep=': ')
      print('iter', i, sep=': ')
    return x

  def J(self, fun, x, args=[], dx=1e-8):
    """
    Jacobian matrix calculator, to verify the suficient condition for the
    fixed point calculation of Pi
    """
    J = np.zeros([len(fun(x, *args)), len(x)])
    for j in range(len(x)):
      h = np.zeros(len(x))
      h[j] = dx
      J[:, j] = (fun(x+h, *args) - fun(x-h, *args))/(2*dx)
    return J

  # Network randomization functions
  def randomize_query_key(self, network, freq_1, freq_2=0):
    query = np.where(network == freq_1, 1, 0)
    key = np.where(network == freq_2, 1, 0)
    return query, key

  def randomize_find_central_banks(self, query, key):
    value = query.T @ key
    cbs_to_exchange = value.T * value
    cbs_to_exchange = np.where(np.triu(cbs_to_exchange) > 0)
    cbs_to_exchange = list(zip(cbs_to_exchange[0], cbs_to_exchange[1]))
    return cbs_to_exchange

  def randomize_find_words(self, query, key, cb_pair):
    w_1 = np.where(query[:,cb_pair[0]] * key[:,cb_pair[1]] > 0)
    w_2 = np.where(query[:,cb_pair[1]] * key[:,cb_pair[0]] > 0)
    word_pairs = np.meshgrid(w_1, w_2)
    word_pairs = np.array(word_pairs).T.reshape(-1,2)
    word_pairs = [(w[0], w[1]) for w in word_pairs]
    return word_pairs

  def randomize_pair_exchange(self, network, word_pair, cb_pair):
    words = list(word_pair)
    cbs = list(cb_pair)
    network_copy = network.copy()
    network_copy.iloc[words, cbs] = network_copy.iloc[words, cbs[::-1]]
    return network_copy

  def randomize_pair_mask(self, network, word_pair, cb_pair):
    """
    Masks the exchanged pair in the original network so that query and key
    won't find them, to avoid double excahnge
    """
    words = list(word_pair)
    cbs = list(cb_pair)
    network_copy = network.copy()
    network_copy.iloc[words, cbs] = -1
    return network_copy

  def randomize_single_exchange(self, network, masked_network,
                                freq_exclusions=[0]):
    rng = np.random.default_rng()
    network_copy = network.copy()
    masked_network_copy = masked_network.copy()
    freq_exclusions_copy = freq_exclusions.copy()
    
    #frequencies = np.unique(network_copy) # not weighted
    frequencies = network.to_numpy().flat # weighted
    frequencies = np.setdiff1d(frequencies, freq_exclusions_copy)
    freq = rng.choice(frequencies)
    
    q, k = self.randomize_query_key(masked_network_copy, freq)
    cbs_to_exchange = self.randomize_find_central_banks(q, k)
    
    # only move on if there is somethong to exchange
    if cbs_to_exchange:
      cb_pair = rng.choice(cbs_to_exchange)
      
      word_pairs = self.randomize_find_words(q, k, cb_pair)
      word_pair = rng.choice(word_pairs)

      network_copy = self.randomize_pair_exchange(network_copy,
                                                  word_pair,
                                                  cb_pair,
                                                  )
      masked_network_copy = self.randomize_pair_mask(masked_network_copy,
                                                     word_pair,
                                                     cb_pair,
                                                     )
      success = True
    else:
      freq_exclusions_copy.append(int(freq))
      success = False
    return network_copy, masked_network_copy, freq_exclusions_copy, success
  
  def randomize_two_freq_exchange(self, network, masked_network,
                                  freq_exclusions=[]):
    rng = np.random.default_rng()
    network_copy = network.copy()
    masked_network_copy = masked_network.copy()
    freq_exclusions_copy = freq_exclusions.copy()
    
    #frequencies = np.unique(network_copy) # not weighted
    frequencies = network.to_numpy().flat # weighted
    
    freq_1 = int(rng.choice(frequencies))
    while True:
      freq_2 = int(rng.choice(frequencies))
      if freq_2 != freq_1:
        break
    
    q, k = self.randomize_query_key(masked_network_copy, freq_1, freq_2)
    cbs_to_exchange = self.randomize_find_central_banks(q, k)
    
    # only move on if there is somethong to exchange
    # and the freq pair is not in the exclusion list
    if cbs_to_exchange and not ({freq_1, freq_2} in freq_exclusions_copy):
      cb_pair = rng.choice(cbs_to_exchange)
      
      word_pairs = self.randomize_find_words(q, k, cb_pair)
      word_pair = rng.choice(word_pairs)

      network_copy = self.randomize_pair_exchange(network_copy,
                                                  word_pair,
                                                  cb_pair,
                                                  )
      masked_network_copy = self.randomize_pair_mask(masked_network_copy,
                                                     word_pair,
                                                     cb_pair,
                                                     )
      success = True
    # if there is nothing to exchange, but freq pair is not in exclusion list
    elif not cbs_to_exchange and not {freq_1, freq_2} in freq_exclusions_copy:
      freq_exclusions_copy.append({freq_1, freq_2})
      success = False
    else:
      success = False
    return network_copy, masked_network_copy, freq_exclusions_copy, success
  
  def randomize_n_exchanges(self, network, n, two_freq=False, max_iter='N_len'):
    if max_iter == 'N_len':
      max_iter = network.shape[0] * network.shape[1]
    net_copy = network.copy()
    masked = network.copy()
    frequencies = np.unique(net_copy)
    success = False
    n_exchanges = 0
    iter = 0
    
    continue_condition = n_exchanges < n
    continue_condition *= iter < max_iter

    if two_freq:
      freq_excl = []
      func = self.randomize_two_freq_exchange
    else:
      freq_excl = [0]
      func = self.randomize_single_exchange
      continue_condition *= list(np.setdiff1d(frequencies, freq_excl))

    while continue_condition:
      net_copy, masked, freq_excl, success = func(net_copy,
                                                  masked,
                                                  freq_excl)
      # for not using the mask functionality
      # net_copy, masked, freq_excl, success = func(net_copy,
      #                                             net_copy,
      #                                             freq_excl)
      if success:
        n_exchanges += 1
      iter += 1
      continue_condition = n_exchanges < n
      continue_condition *= iter < max_iter
      if not two_freq:
        continue_condition *= list(np.setdiff1d(frequencies, freq_excl))
    return net_copy, n_exchanges

  def randomize_every_exchange(self, network):
    net_copy = network.copy()
    masked_net = network.copy()
    frequencies = np.unique(net_copy)
    success = False
    n_exchanges = 0
    freq_excl = [0]
    while list(np.setdiff1d(frequencies, freq_excl)):
      net_copy, masked_net, freq_excl, success = self.randomize_single_exchange(
          net_copy, masked_net, freq_excl)
      if success:
        n_exchanges += 1
    return net_copy, n_exchanges

  def randomize_permute_terms(self, network):
    """
    This version of the randomization algorithm does not restrain the total
    count for each term. It only guarantees that each speech will have the same
    count of words as before, not altering its original size.
    """
    rng = np.random.default_rng()
    network_copy = network.copy()
    
    random_net = rng.permuted(network_copy.to_numpy(),
                              axis=0)
    
    random_net = pd.DataFrame(random_net,
                              index=network_copy.index,
                              columns=network_copy.columns)
    n_exchanges = 0
    return random_net, n_exchanges

  def randomize_filter(self, Q, Q_rand, significance=0.05):
    Q_copy = Q.copy()
    threshold = np.percentile(Q_rand.to_numpy().flat, 100 * (1 - significance))
    Q_copy[Q_copy < threshold] = 0
    return Q_copy, threshold

  # Model calculation functions
  def run_single_period(self, y0='first', y1='last', randomize_tp=1,
                        n=10000, max_iter='N_len'):
    """
    Calculates a subset of the time horizon for the entire study, making it
    possible to analyse speech similarity in rolling windows.

    randomize_tp:
      0 for no random network generation
      1 for single freq, every exchange (default, according to article)
      2 for single freq, n exchanges
      3 for two freqs, n exchanges
      4 for term permuted network
    """
    if y0 == 'first':
      y0 = self.term_central_bank_year.head().columns.levels[1].min()
    if y1 == 'last':
      y1 = self.term_central_bank_year.head().columns.levels[1].max()

    term_central_bank_year_period = self.term_central_bank_year.loc[
        :, pd.IndexSlice[:, y0:y1]
        ]

    N = self.calc_N(term_central_bank_year_period)
    F, omega_local, omega_global, P = self.calc_F(N)
    Q, Q_numerator, Q_denominator, Q_norms = self.calc_Q(F)
    M = self.calc_M(Q)
    G = self.calc_G(M)

    Pi_0 = np.ones(M.shape[0])
    Pi_star = self.fixed_point_iter(self.Pi, Pi_0, args=[M])

    J_Pi_0 = self.J(self.Pi, Pi_0, args=[M])
    max_norm_J_Pi_0 = np.linalg.norm(J_Pi_0, ord=np.inf)
    J_Pi_star = self.J(self.Pi, Pi_star, args=[M])
    max_norm_J_Pi_star = np.linalg.norm(J_Pi_star, ord=np.inf)

    N_rand = None
    Q_rand = None
    n_exchanges_rand = None
    Q_filtered = None
    threshold = None

    if randomize_tp:
      if randomize_tp == 1:
        N_rand, n_exchanges_rand = self.randomize_every_exchange(N)

      elif randomize_tp == 2:
        N_rand, n_exchanges_rand = self.randomize_n_exchanges(N,
                                                              n,
                                                              two_freq=False,
                                                              max_iter=max_iter
                                                              )
      elif randomize_tp == 3:
        N_rand, n_exchanges_rand = self.randomize_n_exchanges(N,
                                                              n,
                                                              two_freq=True,
                                                              max_iter=max_iter
                                                              )
      elif randomize_tp == 4:
        N_rand, n_exchanges_rand = self.randomize_permute_terms(N)
      
      F_rand, omega_local_rand, omega_global_rand, P_rand = self.calc_F(N_rand)
      Q_rand, Q_numerator_rand, Q_denominator_rand, Q_norms_rand = self.calc_Q(
          F_rand)
      
      Q_filtered, threshold = self.randomize_filter(Q, Q_rand, 0.05)
      
      M_filtered = self.calc_M(Q_filtered)
      G_filtered = self.calc_G(M_filtered)

      Pi_filtered_star = self.fixed_point_iter(self.Pi, Pi_0, args=[M_filtered])

      J_Pi_filtered_star = self.J(self.Pi, Pi_filtered_star, args=[M_filtered])
      max_norm_J_Pi_filtered_star = np.linalg.norm(J_Pi_filtered_star,
                                                   ord=np.inf)

    results = {'N': N,
               'F': F,
               'omega_local': omega_local,
               'omega_global': omega_global,
               'P': P,
               'Q': Q,
               'Q_numerator': Q_numerator,
               'Q_denominator': Q_denominator,
               'Q_norms': Q_norms,
               'M': M,
               'G': G,
               'Pi_0': Pi_0,
               'Pi_star': Pi_star,
               'J_Pi_0': J_Pi_0,
               'max_norm_J_Pi_0': max_norm_J_Pi_0,
               'J_Pi_star': J_Pi_star,
               'max_norm_J_Pi_star': max_norm_J_Pi_star,
               'N_rand': N_rand,
               'n_exchanges_rand': n_exchanges_rand,
               'Q_rand': Q_rand,
               'Q_filtered': Q_filtered,
               'threshold': threshold,
               'M_filtered': M_filtered,
               'G_filtered': G_filtered,
               'Pi_filtered_star': Pi_filtered_star,
               'J_Pi_filtered_star': J_Pi_filtered_star,
               'max_norm_J_Pi_filtered_star': max_norm_J_Pi_filtered_star,
               }
    period_key = "-".join([str(y) for y in [y0, y1]])
    self.period_results[period_key] = results

  def run_full_study(self, start='first', end='last', interval=2, step=1,
                     randomize_tp=1, n=10000, max_iter='N_len'):
    """
    Runs the full study by calling run_single_period for every period we will
    analyse.
    """
    first = self.term_central_bank_year.columns.levels[1].min()
    last = self.term_central_bank_year.columns.levels[1].max()
    start_aux = first if start == 'first' else start
    end_aux = last if end == 'last' else end
    # validations
    valid = [start_aux < first, end_aux > last]
    if any(valid):
      raise ValueError('Invalid Parameters')
    for y in range(start_aux, end_aux + 1 - (interval - 1), step):
      self.run_single_period(y, y + interval - 1,
                             randomize_tp, n, max_iter)