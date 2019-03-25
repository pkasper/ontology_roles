from __future__ import division
'''
Created on 14.01.2013

@author: psinger
'''

#import PathSim
import csv
from collections import defaultdict
import collections
import scipy.sparse as sp
import numpy as np
import sys
import math
import operator
from scipy import stats
from scipy.special import gammaln
from scipy.special import gamma
import copy
from random import choice
import copy

RESET_STATE = "-1"

UNKNOWN_STATE = 1

#we need this for k = 0
FAKE_ELEM = -10

#Prior
#PRIOR = 1.00

#we currently do not use this
#this means that we observe a state where we know the original page/feature
UNKNOWN_STATE_ELEM_KNOWN = -2
#this means that we observe a state where we DO NOT know the original page/feature
UNKNOWN_STATE_ELEM_UNKNOWN = -3

class MarkovChain():
    '''
    classdocs
    '''

    def __init__(self, k=1, reverse=False, use_prior=False, reset=True, prior=1., specific_prior = None, alpha = 0., modus="mle"):
        '''
        Constructor
        modus = specifies the modus of the class, there are two possibilities: modus='mle' is focused on working with mle matrices representing probabilities
        modus = 'bayes' focuses on working with bayesian evidence and only works with plain transition counts
        reverse = revert the paths
        use_prior = flag if script should use a prior
        reset = flag for using generic reset state
        prior = prior
        specific_prior = dictionary of specific priors for specific term combinations
        '''
        self.k_ = k
        self.reset_ = reset

        self.state_count_initial_ = 0
        self.states_initial_ = []
        self.parameter_count_ = 0
        self.observation_count_ = 0

        self.paths_ = list()
        self.paths_test_ = list()

        #probabilities
        self.transition_dict_ = defaultdict(lambda : defaultdict(float))
        self.transition_dict_text = defaultdict(lambda : defaultdict(float))

        self.prediction_position_dict_ = dict()
        self.vocabulary_ = None
        self.state_distr_ = defaultdict(float)

        self.states_ = dict()
        self.states_reverse_ = dict()
        self.dtype_ = np.dtype(float)
        self.reverse_ = reverse
        self.modus_ = modus

        self.use_prior_ = use_prior
        self.prior_ = prior
        #order?
        #print "spec prior", len(specific_prior)
        if specific_prior is None:
            self.specific_prior_ = defaultdict(lambda : defaultdict(float))
        else:
            self.specific_prior_ = specific_prior

        #print "spec prior", len(self.specific_prior_)
        #print self.specific_prior_
        if len(self.specific_prior_) > 0 and k != 1:
            raise Exception("using specific priors with higher orders not yet implemented")

        self.alpha_ = alpha

        self.proba_from_unknown_ = 0
        self.proba_to_unknown_ = dict()

    def _dict_divider(self, d):
        '''
        Internal function for dict divider and smoothing
        '''

        if self.use_prior_ == True:
            smoothing_divider = float(self.state_count_initial_ * self.prior_)
            print("smoothing divider: ", smoothing_divider)
            self.proba_from_unknown_ = self.prior_ / smoothing_divider
            print("proba_from_unknown_: ", self.proba_from_unknown_)

            for k, v in d.items():
                s = float(sum(v.values()))
                smoothing_divider = float(sum([round(x*self.alpha_)+self.prior_ for x in self.specific_prior_[k].values()]))
                smoothing_divider += float((self.state_count_initial_ - len(self.specific_prior_[k].values())) * self.prior_)

                divider = s + smoothing_divider
                self.observation_count_ += divider
                for i, j in v.items():
                    #print "v[i]", v[i]
                    v[i] = (j + self.prior_) / divider
                    #print "v[i] = j / s = ", v[i]

                self.proba_to_unknown_[k] = self.prior_ / divider
                #print "row sum: ", (float(sum(v.values())) + ((self.state_count_initial_ - len(v)) * self.proba_to_unknown_[k]))
        else:
            #print "hulululul"
            #print d
            for k, v in d.items():
                s = float(sum(v.values()))
                self.observation_count_ += s
                for i, j in v.items():
                    #print "v[i] = ", v[i]
                    try:
                        #v[i] = j / s
                        v[i] = j / s
                    except:
                        v[i] = 0.0
                    #print "v[i] = j / s = ", v[i]

                #print "row sum: ", float(sum(v.values()))

    # def _dict_ranker(self, d):
    #     '''
    #     Apply ranks to a dict according to the values
    #     Averages ties
    #     '''
    #     my_d = collections.defaultdict(list)
    #     for key, val in d.items():
    #         my_d[val].append(key)
    #
    #     ranked_key_dict = {}
    #     n = v = 1
    #     for _, my_list in sorted(my_d.items(), reverse=True):
    #         v = n + (len(my_list)-1)/2.
    #         for e in my_list:
    #             n += 1
    #             ranked_key_dict[e] = v
    #
    #     #little hack for storing the other unobserved average ranks
    #     #this is wanted so that we do not have to calculate it all the time again
    #     ranked_key_dict[FAKE_ELEM] = n + ((self.state_count_initial_-len(ranked_key_dict)-1)/2.)
    #
    #     return ranked_key_dict


    def _dict_ranker(self, d):
        '''
        Apply ranks to a dict according to the values
        Averages ties
        '''
        my_d = collections.defaultdict(list)
        for key, val in d.items():
            my_d[val].append(key)

        ranked_key_dict = {}
        n = v = 1
        for _, my_list in sorted(my_d.items(), reverse=True):
            #v = n + (len(my_list)-1)/2.
            v = n + len(my_list)-1
            for e in my_list:
                n += 1
                ranked_key_dict[e] = v

        #little hack for storing the other unobserved average ranks
        #this is wanted so that we do not have to calculate it all the time again
        #ranked_key_dict[FAKE_ELEM] = n + ((self.state_count_initial_-len(ranked_key_dict)-1)/2.)
        ranked_key_dict[FAKE_ELEM] = self.state_count_initial_

        return ranked_key_dict


    def prepare_data(self, paths):
        '''
        preparing data
        ALWAYS CALL FIRST
        '''
        states = set()
        if self.reset_:
            states.add(RESET_STATE)

        for line in paths:
            for ele in line:
                states.add(ele)
                self.state_distr_[ele] += 1

        #print self.state_distr_

        self.states_initial_ = frozenset(states)

        sum_state_occ = sum(self.state_distr_.values())
        for k,v in self.state_distr_.items():
            self.state_distr_[k] = float(v) / float(sum_state_occ)

        #self.state_count_ = math.pow(float(len(states)), self.k_)
        self.state_count_initial_ = float(len(states))
        self.parameter_count_ = pow(self.state_count_initial_, self.k_) * (self.state_count_initial_ - 1)

        #print "initial state count", self.state_count_initial_
        #print self.states_initial_
        #print "parameter count", self.parameter_count_
        #sys.exit()

    def fit(self, paths, ret=False, vocab=None):
        '''
        fitting the data and constructing MLE
        '''
        #print "====================="
        #print "K: ", self.k_

        for line in paths:
            if self.reset_:
                self.paths_.append(self.k_*[RESET_STATE] + [x for x in line] + [RESET_STATE])
            else:
                self.paths_.append([x for x in line])

        states = dict()
        state_cnt = 0
        observations = 0
        for path in self.paths_:
            i = 0
            for j in range(self.k_, len(path)):
                elemA = tuple(path[i:j])
                #print elemA
                i += 1
                #elemB = tuple(path[j:j+self.k_])
                elemB = path[j]
                if self.k_ == 0:
                    self.transition_dict_[FAKE_ELEM][elemB] += 1
                else:
                    #print "trans_dict[%s][%s] += 1" % (elemA, elemB)
                    self.transition_dict_[elemA][elemB] += 1
                observations += 1

        if vocab is not None:
            for i in vocab:
                for j in vocab:
                    if self.transition_dict_[(i,)][j] > 0:
                        continue
                    else:
                        self.transition_dict_[(i,)][j] = 0

        #print self.transition_dict_
        #print self.paths_


        if self.modus_ == "mle":
            self._dict_divider(self.transition_dict_)
        #for k,v in self.transition_dict_.items():
            #print k, v
            #print self.proba_to_unknown_[k]

        #print "Transition dict:"
        #print self.transition_dict_

        if ret:
            return self.transition_dict_

        #sys.exit()



    def loglikelihood(self):
        '''
        Calculating the log likelihood of the fitted MLE
        '''
        likelihood = 0
        counter = 0
        prop_counter = 0

        for path in self.paths_:
            i = 0
            for j in range(self.k_, len(path)):
                elemA = tuple(path[i:j])
                i += 1
                elemB = path[j]
                if self.k_ == 0:
                    prop = self.transition_dict_[FAKE_ELEM][elemB]
                else:
                    prop = self.transition_dict_[elemA][elemB]
                #print "elemA {} - elemB {} :: {}".format(elemA, elemB, prop)
                likelihood += math.log(prop)
                prop_counter += 1

        #print "likelihood", likelihood
        #print "prop_counter", prop_counter
        return likelihood

    def bayesian_evidence(self):
        '''
        Calculating the bayesian evidence of the fitted MLE
        '''
        if self.modus_ != "bayes":
            raise Exception("Bayesian evidence does not work with modus='mle'")

        print("starting to do bayesian evidence calculation")

        evidence = 0

        #the first term is always the same
#        first_term_enum = 0
#        first_term_denom = 0
#        for k in self.states_:
#
#        first_term_enum = gammaln(self.state_count_initial_ * self.prior_)
#        first_term_denom = self.state_count_initial_ * gammaln(self.prior_)
#        first_term = first_term_enum - first_term_denom
#

        i = 0
        for k,v in self.transition_dict_.items():
            #print k
            i += 1
            #if i % 100 == 0:
                #print i
            #print k,v
            #print first_term
            #print math.exp(first_term)
            first_term_enum = 0
            first_term_denom = 0
            second_term_enum = 0
            second_term_denom = 0

#            first_term_enum = gammaln(self.state_count_initial_ * self.prior_)
#            first_term_denom = self.state_count_initial_ * gammaln(self.prior_)
#            first_term = first_term_enum - first_term_denom

            done_counter = 0
            for x, c in v.items():
                if self.specific_prior_[k][x]:
                    prior = round(self.specific_prior_[k][x] * self.alpha_) + self.prior_
                else:
                    prior = self.prior_

                #count + prior
                cp = c + prior

                first_term_enum += prior
                first_term_denom += gammaln(prior)

                second_term_enum += gammaln(cp)
                second_term_denom += cp

                done_counter += 1

            for c in [round(b*self.alpha_)+self.prior_ for a,b in self.specific_prior_[k].items() if a not in v.keys()] :
                first_term_enum += c
                first_term_denom += gammaln(c)

                second_term_enum += gammaln(c)
                second_term_denom += c

                done_counter += 1



            non_trans_count = int(self.state_count_initial_ - done_counter)

            first_term_enum += (self.prior_ * non_trans_count)

            #print "first term enum", first_term_enum

            first_term_denom += (gammaln(self.prior_) * non_trans_count)

            second_term_enum += (gammaln(self.prior_) * non_trans_count)
            second_term_denom += (self.prior_ * non_trans_count)
#                second_term_enum += gammaln(self.prior_)
#                second_term_denom += self.prior_

            first_term_enum = gammaln(first_term_enum)
            first_term = first_term_enum - first_term_denom

            second_term_denom = gammaln(second_term_denom)
            second_term = second_term_enum - second_term_denom


            evidence += (first_term + second_term)
            #print "woo"
            #print "%.30f" %evidence

            #sys.exit()
        #evidence = math.exp(evidence)
        print("final: %.30f" %evidence)
        print(evidence)
        return evidence

    def predict(self, test):
        '''
        Predicting sequencies using MLE
        '''

        if self.modus_ != 'mle':
            raise Exception("Prediction only works with MLE mode!")

        if self.use_prior_ != True:
            raise Exception("Prediction only works with smoothing on!")

        for k,v in self.transition_dict_.items():
            #print v
            self.prediction_position_dict_[k] = self._dict_ranker(v)

        known_states = frozenset(self.transition_dict_.keys())

        for line in test:
            #if self.k
            self.paths_test_.append(self.k_*[RESET_STATE] + [x for x in line] + [RESET_STATE])

        position = 0
        counter = 0
        print("clicks test", len(self.paths_test_))

        for path in self.paths_test_:
            i = 0
            for j in range(self.k_, len(path)):
                elem = tuple(path[i:j])
                i += 1
                true_elem = path[j]

                if self.k_ == 0:
                    p = self.prediction_position_dict_[FAKE_ELEM].get(true_elem,self.prediction_position_dict_[FAKE_ELEM][FAKE_ELEM])
                else:
                    #We go from an unknown state to some other
                    #We come up with an uniform prob distribution
                    if elem not in known_states:
                        p = self.state_count_initial_
                    #We go from a known/learned state to some other
                    else:
                        p = self.prediction_position_dict_[elem].get(true_elem,self.prediction_position_dict_[elem][FAKE_ELEM])

                position += p
                counter += 1


        average_pos = position / counter
        #print "unknown elem counter", unknown_elem_counter
        print("counter", counter)
        print("average position", average_pos)
        return average_pos






