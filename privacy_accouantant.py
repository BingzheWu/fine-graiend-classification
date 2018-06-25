import abc
import collections
import math
import sys
import numpy 
import torch as th
import privacy_utils
EpsDelta = collections.namedtuple("EpsDelta", ["spent_eps", "spent_delta"])

class MomentAccountant(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, total_patients, moment_orders = 32):
        """
        total patients: total number of selected patients in one step
        """
        assert total_patients > 0
        self._total_patients = total_patients
        self._moment_orders = (moment_orders if isinstance(moment_orders, (list, tuple))
                                         else range(1, moment_orders+1))
        self._max_moment_orders = max(self._moment_orders)
        self._log_moments = [th.tensor(numpy.float(0.0)) for moment_order in self._moment_orders]
    @abc.abstractmethod
    def _compute_log_moment(self, sigma, q, moment_order):
        pass
    
    def accumulate_privacy_spending(self, unused_eps_delta, sigma, num_patients):
        q = float(num_patients)/self._total_patients
        moments_accu_ops = []
        for i in range(len(self._log_moments)):
            moment = self._compute_log_moment(sigma, q, self._moment_orders[i])
            moments_accu_ops.append(tf.assign_add(self._log_moments[i], moment))
        return tf.group(*moments_accu_ops)
    def _compute_delta(self, log_moments, eps):
        min_delta = 1.0
        for moment_order, log_moment in log_moments:
            if math.isinf(log_moment) or math.isnan(log_moment):
                sys.stderr.write("The %d-th order is inf or nan\n"%moment_order)
                continue
            if log_moment < moment_order * eps:
                min_delta = min(min_delta, math.exp(log_moment-moment_order*eps))
        return min_delta
    def _compute_eps(self, log_moments, delta):
        min_eps = float("inf")
        for moment_order, log_moment in log_moments:
            if math.isinf(log_moment) or math.isnan(log_moment):
                continue
            min_eps = min(min_eps, (log_moment-math.log(delta))/moment_order)
    def get_privacy_spent(self, sess, target_eps = None, target_deltas = None):
        eps_deltas = []
        log_moments = sess.run(self._log_moments)
        log_moments_with_order = zip(self._moment_orders, log_moments)
        if target_eps is not None:
            for eps in target_eps:
                eps_deltas.append(EpsDelta(eps, self._compute_delta(log_moments_with_order, eps), eps))
        else:
            for delta in target_deltas:
                eps_deltas.append(EpsDelta(self._compute_eps(log_moments_with_order, delta), delta))
        return eps_deltas
    class GaussianMomentsAcc(MomentAccountant):
        def __init__(self, total_patients, moment_orders = 32):
            super(self.__class__, self).__init__(total_patients, moment_orders)
            self._binominal_table = privacy_utils.GnerateBinomialTable(self._max_moment_order)
        def _differential_moments(self, sigma, s, t):
            pass