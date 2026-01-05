import sys
import copy
import torch
import random
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Queue

def build_index(dataset_name):

    ui_mat = np.loadtxt('data/%s.txt' % dataset_name, dtype=np.int32)

    n_users = ui_mat[:, 0].max()
    n_items = ui_mat[:, 1].max()

    u2i_index = [[] for _ in range(n_users + 1)]
    i2u_index = [[] for _ in range(n_items + 1)]

    for ui_pair in ui_mat:
        u2i_index[ui_pair[0]].append(ui_pair[1])
        i2u_index[ui_pair[1]].append(ui_pair[0])

    return u2i_index, i2u_index

# sampler for batch generation
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED):
    def sample(uid):

        # uid = np.random.randint(1, usernum + 1)
        while len(user_train[uid]) <= 1: uid = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[uid][-1]
        idx = maxlen - 1

        ts = set(user_train[uid])
        for i in reversed(user_train[uid][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            neg[idx] = random_neq(1, itemnum + 1, ts)          # Don't need "if nxt != 0"
            nxt = i
            idx -= 1
            if idx == -1: break

        return (uid, seq, pos, neg)

    np.random.seed(SEED)
    uids = np.arange(1, usernum+1, dtype=np.int32)
    counter = 0
    while True:
        if counter % usernum == 0:
            np.random.shuffle(uids)
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample(uids[counter % usernum]))
            counter += 1
        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()


# train/val/test data generation
# def data_partition(fname):
#     usernum = 0
#     itemnum = 0
#     User = defaultdict(list)
#     user_train = {}
#     user_valid = {}
#     user_test = {}
#     # assume user/item index starting from 1
#     f = open('data/%s.txt' % fname, 'r')
#     for line in f:
#         u, i = line.rstrip().split(' ')
#         u = int(u)
#         i = int(i)
#         usernum = max(u, usernum)
#         itemnum = max(i, itemnum)
#         User[u].append(i)

#     for user in User:
#         nfeedback = len(User[user])
#         if nfeedback < 4:                          # To be rigorous, the training set needs at least two data points to learn
#             user_train[user] = User[user]
#             user_valid[user] = []
#             user_test[user] = []
#         else:
#             user_train[user] = User[user][:-2]
#             user_valid[user] = []
#             user_valid[user].append(User[user][-2])
#             user_test[user] = []
#             user_test[user].append(User[user][-1])
#     return [user_train, user_valid, user_test, usernum, itemnum]

def data_partition(fname, min_cold_freq=5, max_cold_freq=20):
    """
    Controlled ZERO-SHOT item cold-start split.

    - Define cold item set C using global item frequency on ALL interactions:
        C = { i : min_cold_freq <= global_count[i] <= max_cold_freq }
    - Train contains NO items in C (zero-shot).
    - For users who interacted with any cold item:
        test = last cold interaction in the user's timeline
        valid = last warm interaction before test (if exists)
        train = all earlier warm interactions (excluding cold items)
    - For users with no cold interactions:
        fall back to the original split:
            if nfeedback < 4: all train
            else: last -> test, second last -> valid, rest train
    """
    usernum = 0
    itemnum = 0
    User = defaultdict(list)

    # Load full interactions (already time-sorted in your file)
    f = open('data/%s.txt' % fname, 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)
    f.close()

    # 1) Global item frequency across ALL interactions
    global_item_count = defaultdict(int)
    for u in User:
        for i in User[u]:
            global_item_count[i] += 1

    # 2) Cold item set C
    cold_items = set()
    for i, c in global_item_count.items():
        if min_cold_freq <= c <= max_cold_freq:
            cold_items.add(i)

    user_train = {}
    user_valid = {}
    user_test = {}

    for u in User:
        seq = User[u]
        nfeedback = len(seq)

        # Find last index in this user's timeline whose item is cold
        last_cold_idx = -1
        for idx in range(nfeedback - 1, -1, -1):
            if seq[idx] in cold_items:
                last_cold_idx = idx
                break

        if last_cold_idx != -1:
            # -------- Controlled zero-shot user split --------
            # test is last cold item
            test_item = seq[last_cold_idx]

            # valid is last warm item before test, if any
            valid_item = None
            for idx in range(last_cold_idx - 1, -1, -1):
                if seq[idx] not in cold_items:
                    valid_item = seq[idx]
                    break

            # train: all warm items before test (exclude cold items)
            train_seq = [x for x in seq[:last_cold_idx] if x not in cold_items]

            user_train[u] = train_seq
            user_valid[u] = [valid_item] if valid_item is not None else []
            user_test[u]  = [test_item]

        else:
            # -------- Fallback: original split --------
            if nfeedback < 4:
                user_train[u] = seq
                user_valid[u] = []
                user_test[u] = []
            else:
                user_train[u] = seq[:-2]
                user_valid[u] = [seq[-2]]
                user_test[u]  = [seq[-1]]

    return [user_train, user_valid, user_test, usernum, itemnum]

def evaluate_cold_test(model, dataset, args, k=0):
    """
    Evaluate only TEST cases where the ground-truth test item is cold by TRAIN frequency.
    cold iff train_count(test_item) <= k.
    For zero-shot: k=0.
    """
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    # Count item frequencies in TRAIN
    train_item_count = defaultdict(int)
    for u in train:
        for i in train[u]:
            train_item_count[i] += 1

    NDCG = 0.0
    HT = 0.0
    cold_users = 0.0

    users = range(1, usernum + 1)
    for u in users:
        if len(train[u]) < 1 or len(test[u]) < 1:
            continue

        gt_item = test[u][0]
        if train_item_count.get(gt_item, 0) > k:
            continue  # not cold

        # Build sequence same way as evaluate()
        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1

        # In evaluate() for test: it prepends valid[u][0] to the sequence
        if len(valid[u]) > 0:
            seq[idx] = valid[u][0]
            idx -= 1

        for i in reversed(train[u]):
            if idx == -1:
                break
            seq[idx] = i
            idx -= 1

        rated = set(train[u])
        rated.add(0)

        item_idx = [gt_item]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated:
                t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0]
        rank = predictions.argsort().argsort()[0].item()

        cold_users += 1
        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1

        # DEBUG: print first few cold cases
        if cold_users <= 5:
            print(f"[cold dbg] user={u}, gt_item={gt_item}, rank={rank}")


    if cold_users == 0:
        return (0.0, 0.0, 0)

    return (NDCG / cold_users, HT / cold_users, int(cold_users))


# TODO: merge evaluate functions for test and val set
# evaluate on test set
def evaluate(model, dataset, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:

        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        if len(valid[u]) > 0:
            seq[idx] = valid[u][0]
            idx -= 1
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        rated = set(train[u])
        rated.add(0)
        item_idx = [test[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0] # - for 1st argsort DESC

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user


# evaluate on val set
def evaluate_valid(model, dataset, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(train[u])
        rated.add(0)
        item_idx = [valid[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user
