import pandas as pd
import numpy as np
import re, math
from collections import Counter

# ── Load data ──────────────────────────────────────────────────────────────────
gold = pd.read_csv('final_gold_combined.csv')
hum  = pd.read_csv('human_aggregated.csv')
docs = pd.read_csv('clap_combined.csv')
docs = docs[["file_name", "gpt_description"]].copy()

LABEL_COLS = ["final_top1", "final_top2", "final_top3"]
GRADE      = {"final_top1": 3, "final_top2": 2, "final_top3": 1}
labels     = sorted(pd.unique(gold[LABEL_COLS].values.ravel()))
K          = 50

# ── Build qrels (from Building_QRels.ipynb) ────────────────────────────────────
qrels = {lab: {} for lab in labels}
for _, r in gold.iterrows():
    docid = r["filename"]
    for col, rel in GRADE.items():
        lab = r[col]
        qrels[lab][docid] = max(qrels[lab].get(docid, 0), rel)

all_docids = sorted(gold["filename"].tolist())

# ── Metrics (exact from Building_QRels.ipynb) ──────────────────────────────────
def dcg_at_k(rels, k):
    rels = rels[:k]
    s = 0.0
    for i, r in enumerate(rels, start=1):
        s += (2**r - 1) / math.log2(i + 1)
    return s

def ndcg_at_k(rels, ideal_rels, k):
    denom = dcg_at_k(ideal_rels, k)
    return 0.0 if denom == 0 else dcg_at_k(rels, k) / denom

def ap_at_k(binary_rels, k):
    binary_rels = binary_rels[:k]
    hits = 0; s = 0.0
    for i, r in enumerate(binary_rels, start=1):
        if r:
            hits += 1
            s += hits / i
    return 0.0 if hits == 0 else s / hits

def recall_at_k(binary_rels, total_relevant, k):
    if total_relevant == 0:
        return 0.0
    return sum(binary_rels[:k]) / total_relevant

def evaluate_run(ranked, qrel_for_query, k=50):
    rels   = [qrel_for_query.get(d, 0) for d in ranked]
    ideal  = sorted(qrel_for_query.values(), reverse=True)
    ndcg   = ndcg_at_k(rels, ideal, k)
    binary = [r > 0 for r in rels]
    ap     = ap_at_k(binary, k)
    rec    = recall_at_k(binary, total_relevant=len(qrel_for_query), k=k)
    return ndcg, ap, rec

# ── RANDOM baseline (from Building_QRels.ipynb) ────────────────────────────────
def random_run(all_docids, seed=0):
    rng    = np.random.default_rng(seed)
    ranked = all_docids.copy()
    rng.shuffle(ranked)
    return ranked

seeds = [0, 1, 2, 3, 4]
rand_perq = {q: [] for q in labels}
for seed in seeds:
    for q in labels:
        ranked = random_run(all_docids, seed=seed)
        ndcg, ap, rec = evaluate_run(ranked, qrels[q], k=K)
        rand_perq[q].append((ndcg, ap, rec))

random_results = {}
for q in labels:
    arr = np.array(rand_perq[q])
    random_results[q] = {'ndcg': arr[:,0].mean(), 'map': arr[:,1].mean(), 'recall': arr[:,2].mean()}

# ── CLAP baseline (from Building_QRels.ipynb) ──────────────────────────────────
CLAP_GRADE = {"clap_top1": 3, "clap_top2": 2, "clap_top3": 1}
clap_scores = {lab: {} for lab in labels}
for _, r in hum.iterrows():
    docid = r["filename"]
    for col, score in CLAP_GRADE.items():
        lab = r[col]
        clap_scores[lab][docid] = max(clap_scores[lab].get(docid, 0), score)

def rank_docs_for_label(score_dict, all_docids):
    return sorted(all_docids, key=lambda d: (-score_dict.get(d, 0), d))

clap_results = {}
for q in labels:
    ranked = rank_docs_for_label(clap_scores[q], all_docids)
    ndcg, ap, rec = evaluate_run(ranked, qrels[q], k=K)
    clap_results[q] = {'ndcg': ndcg, 'map': ap, 'recall': rec}

# ── BM25 baseline (from BM25_Benchmarking.ipynb) ──────────────────────────────
def tokenize(text):
    return re.findall(r"[a-z0-9]+", str(text).lower())

class SimpleBM25:
    def __init__(self, corpus_tokens, k1=1.2, b=0.75):
        self.k1, self.b = k1, b
        self.corpus = corpus_tokens
        self.N = len(corpus_tokens)
        self.doc_len = np.array([len(d) for d in corpus_tokens], dtype=float)
        self.avgdl = float(np.mean(self.doc_len)) if self.N > 0 else 0.0
        df = Counter()
        for doc in corpus_tokens:
            for w in set(doc):
                df[w] += 1
        self.df = df
        self.idf = {w: math.log(1 + (self.N - df[w] + 0.5) / (df[w] + 0.5)) for w in df}
        self.tf = [Counter(doc) for doc in corpus_tokens]

    def score(self, query_tokens):
        scores = np.zeros(self.N, dtype=float)
        for i in range(self.N):
            dl = self.doc_len[i]
            denom_const = self.k1 * (1 - self.b + self.b * (dl / self.avgdl if self.avgdl > 0 else 0.0))
            tf_i = self.tf[i]
            s = 0.0
            for w in query_tokens:
                if w not in self.idf: continue
                f = tf_i.get(w, 0)
                if f == 0: continue
                s += self.idf[w] * (f * (self.k1 + 1)) / (f + denom_const)
            scores[i] = s
        return scores

gold_docids = set(gold["filename"].astype(str))
docs["file_name"] = docs["file_name"].astype(str)
docs = docs[docs["file_name"].isin(gold_docids)].drop_duplicates("file_name").copy()
docs = docs.sort_values("file_name").reset_index(drop=True)

docids_bm25    = docs["file_name"].tolist()
corpus_tokens  = [tokenize(t) for t in docs["gpt_description"].fillna("")]
bm25           = SimpleBM25(corpus_tokens, k1=1.2, b=0.75)

def label_to_query_text(label):
    return label.replace("-", " ")

bm25_results = {}
for q in labels:
    q_tokens   = tokenize(label_to_query_text(q))
    scores     = bm25.score(q_tokens)
    ranked_idx = np.lexsort((np.array(docids_bm25), -scores))
    ranked     = [docids_bm25[i] for i in ranked_idx]
    ndcg, ap, rec = evaluate_run(ranked, qrels[q], k=K)
    bm25_results[q] = {'ndcg': ndcg, 'map': ap, 'recall': rec}

# ── Print results ──────────────────────────────────────────────────────────────
print(f"\n{'Query':<22} {'nDCG':>7} {'MAP':>7} {'Rec':>7}   {'nDCG':>7} {'MAP':>7} {'Rec':>7}   {'nDCG':>7} {'MAP':>7} {'Rec':>7}")
print(f"{'':22} {'--- Random ---':^23}   {'--- BM25 ---':^23}   {'--- CLAP ---':^23}")
print("-" * 95)
for q in labels:
    r = random_results[q]; b = bm25_results[q]; c = clap_results[q]
    print(f"{q:<22} {r['ndcg']:>7.3f} {r['map']:>7.3f} {r['recall']:>7.3f}   "
          f"{b['ndcg']:>7.3f} {b['map']:>7.3f} {b['recall']:>7.3f}   "
          f"{c['ndcg']:>7.3f} {c['map']:>7.3f} {c['recall']:>7.3f}")
print("-" * 95)
for name, res in [('Random', random_results), ('BM25', bm25_results), ('CLAP', clap_results)]:
    macro = {m: np.mean([res[q][m] for q in labels]) for m in ['ndcg','map','recall']}
    print(f"{'Macro-avg (' + name + ')':<22} {macro['ndcg']:>7.3f} {macro['map']:>7.3f} {macro['recall']:>7.3f}")

# # ── Generate LaTeX ─────────────────────────────────────────────────────────────
# def fmt(v): return f"{v:.3f}"

# latex = r"""\begin{table}[t]
# \centering
# \small
# \caption{Per-query retrieval performance across the eight therapeutic intent labels at $k=50$. Best result per metric per query in \textbf{bold}.}
# \label{tab:perquery_results}
# \resizebox{\columnwidth}{!}{%
# \begin{tabular}{l ccc ccc ccc}
# \toprule
# & \multicolumn{3}{c}{\textbf{Random}} & \multicolumn{3}{c}{\textbf{BM25}} & \multicolumn{3}{c}{\textbf{CLAP}} \\
# \cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}
# \textbf{Query} & nDCG & MAP & Rec & nDCG & MAP & Rec & nDCG & MAP & Rec \\
# \midrule
# """

# metrics = ['ndcg', 'map', 'recall']
# method_keys = [('Random', random_results), ('BM25', bm25_results), ('CLAP', clap_results)]

# for q in labels:
#     vals = {name: res[q] for name, res in method_keys}
#     row_parts = [f"\\textit{{{q}}}"]
#     for name, res in method_keys:
#         for m in metrics:
#             v = res[q][m]
#             # Bold if best across methods for this query+metric
#             best = max(random_results[q][m], bm25_results[q][m], clap_results[q][m])
#             cell = f"\\textbf{{{fmt(v)}}}" if abs(v - best) < 1e-9 else fmt(v)
#             row_parts.append(cell)
#     latex += " & ".join(row_parts) + " \\\\\n"

# latex += r"\midrule" + "\n"
# for name, res in method_keys:
#     macro = {m: np.mean([res[q][m] for q in labels]) for m in metrics}
#     row = [f"\\textbf{{Macro ({name})}}", fmt(macro['ndcg']), fmt(macro['map']), fmt(macro['recall']),
#            "", "", "", "", "", ""]
#     latex += " & ".join(row) + " \\\\\n"

# latex += r"""\bottomrule
# \end{tabular}%
# }
# \end{table}"""

# with open('/mnt/user-data/outputs/perquery_table.tex', 'w') as f:
#     f.write(latex)

# print("\n\nLaTeX table saved.")
