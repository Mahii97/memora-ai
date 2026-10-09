class HybridRanker:
    VECTOR_WEIGHT = 0.60
    BM25_WEIGHT = 0.40

    def normalize(self, results, key):

        if len(results) == 0:
            return results

        values = [r[key] for r in results]

        mx = max(values)
        mn = min(values)

        if mx == mn:
            for r in results:
                r["normalized"] = 1.0
            return results

        for r in results:
            r["normalized"] = (r[key] - mn) / (mx - mn)

        return results

    def fuse(self, vector_results, bm25_results):

        vector_results = self.normalize(vector_results, "score")
        bm25_results = self.normalize(bm25_results, "score")

        merged = {}

        # ---------- Vector ----------
        for r in vector_results:

            cid = r["chunk"]["chunk_id"]

            merged[cid] = {
                "chunk": r["chunk"],
                "vector_score": r["score"],
                "normalized_vector_score": r["normalized"],
                "bm25_score": 0.0,
                "normalized_bm25_score": 0.0,
            }

        # ---------- BM25 ----------
        for r in bm25_results:

            cid = r["chunk"]["chunk_id"]

            if cid not in merged:

                merged[cid] = {
                    "chunk": r["chunk"],
                    "vector_score": 0.0,
                    "normalized_vector_score": 0.0,
                    "bm25_score": r["score"],
                    "normalized_bm25_score": r["normalized"],
                }

            else:

                merged[cid]["bm25_score"] = r["score"]
                merged[cid]["normalized_bm25_score"] = r["normalized"]

        final = []

        for item in merged.values():

            score = (
                self.VECTOR_WEIGHT * item["normalized_vector_score"]
                + self.BM25_WEIGHT * item["normalized_bm25_score"]
            )

            chunk = dict(item["chunk"])
            chunk.update(
                {
                    "vector_score": item["vector_score"],
                    "normalized_vector_score": item["normalized_vector_score"],
                    "bm25_score": item["bm25_score"],
                    "normalized_bm25_score": item["normalized_bm25_score"],
                    "hybrid_score": round(score, 3),
                }
            )

            final.append(chunk)

        final.sort(
            key=lambda x: x["hybrid_score"],
            reverse=True
        )

        return final