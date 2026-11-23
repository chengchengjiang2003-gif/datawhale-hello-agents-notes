import unittest
from practice import Memory, archive_candidates, retention_score, semantic_chunks


class PracticeTests(unittest.TestCase):
    def test_semantic_boundary_and_content(self):
        text = "向量用于检索。相似度用于排序。今天下雨了。出门需要带伞。"
        def embed(sentences):
            return [[1, 0], [1, 0], [0, 1], [0, 1]]
        chunks = semantic_chunks(text, embed)
        self.assertEqual(chunks, ["向量用于检索。相似度用于排序。", "今天下雨了。出门需要带伞。"])
        self.assertEqual(''.join(chunks), text)

    def test_empty_and_length_cap(self):
        self.assertEqual(semantic_chunks('', lambda _: []), [])
        text = '没有标点的长句' * 10
        chunks = semantic_chunks(text, lambda xs: [[1] for _ in xs], max_chars=12)
        self.assertTrue(all(len(c) <= 12 for c in chunks))
        self.assertEqual(''.join(chunks), text)

    def test_bad_embeddings(self):
        for fn in [lambda _: [], lambda _: [[float('nan')]], lambda _: [[]]]:
            with self.assertRaises(ValueError):
                semantic_chunks('一句话。', fn)

    def test_recency_and_frequency(self):
        old = Memory('old', .5, 1, 90)
        recent = Memory('recent', .5, 1, 0)
        frequent = Memory('frequent', .5, 50, 90)
        self.assertGreater(retention_score(recent), retention_score(old))
        self.assertGreater(retention_score(frequent), retention_score(old))

    def test_archive_protection_and_limit(self):
        memories = [Memory('pinned', 0, 0, 300, True), Memory('cold', .1, 0, 300),
                    Memory('active', .9, 100, 0)]
        self.assertEqual(archive_candidates(memories), ['cold'])
        self.assertEqual(archive_candidates(memories, max_count=0), [])

    def test_invalid_memory(self):
        for m in [Memory('a', 2, 0, 0), Memory('b', .5, -1, 0), Memory('c', .5, 1, -1)]:
            with self.assertRaises(ValueError):
                retention_score(m)


if __name__ == '__main__':
    unittest.main()
