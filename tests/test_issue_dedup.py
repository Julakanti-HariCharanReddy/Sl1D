"""Tests for examples/issue_dedup.py. No model and no network: a fake local server answers."""
import io
import json
import sys
import threading
import unittest
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples'))
import issue_dedup as dd  # noqa: E402


def answers(verdict='duplicate', conf=0.9, same=0.9, impact=1, review=False):
    probs = {str(i): 0.02 for i in range(5)}
    probs[str(impact)] = 0.92
    unc = {'prediction_set': None, 'abstain': None, 'out_of_scope': None, 'needs_review': review}
    return {'model': 'fake', 'answers': {
        'verdict': {'type': 'choice', 'choice': verdict, 'confidence': conf, 'uncertainty': unc},
        'wrong_merge_impact': {'type': 'score', 'score': float(impact), 'probabilities': probs, 'uncertainty': unc},
        'same_root_symptom': {'type': 'noul', 'noul': same, 'uncertainty': unc}},
        'usage': {'input_tokens': 1, 'output_tokens': 0}, 'latency_ms': 1.0, 'warnings': []}


class FakeAPI(BaseHTTPRequestHandler):
    reply, status, seen = None, 200, []

    def do_POST(self):
        FakeAPI.seen.append((self.path, json.loads(self.rfile.read(int(self.headers['Content-Length'])))))
        body = json.dumps(FakeAPI.reply).encode()
        self.send_response(FakeAPI.status)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


class ScenarioTest(unittest.TestCase):
    def test_scenario_is_consistent(self):
        s = dd.load_scenario()
        q = s['questions']
        self.assertEqual(set(q), {'verdict', 'wrong_merge_impact', 'same_root_symptom'})
        self.assertEqual(s['gates']['same_root_symptom']['pass'], dd.MERGE)
        self.assertEqual(s['gates']['same_root_symptom']['fail'], dd.REVIEW)
        for item in s['items']:
            self.assertEqual(set(item['expected']), set(q))
            self.assertIn(item['expected']['verdict'], q['verdict']['criteria'])

    def test_build_request_follows_contract(self):
        req = dd.build_request(' Export CSV hangs ', 'Safari CSV export never finishes', 'vega-0.8b')
        self.assertEqual(set(req), {'model', 'state', 'questions'})
        self.assertEqual(req['model'], 'vega-0.8b')
        self.assertEqual(req['state'], 'New issue: Export CSV hangs\nPrior issue: Safari CSV export never finishes')


class RecommendTest(unittest.TestCase):
    def test_all_signals_agree_suggests_merge(self):
        rec = dd.recommend(answers())
        self.assertEqual(rec['action'], dd.MERGE)
        self.assertEqual(rec['reasons'], [])
        self.assertEqual(rec['wrong_merge_impact_label'], 'low')

    def test_each_failing_signal_sends_to_review(self):
        cases = {'not a likely duplicate': dict(verdict='related'),
                 'below the 0.7 gate': dict(same=0.69),
                 'would be high': dict(impact=3),
                 'unsure about verdict': dict(review=True)}
        for why, kw in cases.items():
            with self.subTest(why):
                rec = dd.recommend(answers(**kw))
                self.assertEqual(rec['action'], dd.REVIEW)
                self.assertTrue(any(why in r for r in rec['reasons']), rec['reasons'])

    def test_gate_threshold_is_inclusive(self):
        self.assertEqual(dd.recommend(answers(same=0.7))['action'], dd.MERGE)

    def test_matches_expected(self):
        exp = {'verdict': 'duplicate', 'wrong_merge_impact': [0, 1], 'same_root_symptom': 'yes'}
        self.assertEqual(dd.matches_expected(answers(), exp), {k: True for k in exp})
        self.assertEqual(dd.matches_expected(answers(verdict='new', impact=2, same=0.3), exp), {k: False for k in exp})


class HTTPTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = HTTPServer(('127.0.0.1', 0), FakeAPI)
        cls.url = f'http://127.0.0.1:{cls.srv.server_port}'
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def setUp(self):
        FakeAPI.reply, FakeAPI.status, FakeAPI.seen = answers(), 200, []

    def test_cli_posts_request_and_prints_merge(self):
        out = io.StringIO()
        with redirect_stdout(out):
            code = dd.main(['--new', 'Export CSV hangs for 30 seconds in Safari',
                            '--prior', 'Safari CSV export never finishes', '--url', self.url, '--json'])
        self.assertEqual(code, 0)
        path, body = FakeAPI.seen[0]
        self.assertEqual(path, '/v1/systemone')
        self.assertEqual(body['model'], 's1ced')
        self.assertEqual(json.loads(out.getvalue())['action'], dd.MERGE)

    def test_api_error_is_reported(self):
        FakeAPI.reply, FakeAPI.status = {'detail': {'error_type': 'max_tokens_exceeded', 'message': 'too long'}}, 400
        with self.assertRaisesRegex(dd.DecisionError, '400: too long'):
            dd.decide(dd.build_request('a', 'b'), self.url)

    def test_check_runs_every_case(self):
        out = io.StringIO()
        with redirect_stdout(out):
            code = dd.main(['--check', '--url', self.url])
        self.assertEqual(len(FakeAPI.seen), 2)
        self.assertEqual(code, 1)           # the fake gives the same answers to both cases, so one misses
        self.assertIn('3/6 expected answers right', out.getvalue())

    def test_unreachable_server(self):
        with self.assertRaisesRegex(dd.DecisionError, 'cannot reach'):
            dd.decide(dd.build_request('a', 'b'), 'http://127.0.0.1:9', timeout=2)


if __name__ == '__main__':
    unittest.main()
