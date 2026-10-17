"""GitHub issue deduplicator on top of the System One decision API.

Asks a decision model (S1-CED, Vega or Jev) whether a new GitHub issue duplicates
a prior one, how bad a wrong merge would be and whether both describe the same
root symptom, then recommends "Suggest merge" or "Manual review".

The questions and the merge gate come from github-issue-deduplicator.json, the
same file the playground loads with --examples, so the two never drift apart.
Standard library only.

  # start the playground (or `s1ced --serve`) first, then:
  python examples/issue_dedup.py --new "Export CSV hangs for 30 seconds in Safari" \\
                                 --prior "Safari CSV export never finishes"
  python examples/issue_dedup.py --check        # run the scenario's cases against their expected answers
  python examples/issue_dedup.py --new ... --prior ... --request-only   # print the request, call nothing
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

SCENARIO = Path(__file__).resolve().parent / 'github-issue-deduplicator.json'
DEFAULT_URL = 'http://127.0.0.1:8770'
MERGE, REVIEW = 'Suggest merge', 'Manual review'
HIGH_IMPACT = 3                     # score level "high": a wrong merge this costly always goes to a human


class DecisionError(Exception):
    """The decision API answered with an error, or could not be reached."""


def load_scenario(path=SCENARIO):
    group = json.loads(Path(path).read_text(encoding='utf-8'))['groups'][0]
    return group


def build_request(new_issue, prior_issue, model='s1ced', scenario=None):
    """The System One request for one pair of issues."""
    scenario = scenario or load_scenario()
    state = f'New issue: {new_issue.strip()}\nPrior issue: {prior_issue.strip()}'
    return {'model': model, 'state': state, 'questions': scenario['questions']}


def decide(request, base_url=DEFAULT_URL, api_key=None, timeout=120):
    """POST the request to /v1/systemone and return the parsed response."""
    headers = {'Content-Type': 'application/json'}
    key = api_key or os.environ.get('TYPESAFE_API_KEY')
    if key:
        headers['Authorization'] = f'Bearer {key}'
    req = urllib.request.Request(base_url.rstrip('/') + '/v1/systemone', data=json.dumps(request).encode(),
                                 headers=headers, method='POST')
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            detail = json.loads(e.read()).get('detail', {})
            msg = detail.get('message') if isinstance(detail, dict) else str(detail)
        except (ValueError, AttributeError):
            msg = e.reason
        raise DecisionError(f'{e.code}: {msg}') from None
    except urllib.error.URLError as e:
        raise DecisionError(f'cannot reach {base_url} ({e.reason}). Is s1ced-playground running?') from None


def recommend(response, scenario=None):
    """Turn the model's answers into a recommendation.

    Suggest a merge only when every signal agrees: the verdict is "duplicate", the
    same-root-symptom probability clears the scenario's gate, a wrong merge would not
    be high impact, and the model does not flag either deciding answer for review.
    Anything else goes to a human, with the reasons listed.
    """
    scenario = scenario or load_scenario()
    gate = scenario['gates']['same_root_symptom']
    levels = scenario['questions']['wrong_merge_impact']['criteria']
    verdicts = scenario['questions']['verdict']['criteria']
    a = response['answers']
    verdict, same, impact = a['verdict'], a['same_root_symptom'], a['wrong_merge_impact']
    level = top_level(impact)

    reasons = []
    if verdict['choice'] != 'duplicate':
        reasons.append(f'verdict is "{verdicts.get(verdict["choice"], verdict["choice"])}", not a likely duplicate')
    if same['noul'] < gate['threshold']:
        reasons.append(f'same root symptom at {same["noul"]:.2f}, below the {gate["threshold"]} gate')
    if level >= HIGH_IMPACT:
        reasons.append(f'a wrong merge would be {levels[level]}')
    for key, ans in (('verdict', verdict), ('same_root_symptom', same)):
        if (ans.get('uncertainty') or {}).get('needs_review'):
            reasons.append(f'the model is unsure about {key}')

    return {
        'action': REVIEW if reasons else MERGE,
        'reasons': reasons,
        'verdict': verdict['choice'],
        'verdict_label': verdicts.get(verdict['choice'], verdict['choice']),
        'verdict_confidence': verdict.get('confidence'),
        'same_root_symptom': same['noul'],
        'wrong_merge_impact': level,
        'wrong_merge_impact_label': levels[level],
        'model': response.get('model'),
    }


def top_level(score_answer):
    """The most likely score level, as an int."""
    probs = score_answer.get('probabilities') or {}
    if probs:
        return int(max(probs, key=probs.get))
    return round(score_answer['score'])


def matches_expected(response, expected):
    """Per-question True/False against a scenario item's expected answers."""
    a, out = response['answers'], {}
    for key, want in expected.items():
        ans = a[key]
        if ans['type'] == 'noul':
            out[key] = ('yes' if ans['noul'] >= 0.5 else 'no') == want
        elif ans['type'] == 'score':
            lo, hi = (want, want) if isinstance(want, int) else (min(want), max(want))
            out[key] = lo <= top_level(ans) <= hi
        else:
            out[key] = ans['choice'] == want
    return out


def format_result(rec):
    lines = [f'{rec["action"]}',
             f'  verdict:            {rec["verdict_label"]}'
             + (f' ({rec["verdict_confidence"]:.2f})' if rec['verdict_confidence'] is not None else ''),
             f'  same root symptom:  {rec["same_root_symptom"]:.2f}',
             f'  wrong-merge impact: {rec["wrong_merge_impact_label"]}']
    lines += [f'  why review: {r}' for r in rec['reasons']]
    return '\n'.join(lines)


def run_check(args, scenario):
    total = right = 0
    for item in scenario['items']:
        req = {'model': args.model, 'state': item['state'], 'questions': scenario['questions']}
        resp = decide(req, args.url, args.api_key)
        hits = matches_expected(resp, item['expected'])
        total += len(hits)
        right += sum(hits.values())
        print(f'{item["name"]}: {sum(hits.values())}/{len(hits)} right'
              + ''.join(f'\n  {k}: {"ok" if ok else "MISS"}' for k, ok in hits.items()))
        print('  ' + format_result(recommend(resp, scenario)).replace('\n', '\n  '))
    print(f'\n{right}/{total} expected answers right')
    return 0 if right == total else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description='Check whether a new GitHub issue duplicates a prior one.')
    ap.add_argument('--new', help='text of the new issue (title, and body if you like)')
    ap.add_argument('--prior', help='text of the prior issue')
    ap.add_argument('--model', default='s1ced', help='s1ced (default), vega-0.8b, vega-4b or jev-latest')
    ap.add_argument('--url', default=os.environ.get('S1CED_URL', DEFAULT_URL), help=f'decision API (default {DEFAULT_URL})')
    ap.add_argument('--api-key', help='TypeSafe key for Jev (or set TYPESAFE_API_KEY)')
    ap.add_argument('--scenario', type=Path, default=SCENARIO, help='scenario file with the questions and gate')
    ap.add_argument('--json', action='store_true', help='print the recommendation as JSON')
    ap.add_argument('--request-only', action='store_true', help='print the request JSON and stop')
    ap.add_argument('--check', action='store_true', help="run the scenario's cases and compare with their expected answers")
    args = ap.parse_args(argv)
    scenario = load_scenario(args.scenario)

    try:
        if args.check:
            return run_check(args, scenario)
        if not (args.new and args.prior):
            ap.error('give --new and --prior, or --check')
        req = build_request(args.new, args.prior, args.model, scenario)
        if args.request_only:
            print(json.dumps(req, indent=2))
            return 0
        rec = recommend(decide(req, args.url, args.api_key), scenario)
    except DecisionError as e:
        print(f'error: {e}', file=sys.stderr)
        return 2
    print(json.dumps(rec, indent=2) if args.json else format_result(rec))
    return 0


if __name__ == '__main__':
    sys.exit(main())
