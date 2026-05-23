#!/usr/bin/env bash
# Runs all four acceptance queries from a CLEAN state and tees each trace to
# traces/. Query C runs TWICE against the same state/ (records, then recalls).
#
#   ./run_all_tests.sh
#
# Prereqs: gateway up on :8101, .env set. Does NOT use `set -e` so one failure
# doesn't abort the rest.

cd "$(dirname "$0")"
mkdir -p traces

A="Fetch https://en.wikipedia.org/wiki/Claude_Shannon and tell me his birth date, death date, and three key contributions to information theory."
B="Find 3 family-friendly things to do in Tokyo this weekend. Check Saturday's weather forecast there and tell me which one is most appropriate."
C1="My mom's birthday is 15 May 2026. Remember that and give me a calendar reminder for two weeks before and on the day."
C2="When is mom's birthday?"
D="Search for 'Python asyncio best practices', read the top 3 results, and give me a short numbered list of the advice they agree on."

echo "########## QUERY A — Shannon ##########"
rm -rf state
uv run python agent6.py "$A" 2>&1 | tee traces/A_shannon.txt

echo "########## QUERY B — Tokyo ##########"
rm -rf state
uv run python agent6.py "$B" 2>&1 | tee traces/B_tokyo.txt

echo "########## QUERY C — Mom's birthday (run 1: record) ##########"
rm -rf state
uv run python agent6.py "$C1" 2>&1 | tee traces/C1_record.txt
echo "########## QUERY C — Mom's birthday (run 2: recall, SAME state/) ##########"
uv run python agent6.py "$C2" 2>&1 | tee traces/C2_recall.txt

echo "########## QUERY D — asyncio synthesis ##########"
rm -rf state
uv run python agent6.py "$D" 2>&1 | tee traces/D_asyncio.txt

echo ""
echo "Done. Traces saved under traces/.  (rm -rf state/ before a graded run.)"
