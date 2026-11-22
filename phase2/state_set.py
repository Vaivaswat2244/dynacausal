#!/usr/bin/env python3
"""state_set.py KEY VALUE: record a cluster setting in state.json next to this
file. run_experiments.sh copies state.json into each batch's settings.json."""
import json, os, sys, time
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "state.json")
state = json.load(open(path)) if os.path.exists(path) else {}
state[sys.argv[1]] = sys.argv[2]
state[sys.argv[1] + "_set_unix"] = int(time.time())
json.dump(state, open(path, "w"), indent=1)
