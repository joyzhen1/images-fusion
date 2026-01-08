import sys
from tools.profile_models_min import main

sys.argv = [
    "profile_models_min.py",
    "--models", "Ours=./models/ours/model.py,StdConv=./models/stdCon/model.py",
    "--height", "480",
    "--width", "640",
    "--bs", "1",
    "--warmup", "10",
    "--runs", "100",
    "--device", "0",
    "--out_csv", "C:/Users/xxx/Project/results/efficiency_results.csv",
    "--out_md",  "C:/Users/xxx/Project/results/efficiency_results.md"
]

main()
