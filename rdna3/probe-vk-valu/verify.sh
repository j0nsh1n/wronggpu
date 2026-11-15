# Sourced by run.sh (needs its gpu / capped helpers): GPU quantizer outputs versus verify.py's CPU references.
capped python3 verify.py gen
gpu run spv/VERIFY_Q8.spv 16384 1 1 raw/verify_q8.in 16777216 zero:4 raw/verify_q8.out > logs/verify_q8.log
gpu run spv/VERIFY_RQ.spv 16384 1 1 raw/verify_rq.in 16777216 zero:4 raw/verify_rq.out > logs/verify_rq.log
gpu run spv/VERIFY_E4H.spv 1 1 1 zero:4 1024 zero:4 raw/verify_e4h.out > logs/verify_e4h.log
capped python3 verify.py check | tee logs/verify.txt
