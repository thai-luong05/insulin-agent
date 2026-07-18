export RUN_TS=$(date +%Y%m%d_%H%M%S)
OUT="bash_results/result_all/results_at_$RUN_TS"
echo "RUN_TS=$RUN_TS  ->  $OUT"
# python hycpap_full.py --meta --cohort adults
# _pools = {'adults': list(range(20, 30)), 'adolescent': list(range(0, 10)),
#               'child': list(range(10, 20)), 'all': list(range(30))}
for i in {0..29}
do
    python hybrid_policy.py --no-drl --patient_id $i --model lti
done
echo "===== ALL 30 DONE ====="
echo "Per-patient plots (each with a stats box under the graph) in: $OUT"
echo "Collected quantitative summary (one line per patient): $OUT/summary.txt"
