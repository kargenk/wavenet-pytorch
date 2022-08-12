dump_org_dir=dump/jsut_sr16000/org

n_jobs=4
qst_path="qst1.hed"

db_root="jsut_ver1.1/basic5000"
train_set="train"
dev_set="dev"
eval_set="eval"
datasets=($train_set $dev_set $eval_set)

echo "stage 2: Feature generation for dulation model"
for s in ${datasets[@]}; do
    python ../dulation/preprocess_dulation.py $s.list $db_root/lab/ $qst_path \
        $dump_org_dir/$s --n_jobs $n_jobs
done
