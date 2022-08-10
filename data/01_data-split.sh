echo "stage 0: Data preparation"
echo "train/dev/eval split"
head -n 3133 utt_list.txt > train.list
tail -300 utt_list.txt > deveval.list
head -n 200 deveval.list > dev.list
tail -n 100 deveval.list > eval.list
rm -f deveval.list
