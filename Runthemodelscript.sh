export MODEL=LinkBERT-base
export MODEL_PATH=michiyasunaga/$MODEL
task=hotpot_hf
datadir=../data/qa/$task
outdir=runs/$task/$MODEL
mkdir -p $outdir
python3 -u qa/RunNLPfinal.py --model_name_or_path $MODEL_PATH \
  --train_file $datadir/train.json --validation_file $datadir/dev.json --test_file $datadir/test.json \
  --do_train --do_eval --do_predict --preprocessing_num_workers 10 \
  --per_device_train_batch_size 12 --gradient_accumulation_steps 2  \
  --learning_rate 3e-5 --num_train_epochs 4 --max_seq_length 384 --doc_stride 128 \
  --save_strategy no --evaluation_strategy steps --eval_steps 1000 --output_dir $outdir --overwrite_output_dir \
  |& tee $outdir/log.txt &
