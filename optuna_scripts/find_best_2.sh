f1_max=0.0
f_f1_max=""
for f in $(find .  -regex '.*intra.*/roc.csv'); do

res=$(cat $f)

f1s=$(cat $f |  tail -n +2 | awk 'BEGIN {RS="\n";FS=","} {print $ 4}')

  for f1 in $f1s; do
    echo $f
    echo $f1_max
    if (( $f1 > $f1_max ));then
    f1_max=$f1
    f_f1_max=$f
    echo $f_f1_max
    fi
  done
done

echo $f_f1_max
echo $res
#cat $f_f1_max
