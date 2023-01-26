f1_max=0
for f in $(find .  -regex '.*intra.*/bbc.csv'); do

res=$(cat $f)
f1=$(cat $f |  tail -n1 | awk 'BEGIN {RS="\n";FS=","} {print $ 5}')
if [[ "$f1" -gt "$f1_max" ]];then
  echo $f1_max
  f1_max=$f1
  f_f1_max=$f
  res_max=$res
fi
done

echo $f_f1_max
echo $res_max
#cat $f_f1_max
