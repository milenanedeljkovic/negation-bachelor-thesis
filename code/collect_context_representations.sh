for i in {10000..100001..10000}
do
  j=$i+1
  python3 "collect_context_representations.py" "$i" "$j"
done
