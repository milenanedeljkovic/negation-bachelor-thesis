for i in {10000..100001..10000}
do
  python3 "collect_context_representations.py" "$i" "$((i + 1))"
done
