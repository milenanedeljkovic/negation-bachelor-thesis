for i in {30000..30001..10000}
do
  python3 "collect_context_representations.py" "$i" "$((i + 1))"
done

for i in {340000..340001..10000}
do
  python3 "collect_context_representations.py" "$i" "$((i + 1))"
done


for i in {400000..490001..10000}
do
  python3 "collect_context_representations.py" "$i" "$((i + 1))"
done


for i in {640000..640001..10000}
do
  python3 "collect_context_representations.py" "$i" "$((i + 1))"
done

for i in {800000..800001..10000}
do
  python3 "collect_context_representations.py" "$i" "$((i + 1))"
done

for i in {890000..890001..10000}
do
  python3 "collect_context_representations.py" "$i" "$((i + 1))"
done

for i in {970000..990001..10000}
do
  python3 "collect_context_representations.py" "$i" "$((i + 1))"
done

