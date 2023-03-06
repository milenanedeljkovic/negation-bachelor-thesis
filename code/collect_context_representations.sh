first = $1
second = $2

for i in {first..second..10000}
do
  python3 collect_context_representations.py i i + 1
done
