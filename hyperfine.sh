#!/bin/sh
# NOTE: initially, takuzu.py should have goal = xxx(takuzu), not any particular search

echo "# Execution Times (ms)" > times.md
echo "" >> times.md
searches=( 'depth_first_tree_search' 'breadth_first_tree_search' 'greedy_search' 'astar_search' )

for search in "${searches[@]}"; do
  echo "## $search" >> times.md
  sed -i 's/goal = xxx/goal = '$search'/g' takuzu.py
  for i in {01..13}; do
    echo "Running $search on $i"
    hyperfine --warmup 25 -m 250 "python takuzu.py < testes-takuzu/input_T$i" --export-csv /tmp/takuzu.csv
    # get the mean from takuzu.csv - it's the second column, second row
    mean=$(cat /tmp/takuzu.csv | cut -d',' -f2 | head -n2 | tail -n1)
    # the mean is in seconds: convert to ms
    mean=$(echo "scale=3; $mean * 1000" | bc)
    # get only the first 3 decimal digits
    mean=$(echo "scale=3; $mean / 1" | bc)
    echo "- Time: $search on $i: $mean ms" >> times.md
  done
  echo "" >> times.md
  sed -i 's/goal = '$search'/goal = xxx/g' takuzu.py
done