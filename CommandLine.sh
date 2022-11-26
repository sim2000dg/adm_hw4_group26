#! /bin/zsh
echo 'This are the 10 location with the maximum number of purchases been made: '
awk -F ',' '{print $5}' new_bank.csv | sort | uniq -c | sort -nr | head -n 10

echo "Average number of purchase for the females:"
awk -F ',' '$4=="F"{sum+=$9;cnt++}END{print sum/cnt}' new_bank.csv

echo "Average number of purchase for the males:"
awk -F ',' '$4=="M"{sum+=$9;cnt++}END{print sum/cnt}' new_bank.csv

echo "Customer with the highest average transaction amount in the dataset:"
awk -F ',' '{sum[$2] += $9; cnt[$2]++}; END{ for (id in sum) { print sum[id]/cnt[id],id } }' new_bank.csv | sort -nr | head -n 1
