input="/home/codeis123/examples.csv"
{
	read
	i=1
	j=0
	while IFS=',' read -r str		
	do
	  echo $str | tee -a by_cache_bu_3.csv
	  for j in $(seq 65 1 80)
	  do
	  	./cache /home/codeis123/sp_matrix_examples/$str.cbd $j | tee -a by_cache_bu_3.csv
	  done
	  i='exp $i + 1'
	done
} < "$input"
