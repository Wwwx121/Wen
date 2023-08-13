./wait
export GOMP_CPU_AFFINITY="0-15"
input="/home/codeis123/examples.csv"
{
	read
	i=1
	j=0
	while IFS=',' read -r str		
	do
	  echo $str | tee -a test.csv
	  for j in $(seq 50 50 1000)
	  do
	  	./avx2_nnz /home/codeis123/sp_matrix_examples/$str.cbd 100 $j | tee -a test.csv
	  done
	  i='exp $i + 1'
	done
} < "$input"
