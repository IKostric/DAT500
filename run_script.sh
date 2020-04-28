# #!/bin/bash

# # SEQUENTIAL

# #model_type='sequential', 
# #mutation_rate=0.01, 
# #elite_fraction=0.1, 
# #num_iterations=1, 
# #num_locations=10, 
# #population_size=100

# for l in 1000
# do
# for p in 1000
# do
# for n in 1000
# do
# 	python Driver.py -t sequential \
#         -n $n -p $p -l $l \
#         --elite-fraction 0.1 \
#         --mutation-rate 0.01
# done
# done
# done

# # SPARK - GLOBAL

# #model_type='sequential', 
# #mutation_rate=0.01, 
# #elite_fraction=0.1, 
# #num_iterations=1, 
# #num_locations=10, 
# #population_size=100

# for l in 1000
# do
# for p in 1000
# do
# for n in 1000
# do
# 	python Driver.py -t global-s -r spark --spark-master yarn \
#         -n $n -p $p -l $l \
#         --elite-fraction 0.1 \
#         --mutation-rate 0.01
# done
# done
# done

# # SPARK - ISLAND

# #model_type='sequential', 
# #mutation_rate=0.01, 
# #elite_fraction=0.1, 
# #num_iterations=1, 
# #num_locations=10, 
# #population_size=100
# #--num-islands 4
# #--num-migrations 2
# #--migrant-fraction 0.3

# for l in 1000
# do
# for p in 1000
# do
# for n in 10 100 1000
# do
# 	python Driver.py -t island-s -r spark --spark-master yarn \
#         -n $n -p $p -l $l \
#         --elite-fraction 0.1 \
#         --mutation-rate 0.01 \
#         --num-islands 4 \
#         --num-migrations 4 \
#         --migrant-fraction 0.3
# done
# done
# done

# # MAPREDUCE - ISLAND

# #model_type='sequential', 
# #mutation_rate=0.01, 
# #elite_fraction=0.1, 
# #num_iterations=1, 
# #num_locations=10, 
# #population_size=100

# for l in 1000
# do
# for p in 1000
# do
# for n in 2 10 50
# do
# 	python Driver.py -t island -r hadoop \
#         -n $n -p $p -l $l \
#         --elite-fraction 0.1 \
#         --mutation-rate 0.01 \
#         --num-islands 4 \
#         --num-migrations 4 \
#         --migrant-fraction 0.3
# done
# done
# done

# # # MAPREDUCE - GLOBAL

# # #model_type='sequential', 
# # #mutation_rate=0.01, 
# # #elite_fraction=0.1, 
# # #num_iterations=1, 
# # #num_locations=10, 
# # #population_size=100

# for l in 1000
# do
# for p in 1000
# do
# for n in 1 10
# do
# 	python Driver.py -t global -r hadoop \
#         -n $n -p $p -l $l \
#         --elite-fraction 0.1 \
#         --mutation-rate 0.01 
# done
# done
# done

# echo "NUMBER OF ISLANDS" >> results.txt

# array=(2 3 4 5 8)
# array2=(1000 666 500 400 250)

# for index in ${!array[*]}; do
# python Driver.py -t island-s -r spark --spark-master yarn \
#     -n 200 -p ${array2[$index]} -l 1000 \
#     --elite-fraction 0.1 \
#     --mutation-rate 0.01 \
#     --num-islands ${array[$index]} \
#     --num-migrations 4 \
#     --migrant-fraction 0.3
# done

echo "INCREASE IN VOLUME" >> results.txt

for i in 10000
do
    python Driver.py -t sequential \
        -n $i -p $i -l $i \
        --elite-fraction 0.1 \
        --mutation-rate 0.01

    python Driver.py -t global-s -r spark --spark-master yarn \
        -n $i -p $i -l $i \
        --elite-fraction 0.1 \
        --mutation-rate 0.01 
done

array3=(10 100 1000 5000)
array4=(25 250 2500 12500)
array5=(100 1000 10000 50000)

for index in ${!array[*]}; do
    python Driver.py -t island-s -r spark --spark-master yarn \
        -n ${array3[$index]} -p ${array4[$index]} -l ${array3[$index]} \
        --elite-fraction 0.1 \
        --mutation-rate 0.01 \
        --num-islands 4 \
        --num-migrations 9 \
        --migrant-fraction 0.4
done

for i in 50000
do
    python Driver.py -t sequential \
        -n $i -p $i -l $i \
        --elite-fraction 0.1 \
        --mutation-rate 0.01

    python Driver.py -t global-s -r spark --spark-master yarn \
        -n $i -p $i -l $i \
        --elite-fraction 0.1 \
        --mutation-rate 0.01 
done