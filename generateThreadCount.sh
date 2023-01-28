# for vectorSize in 10
for generateThreadCount in {0..29..1}
do  
    echo "Running for $generateThreadCount"
    newvsize=$((2**generateThreadCount))
    sed -i "s/.*generateThreadCount = .*/	generateThreadCount = $newvsize;/" src/cudaLib.cu
    make
    nvprof --log-file ./output1/mcpi_trial2/outputs/nvprof_${newvsize}.txt ./lab1 <<< 4 >> ./output1/mcpi_trial2/outputs/output_${newvsize}.txt
    # cat ./output1/mcpi_trial2/outputs/nvprof_${newvsize}.txt | grep "CUDA memcpy HtoD" >> output1/mcpi_trial2/htod.txt
    cat ./output1/mcpi_trial2/outputs/output_${newvsize}.txt | grep "It took" >> output1/mcpi_trial2/time_taken.txt
    cat ./output1/mcpi_trial2/outputs/nvprof_${newvsize}.txt | grep "CUDA memcpy DtoH" >> output1/mcpi_trial2/dtoh.txt
    cat ./output1/mcpi_trial2/outputs/nvprof_${newvsize}.txt | grep "generatePoints" >> output1/mcpi_trial2/generatePoints.txt
    cat ./output1/mcpi_trial2/outputs/nvprof_${newvsize}.txt | grep "reduceCounts" >> output1/mcpi_trial2/reduceCounts.txt
    # cat ./output1/mcpi_trial2/outputs/nvprof_${newvsize}.txt | grep "estimatePi" >> output1/mcpi_trial2/estimatePi.txt
done