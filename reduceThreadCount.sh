for reduceThreadCount in {0..29..1}
do  
    echo "Running for $reduceThreadCount"
    newvsize=$((2**reduceThreadCount))
    sed -i "s/.*reduceThreadCount = .*/	reduceThreadCount = $newvsize;/" src/cudaLib.cu
    make
    nvprof --log-file ./output1/reduceThreadCount/outputs/nvprof_${newvsize}.txt ./lab1 <<< 4 >> ./output1/reduceThreadCount/outputs/output_${newvsize}.txt
    cat ./output1/reduceThreadCount/outputs/output_${newvsize}.txt | grep "It took" >> output1/reduceThreadCount/time_taken.txt
    cat ./output1/reduceThreadCount/outputs/nvprof_${newvsize}.txt | grep "CUDA memcpy DtoH" >> output1/reduceThreadCount/dtoh.txt
    cat ./output1/reduceThreadCount/outputs/nvprof_${newvsize}.txt | grep "generatePoints" >> output1/reduceThreadCount/generatePoints.txt
    cat ./output1/reduceThreadCount/outputs/nvprof_${newvsize}.txt | grep "reduceCounts" >> output1/reduceThreadCount/reduceCounts.txt
done