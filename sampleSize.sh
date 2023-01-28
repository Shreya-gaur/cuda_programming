for sampleSize in {0..29..1}
do  
    echo "Running for $sampleSize"
    newvsize=$((2**sampleSize))
    sed -i "s/.*sampleSize = .*/	sampleSize = $newvsize;/" src/cudaLib.cu
    make
    nvprof --log-file ./output1/sampleSize/outputs/nvprof_${newvsize}.txt ./lab1 <<< 4 >> ./output1/sampleSize/outputs/output_${newvsize}.txt
    cat ./output1/sampleSize/outputs/output_${newvsize}.txt | grep "It took" >> output1/sampleSize/time_taken.txt
    cat ./output1/sampleSize/outputs/nvprof_${newvsize}.txt | grep "CUDA memcpy DtoH" >> output1/sampleSize/dtoh.txt
    cat ./output1/sampleSize/outputs/nvprof_${newvsize}.txt | grep "generatePoints" >> output1/sampleSize/generatePoints.txt
    cat ./output1/sampleSize/outputs/nvprof_${newvsize}.txt | grep "reduceCounts" >> output1/sampleSize/reduceCounts.txt
done