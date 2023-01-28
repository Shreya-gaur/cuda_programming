for vectorSize in {0..29..1}
do  
    echo "Running for $vectorSize"
    newvsize=$((2**vectorSize))
    sed -i "s/.*vectorSize = .*/	vectorSize = $newvsize;/" src/cudaLib.cu
    make
    nvprof --log-file ./output1/saxpy/outputs/nvprof_${newvsize}.txt ./lab1 <<< 2 >> ./output1/saxpy/outputs/output_${newvsize}.txt
    cat ./output1/saxpy/outputs/output_${newvsize}.txt | grep "Found " >> output1/saxpy/error.txt
    cat ./output1/saxpy/outputs/nvprof_${newvsize}.txt | grep "CUDA memcpy HtoD" >> output1/saxpy/htod.txt
    cat ./output1/saxpy/outputs/nvprof_${newvsize}.txt | grep "CUDA memcpy DtoH" >> output1/saxpy/dtoh.txt
    cat ./output1/saxpy/outputs/nvprof_${newvsize}.txt | grep "saxpy_gpu" >> output1/saxpy/kernel.txt
done