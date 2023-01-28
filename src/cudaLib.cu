
#include "cudaLib.cuh"

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__global__ 
void saxpy_gpu (float* x, float* y, float scale, int size) {
	//	Insert GPU SAXPY kernel code here

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid < size)
		y[tid] = scale*x[tid] + y[tid];

}

int runGpuSaxpy(int vectorSize) {

	std::cout << "Hello GPU Saxpy!\n";
	 
	printf("\nvectorSize:%i \n", (int)vectorSize);

	//	Insert code here

	float * a, * b, * c;
	float *dev_a, *dev_b;

	a = (float *) malloc(vectorSize * sizeof(float));
	b = (float *) malloc(vectorSize * sizeof(float));
	c = (float *) malloc(vectorSize * sizeof(float));

	cudaMalloc(&dev_a, vectorSize * sizeof(float));
	cudaMalloc(&dev_b, vectorSize * sizeof(float));

	if (a == NULL || b == NULL) {
		printf("Unable to malloc memory ... Exiting!");
		return -1;
	}

	vectorInit(a, vectorSize);
	vectorInit(b, vectorSize);

	float scale = 2.0f;

	#ifndef DEBUG_PRINT_DISABLE 
		printf("\n Adding vectors : \n");
		printf(" scale = %f\n", scale);
		printf(" a = { ");
		for (int i = 0; i < 5; ++i) {
			printf("%3.4f, ", a[i]);
		}
		printf(" ... }\n");
		printf(" b = { ");
		for (int i = 0; i < 5; ++i) {
			printf("%3.4f, ", b[i]);
		}
		printf(" ... }\n");
	#endif

	cudaMemcpy(dev_a, a, vectorSize * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, vectorSize * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid =
            (vectorSize + threadsPerBlock - 1) / threadsPerBlock;
    
	saxpy_gpu<<<blocksPerGrid, threadsPerBlock>>>(dev_a, dev_b, scale, vectorSize);

	cudaMemcpy(c, dev_b, vectorSize * sizeof(float), cudaMemcpyDeviceToHost);

	#ifndef DEBUG_PRINT_DISABLE 
		printf(" c = { ");
		for (int i = 0; i < 5; ++i) {
			printf("%3.4f, ", c[i]);
		}
		printf(" ... }\n");
	#endif

	int errorCount = verifyVector(a, b, c, scale, vectorSize);
	std::cout << "Found " << errorCount << " / " << vectorSize << " errors \n";
	
	cudaFree(dev_a);
	cudaFree(dev_b);

	delete[] a;
	delete[] b;
	delete[] c;

	return 0;
}

__global__
void generatePoints (uint64_t * pSums, uint64_t pSumSize, uint64_t sampleSize) {
	//	Insert code here

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;

	curandState states;

	curand_init(1234, index, 0, &states);

	for (int iter = index; iter < pSumSize; iter+=stride) {
		//	Main CPU Monte-Carlo Code
		int hitCount = 0;
		for (uint64_t idx = 0; idx < sampleSize; ++idx) {

			float x = curand_uniform (&states);
			float y = curand_uniform (&states);

			if ( int(x * x + y * y) == 0 ) {
				++hitCount;
			}

		}
		pSums[iter] = hitCount; 
	}
}

__global__ 
void reduceCounts (uint64_t * pSums, uint64_t * totals, uint64_t pSumSize, uint64_t reduceSize) {
	//	Insert code here

	int index = threadIdx.x;
	int stride = blockDim.x;

	uint64_t sum = 0;
	
	for(int i=index; i < pSumSize; i+=stride){
		sum += pSums[i];
	}

	totals[index] = sum;
}


int runGpuMCPi (uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {

	//  Check CUDA device presence
	int numDev;
	cudaGetDeviceCount(&numDev);
	if (numDev < 1) {
		std::cout << "CUDA device missing!\n";
		return -1;
	}

	auto tStart = std::chrono::high_resolution_clock::now();
	
	float approxPi = estimatePi(generateThreadCount, sampleSize, 
		reduceThreadCount, reduceSize);
	
	std::cout << "\nEstimated Pi = " << approxPi << "\n";

	auto tEnd= std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> time_span = (tEnd- tStart);
	std::cout << "It took " << time_span.count() << " seconds.";

	return 0;
}

double estimatePi(uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {
	
	size_t memsize = generateThreadCount * sizeof(uint64_t);

	uint64_t totalHitCount = 0;
	uint64_t* dev_pSums;
	uint64_t* dev_totals;
	uint64_t* partial_totals;
	// uint64_t* pSums;

	double approxPi = 0;

	partial_totals = (uint64_t*) malloc(reduceThreadCount * sizeof(uint64_t));
	// pSums = (uint64_t*) malloc(memsize);

	cudaMalloc(&dev_pSums, memsize);
	cudaMalloc(&dev_totals, reduceThreadCount*sizeof(uint64_t));

	int threadsPerBlock = 1024;
	int blocksPerGrid = ceil(generateThreadCount/threadsPerBlock);
	if (blocksPerGrid == 0){
		generatePoints<<<1, generateThreadCount>>>(dev_pSums, generateThreadCount, sampleSize);
	}
	else{
		dim3 DimBlock(threadsPerBlock,1,1);
		dim3 DimGrid(blocksPerGrid,1,1);
		generatePoints<<<DimGrid, DimBlock>>>(dev_pSums, generateThreadCount, sampleSize);
	}

	reduceCounts<<<1, reduceThreadCount>>>(dev_pSums, dev_totals, generateThreadCount, reduceSize);
	
	cudaMemcpy(partial_totals, dev_totals, reduceThreadCount*sizeof(uint64_t), cudaMemcpyDeviceToHost);
	
	for(int i=0; i<reduceThreadCount; i++)
		totalHitCount += partial_totals[i];

	// for(int i=0; i<generateThreadCount; i++)
	// 	totalHitCount_test += pSums[i];
 
	// #ifndef DEBUG_PRINT_DISABLE 
	// if (totalHitCount==totalHitCount_test){
	// 	printf("\nCorrect totalHitCount:%i \n", (int)totalHitCount);
	// }
	// else{
	// 	printf("\nCorrect totalHitCount_test:%i \n", (int)totalHitCount_test);
	// 	printf("\nIncorrect totalHitCount:%i \n", (int)totalHitCount);

	// 	approxPi = ((double) totalHitCount_test / sampleSize) / generateThreadCount;
	// }
	// #endif

	approxPi = ((double) totalHitCount / sampleSize) / generateThreadCount;
	
	cudaFree(dev_pSums);
	cudaFree(dev_totals);
	delete[] partial_totals;

	approxPi = approxPi * 4.0f;

	return approxPi;
}
