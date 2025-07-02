__global__ void GenerateSingleSegmentKernel_FlatOutput(char **values, char *output_flat, int count, int total_len_per_guess) ;
__global__ void GenerateLastSegmentKernel_FlatOutput(char *prefix, char **values, char *output_flat, int count, int prefix_len, int total_len_per_guess) ;
