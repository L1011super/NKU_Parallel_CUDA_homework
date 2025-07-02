#include "PCFG.h"
#include <chrono>
#include <fstream>
#include "md5.h"
#include <iomanip>
#include <unordered_set>
#include <pthread.h>
#include <mutex>
#include <cuda_runtime.h> // 包含 CUDA 运行时头文件
using namespace std;
using namespace chrono;

// 编译指令如下
// nvcc main.cpp train.cpp guessing.cpp md5.cu -o main -O2 -lcudart
//nvcc main.cu train.cpp guessing.cu md5.cu -o main -O1 -lcudart

int main()
{
    double time_hash = 0;  // 用于MD5哈希的时间
    double time_guess = 0; // 哈希和猜测的总时长
    double time_train = 0; // 模型训练的总时长
    PriorityQueue q;
    auto start_train = system_clock::now();
    q.m.train("Rockyou-singleLined-full.txt");
    q.m.order();
    auto end_train = system_clock::now();
    auto duration_train = duration_cast<microseconds>(end_train - start_train);
    time_train = double(duration_train.count()) * microseconds::period::num / microseconds::period::den;


    
    // 加载一些测试数据
    unordered_set<std::string> test_set;
    ifstream test_data("Rockyou-singleLined-full.txt");
    int test_count=0;
    string pw;
    while(test_data>>pw)
    {   
        test_count+=1;
        test_set.insert(pw);
        if (test_count>=1000000)
        {
            break;
        }
    }
    int cracked=0;

    q.init();
    cout << "here" << endl;
    int curr_num = 0;
    auto start = system_clock::now();
    // 由于需要定期清空内存，我们在这里记录已生成的猜测总数
    int history = 0;
    // std::ofstream a("./files/results.txt");
    while (!q.priority.empty())
    {
        q.PopNext();
        q.total_guesses = q.guesses.size();
        if (q.total_guesses - curr_num >= 100000)
        {
            cout << "Guesses generated: " <<history + q.total_guesses << endl;
            curr_num = q.total_guesses;

            // 在此处更改实验生成的猜测上限
            int generate_n=10000000;
            if (history + q.total_guesses > 10000000)
            {
                auto end = system_clock::now();
                auto duration = duration_cast<microseconds>(end - start);
                time_guess = double(duration.count()) * microseconds::period::num / microseconds::period::den;
                cout << "Guess time:" << time_guess - time_hash << "seconds"<< endl;
                cout << "Hash time:" << time_hash << "seconds"<<endl;
                cout << "Train time:" << time_train <<"seconds"<<endl;
                cout << "Cracked: " << dec << cracked << endl;  // 确保以十进制输出
                break;
            }
        }
        // 为了避免内存超限，我们在q.guesses中口令达到一定数目时，将其中的所有口令取出并且进行哈希
        // 然后，q.guesses将会被清空。为了有效记录已经生成的口令总数，维护一个history变量来进行记录
        // 在循环外部打开文件（假设这段代码在某个循环中）
        static std::ofstream outfile("result.txt", std::ios::app);  // 只打开一次
        if (!outfile.is_open()) {
            std::cerr << "无法打开文件 result.txt" << std::endl;
            // 可以选择继续执行或退出
        }

        // 批量处理口令猜测进行MD5计算
        if (q.guesses.size() >= 1000) // 达到一定数量才进行GPU处理
        {
            auto start_hash = system_clock::now();

            int num_guesses = q.guesses.size();
            // 为GPU准备输入数据
            char* d_inputs;
            int* d_input_lengths;
            bit32* d_results;

            size_t total_input_length = 0;
            for (const string& pw_str : q.guesses) {
                total_input_length += pw_str.length() + 1; // +1 for null terminator
            }

            cudaMalloc(&d_inputs, total_input_length * sizeof(char));
            cudaMalloc(&d_input_lengths, num_guesses * sizeof(int));
            cudaMalloc(&d_results, num_guesses * 4 * sizeof(bit32)); // 4 bit32 per MD5 hash

            vector<int> input_lengths(num_guesses);
            vector<char> inputs_buffer(total_input_length);
            size_t current_offset = 0;

            for (int i = 0; i < num_guesses; ++i) {
                const string& pw_str = q.guesses[i];
                input_lengths[i] = pw_str.length();
                memcpy(inputs_buffer.data() + current_offset, pw_str.c_str(), pw_str.length() + 1);
                current_offset += pw_str.length() + 1;
            }

            cudaMemcpy(d_inputs, inputs_buffer.data(), total_input_length * sizeof(char), cudaMemcpyHostToDevice);
            cudaMemcpy(d_input_lengths, input_lengths.data(), num_guesses * sizeof(int), cudaMemcpyHostToDevice);

            // 设置CUDA核函数的启动配置
            int blockSize = 256;
            int gridSize = (num_guesses + blockSize - 1) / blockSize;

            // 调用CUDA核函数
            MD5Hash_kernel<<<gridSize, blockSize>>>(d_inputs, d_input_lengths, d_results, num_guesses);
            cudaDeviceSynchronize(); // 等待GPU完成计算

            // 将结果从GPU拷贝回CPU
            vector<bit32> h_results(num_guesses * 4);
            cudaMemcpy(h_results.data(), d_results, num_guesses * 4 * sizeof(bit32), cudaMemcpyDeviceToHost);

            // 处理结果并写入文件
            for (int i = 0; i < num_guesses; ++i) {
                if (test_set.count(q.guesses[i])) { // Use count for unordered_set
                    cracked += 1;
                }
                for (int i1 = 0; i1 < 4; i1 += 1) {
                    outfile << std::setw(8) << std::setfill('0') << std::hex << h_results[i * 4 + i1];
                }
                outfile << std::endl;
            }

            // 释放GPU内存
            cudaFree(d_inputs);
            cudaFree(d_input_lengths);
            cudaFree(d_results);

            // 在这里对哈希所需的总时长进行计算
            auto end_hash = system_clock::now();
            auto duration = duration_cast<microseconds>(end_hash - start_hash);
            time_hash += double(duration.count()) * microseconds::period::num / microseconds::period::den;

            // 记录已经生成的口令总数
            history += curr_num;
            curr_num = 0;
            q.guesses.clear();
        }
    }
    
    return 0;
}