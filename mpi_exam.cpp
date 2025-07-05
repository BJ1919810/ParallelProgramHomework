#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <climits>
#include <unistd.h>

#define MAX_KEYLEN 64
#define INIT_HASH_CAP 1500003
#define MAX_LOAD_FACTOR 0.7
int g_key_len = 0;

struct Entry {
    char key[MAX_KEYLEN + 1];
    int count;
};

// 简化哈希表结构
struct FlatMap {
    size_t cap;
    size_t size;
    Entry* entries;
};

// 优化的哈希函数
inline unsigned long hash_str(const char* s, int key_len) {
    unsigned long h = 5381;
    for (int i = 0; i < key_len; i++) {
        h = ((h << 5) + h) + (unsigned char)s[i];
    }
    return h;
}

// 创建哈希表
FlatMap* createFlatMap(size_t cap) {
    FlatMap* m = (FlatMap*)malloc(sizeof(FlatMap));
    if (!m) return NULL;
    m->cap = cap;
    m->size = 0;
    m->entries = (Entry*)calloc(cap, sizeof(Entry));
    if (!m->entries) {
        free(m);
        return NULL;
    }
    // 初始化所有键为空
    for (size_t i = 0; i < cap; i++) {
        m->entries[i].key[0] = '\0';
    }
    return m;
}

// 释放哈希表
void freeFlatMap(FlatMap* m) {
    if (m) {
        if (m->entries) free(m->entries);
        free(m);
    }
}

// 哈希表扩容
void flatMapExpand(FlatMap* m, int key_len);

// 哈希表插入（内存优化版）
void flatMapAdd(FlatMap* m, const char* key, int cnt, int key_len) {
    if (m->size >= m->cap * MAX_LOAD_FACTOR) {
        flatMapExpand(m, key_len);
    }

    unsigned long h = hash_str(key, key_len) % m->cap;
    unsigned long start_h = h;
    int step = 1;
    
    while (true) {
        Entry* e = &m->entries[h];
        
        // 空槽或相同键
        if (e->key[0] == '\0' || memcmp(e->key, key, key_len) == 0) {
            if (e->key[0] == '\0') {
                memcpy(e->key, key, key_len);
                e->key[key_len] = '\0';
                e->count = 0;
                m->size++;
            }
            e->count += cnt;
            return;
        }
        
        // 二次探测
        h = (start_h + step * step) % m->cap;
        step++;
        if (step > 100) { // 避免无限循环
            flatMapExpand(m, key_len);
            h = hash_str(key, key_len) % m->cap;
            start_h = h;
            step = 1;
        }
    }
}

// 哈希表扩容实现（内存优化）
void flatMapExpand(FlatMap* m, int key_len) {
    size_t new_cap = m->cap * 2;
    Entry* new_entries = (Entry*)calloc(new_cap, sizeof(Entry));
    if (!new_entries) return;

    // 初始化新条目
    for (size_t i = 0; i < new_cap; i++) {
        new_entries[i].key[0] = '\0';
    }
    
    // 创建临时哈希表
    FlatMap tmp_map;
    tmp_map.cap = new_cap;
    tmp_map.size = 0;
    tmp_map.entries = new_entries;
    
    // 重新插入所有条目
    for (size_t i = 0; i < m->cap; i++) {
        if (m->entries[i].key[0] != '\0') {
            flatMapAdd(&tmp_map, m->entries[i].key, m->entries[i].count, key_len);
        }
    }

    // 更新原始哈希表
    free(m->entries);
    m->entries = tmp_map.entries;
    m->cap = tmp_map.cap;
    m->size = tmp_map.size;
}

// 哈希表转数组
Entry* flatMapToArray(FlatMap* m, size_t* out_size) {
    *out_size = m->size;
    Entry* arr = (Entry*)malloc(m->size * sizeof(Entry));
    size_t idx = 0;
    for (size_t i = 0; i < m->cap; i++) {
        if (m->entries[i].key[0] != '\0') {
            memcpy(&arr[idx], &m->entries[i], sizeof(Entry));
            idx++;
        }
    }
    return arr;
}

// 交换函数
void swap(Entry* a, Entry* b) {
    Entry t = *a;
    *a = *b;
    *b = t;
}

// 比较函数 - 用于归并排序
int cmpKey(const Entry* a, const Entry* b) {
    return memcmp(a->key, b->key, g_key_len);
}

// 快速排序按键 - 用于归并排序
void quickSortByKey(Entry* arr, int left, int right) {
    if (left >= right) return;
    
    int i = left, j = right;
    Entry pivot = arr[(left + right) / 2];
    
    while (i <= j) {
        while (cmpKey(&arr[i], &pivot) < 0) i++;
        while (cmpKey(&arr[j], &pivot) > 0) j--;
        if (i <= j) {
            swap(&arr[i], &arr[j]);
            i++;
            j--;
        }
    }
    
    if (left < j) quickSortByKey(arr, left, j);
    if (i < right) quickSortByKey(arr, i, right);
}

// 最终比较函数 - 按计数降序，计数相同时按键升序
int cmpFinal(const void* a, const void* b) {
    const Entry* ea = (const Entry*)a;
    const Entry* eb = (const Entry*)b;
    if (ea->count != eb->count) return eb->count - ea->count; // 降序
    return memcmp(ea->key, eb->key, g_key_len); // 升序
}

// 最终排序函数
void quickSortFinal(Entry* arr, int left, int right) {
    if (left >= right) return;
    
    int i = left, j = right;
    Entry pivot = arr[(left + right) / 2];
    
    while (i <= j) {
        while (cmpFinal(&arr[i], &pivot) < 0) i++;
        while (cmpFinal(&arr[j], &pivot) > 0) j--;
        if (i <= j) {
            swap(&arr[i], &arr[j]);
            i++;
            j--;
        }
    }
    
    if (left < j) quickSortFinal(arr, left, j);
    if (i < right) quickSortFinal(arr, i, right);
}

// 解析文件名
bool parseDatasetName(const char *fname, int &key_len, char *scale_out) {
    return (sscanf(fname, "data_%d_%s.txt", &key_len, scale_out) == 2);
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (rank == 0) printf("MPI processes: %d\n", nprocs);

    // 获取文件列表
    char** filenames = NULL;
    int fileCount = 0;
    
    if (rank == 0) {
        DIR *d = opendir("dataset");
        if (!d) {
            perror("opendir");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        struct dirent *ent;
        while ((ent = readdir(d)) != NULL) {
            if (ent->d_type == DT_REG) fileCount++;
        }
        rewinddir(d);
        
        filenames = (char**)malloc(fileCount * sizeof(char*));
        int idx = 0;
        while ((ent = readdir(d)) != NULL) {
            if (ent->d_type == DT_REG) {
                filenames[idx] = strdup(ent->d_name);
                idx++;
            }
        }
        closedir(d);
    }

    // 广播文件列表
    MPI_Bcast(&fileCount, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0) filenames = (char**)malloc(fileCount * sizeof(char*));
    
    for (int i = 0; i < fileCount; i++) {
        int name_len = 0;
        if (rank == 0) name_len = strlen(filenames[i]) + 1;
        MPI_Bcast(&name_len, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (rank != 0) filenames[i] = (char*)malloc(name_len);
        MPI_Bcast(filenames[i], name_len, MPI_CHAR, 0, MPI_COMM_WORLD);
    }

    // 处理每个文件
    for (int file_idx = 0; file_idx < fileCount; file_idx++) {
        const char* fname = filenames[file_idx];
        int key_len;
        char scale[16];
        if (!parseDatasetName(fname, key_len, scale)) continue;
        g_key_len = key_len;

        // 移除.txt扩展名
        char* dot = strrchr(scale, '.');
        if (dot && strcmp(dot, ".txt") == 0) *dot = '\0';

        char filepath[PATH_MAX];
        snprintf(filepath, sizeof(filepath), "dataset/%s", fname);
        if (rank == 0) printf("\nProcessing %s (key_len=%d scale=%s)\n", filepath, key_len, scale);

        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();

        // 使用MPI-IO并行读取
        MPI_File fh;
        int rc = MPI_File_open(MPI_COMM_WORLD, filepath, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
        if (rc != MPI_SUCCESS) {
            if (rank == 0) {
                char err_str[MPI_MAX_ERROR_STRING];
                int err_len;
                MPI_Error_string(rc, err_str, &err_len);
                fprintf(stderr, "MPI_File_open failed: %s\n", err_str);
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // 获取文件大小
        MPI_Offset file_size;
        MPI_File_get_size(fh, &file_size);
        
        // 计算每个进程的读取范围
        MPI_Offset chunk_size = file_size / nprocs;
        MPI_Offset remainder = file_size % nprocs;
        MPI_Offset start = rank * chunk_size + (rank < remainder ? rank : remainder);
        MPI_Offset end = start + chunk_size - 1;
        if (rank < remainder) end += 1;
        
        // 确保从行首开始读取
        if (rank != 0 && start > 0) {
            char prev_char;
            MPI_File_read_at(fh, start - 1, &prev_char, 1, MPI_CHAR, MPI_STATUS_IGNORE);
            if (prev_char != '\n') {
                MPI_Offset pos = start - 1;
                while (pos >= 0) {
                    MPI_File_read_at(fh, pos, &prev_char, 1, MPI_CHAR, MPI_STATUS_IGNORE);
                    if (prev_char == '\n') {
                        start = pos + 1;
                        break;
                    }
                    if (pos == 0) break;
                    pos--;
                }
            }
        }

        // 处理最后一个进程
        if (rank == nprocs - 1) end = file_size - 1;

        // 计算读取大小
        MPI_Offset read_size = end - start + 1;
        if (read_size <= 0) read_size = 0;
        
        // 读取数据
        char* local_buf = NULL;
        if (read_size > 0) {
            local_buf = (char*)malloc(read_size + 1);
            if (local_buf) {
                MPI_File_read_at(fh, start, local_buf, read_size, MPI_CHAR, MPI_STATUS_IGNORE);
                local_buf[read_size] = '\0';
            }
        }
        MPI_File_close(&fh);

        // 根据数据集规模动态调整初始容量
        size_t init_cap = INIT_HASH_CAP;
        if (key_len <= 16) {
            if (strstr(scale, "40M")) {
                // 40M数据集 - 根据进程数调整容量
                init_cap = 10000000 + (20000000 / nprocs); // 10M + 20M/进程数
            } else if (strstr(scale, "10M")) {
                init_cap = 5000000;  // 5M
            } else if (strstr(scale, "1M")) {
                init_cap = 1000000;  // 1M
            }
        }
        
        if (rank == 0) printf("  Initial hash capacity: %zu\n", init_cap);
        
        // 创建哈希表并统计
        FlatMap* map = createFlatMap(init_cap);
        if (!map) {
            fprintf(stderr, "[%d] createFlatMap failed\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        if (local_buf) {
            char* ptr = local_buf;
            while (*ptr) {
                char* end_ptr = strchr(ptr, '\n');
                if (!end_ptr) break;
                
                if (end_ptr - ptr == key_len) {
                    char temp = *end_ptr;
                    *end_ptr = '\0';
                    flatMapAdd(map, ptr, 1, key_len);
                    *end_ptr = temp;
                }
                ptr = end_ptr + 1;
            }
            free(local_buf);
        }

        // 转换为数组并排序
        size_t local_size = 0;
        Entry* local_arr = NULL;
        if (map) {
            local_arr = flatMapToArray(map, &local_size);
            freeFlatMap(map);
        }
        
        if (local_size > 0) {
            if (rank == 0) printf("  Local keys: %zu\n", local_size);
            quickSortByKey(local_arr, 0, local_size - 1);
        }
        
        // 树形归并
        int step = 1;
        while (step < nprocs) {
            if (rank % (2 * step) == 0) {
                int src_rank = rank + step;
                if (src_rank < nprocs) {
                    int src_size;
                    MPI_Recv(&src_size, 1, MPI_INT, src_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    
                    Entry* src_arr = NULL;
                    if (src_size > 0) {
                        src_arr = (Entry*)malloc(src_size * sizeof(Entry));
                        if (src_arr) {
                            MPI_Recv(src_arr, src_size * sizeof(Entry), MPI_BYTE, 
                                    src_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        }
                    }
                    
                    // 合并数组
                    if (local_size > 0 || src_size > 0) {
                        // 创建合并数组
                        size_t merged_size = local_size + src_size;
                        Entry* merged_arr = (Entry*)malloc(merged_size * sizeof(Entry));
                        
                        size_t i = 0, j = 0, k = 0;
                        while (i < local_size && j < (size_t)src_size) {
                            if (cmpKey(&local_arr[i], &src_arr[j]) <= 0) {
                                merged_arr[k++] = local_arr[i++];
                            } else {
                                merged_arr[k++] = src_arr[j++];
                            }
                        }
                        while (i < local_size) merged_arr[k++] = local_arr[i++];
                        while (j < (size_t)src_size) merged_arr[k++] = src_arr[j++];
                        
                        if (local_arr) free(local_arr);
                        if (src_arr) free(src_arr);
                        
                        local_arr = merged_arr;
                        local_size = merged_size;
                    }
                }
            } else {
                int dst_rank = rank - step;
                int send_size = (int)local_size;
                MPI_Send(&send_size, 1, MPI_INT, dst_rank, 0, MPI_COMM_WORLD);
                if (send_size > 0 && local_arr) {
                    MPI_Send(local_arr, send_size * sizeof(Entry), MPI_BYTE, dst_rank, 0, MPI_COMM_WORLD);
                }
                if (local_arr) free(local_arr);
                local_arr = NULL;
                local_size = 0;
                break;
            }
            step *= 2;
        }

        // 最终处理（只在rank 0）
        if (rank == 0 && local_size > 0) {
            // 合并相同键
            size_t unique_size = 0;
            Entry* unique_arr = (Entry*)malloc(local_size * sizeof(Entry));
            if (local_size > 0) {
                memcpy(&unique_arr[unique_size], &local_arr[0], sizeof(Entry));
                unique_size++;
                
                for (size_t i = 1; i < local_size; i++) {
                    if (memcmp(unique_arr[unique_size-1].key, local_arr[i].key, key_len) == 0) {
                        unique_arr[unique_size-1].count += local_arr[i].count;
                    } else {
                        memcpy(&unique_arr[unique_size], &local_arr[i], sizeof(Entry));
                        unique_size++;
                    }
                }
            }
            
            // 关键修复：添加最终排序
            quickSortFinal(unique_arr, 0, unique_size - 1);
            
            // 写入结果文件
            char outpath[PATH_MAX];
            snprintf(outpath, sizeof(outpath), "output/result%d-%s.txt", key_len, scale);
            FILE *fo = fopen(outpath, "w");
            if (!fo) {
                perror("fopen output");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            fprintf(fo, "%zu\n", unique_size);
            for (size_t i = 0; i < unique_size; i++) {
                fprintf(fo, "%.*s %d\n", key_len, unique_arr[i].key, unique_arr[i].count);
            }
            fclose(fo);

            double elapsed = MPI_Wtime() - t0;
            printf("Done %s in %.3f seconds, unique keys = %zu\n", fname, elapsed, unique_size);
            free(unique_arr);
        }
        
        if (local_arr) free(local_arr);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // 清理内存
    for (int i = 0; i < fileCount; i++) {
        if (filenames[i]) free(filenames[i]);
    }
    if (filenames) free(filenames);
    
    if (rank == 0) printf("\nAll datasets processed.\n");
    MPI_Finalize();
    return 0;
}
