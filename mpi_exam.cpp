#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <climits>
#include <cmath>

#define MAX_KEYLEN 64
#define INIT_HASH_CAP 100003   // 初始哈希表容量
#define MAX_LOAD_FACTOR 0.75   // 哈希表最大负载因子

struct Entry {
    char key[MAX_KEYLEN + 1];
    int count;
};

struct Node {
    char key[MAX_KEYLEN + 1];
    int count;
    Node* next;
};

struct HashMap {
    size_t cap;
    size_t size;     // 当前元素数量
    Node** buckets;
};

unsigned long hash_str(const char* s, int key_len) {
    unsigned long h = 5381;
    for (int i = 0; i < key_len && s[i] != '\0'; i++) {
        h = ((h << 5) + h) + s[i];
    }
    return h;
}

HashMap* createHashMap(size_t cap) {
    HashMap* m = (HashMap*)malloc(sizeof(HashMap));
    if (!m) return NULL;
    m->cap = cap;
    m->size = 0;
    m->buckets = (Node**)calloc(cap, sizeof(Node*));
    if (!m->buckets) {
        free(m);
        return NULL;
    }
    return m;
}

void mapExpand(HashMap* m, int key_len) {
    size_t new_cap = m->cap * 2;
    Node** new_buckets = (Node**)calloc(new_cap, sizeof(Node*));
    if (!new_buckets) return;
    
    // 重哈希所有节点
    for (size_t i = 0; i < m->cap; i++) {
        Node* cur = m->buckets[i];
        while (cur) {
            Node* next = cur->next;
            unsigned long h = hash_str(cur->key, key_len) % new_cap;
            cur->next = new_buckets[h];
            new_buckets[h] = cur;
            cur = next;
        }
    }
    
    free(m->buckets);
    m->buckets = new_buckets;
    m->cap = new_cap;
}

void mapAdd(HashMap* m, const char* key, int cnt, int key_len) {
    // 检查是否需要扩展哈希表
    if (m->size >= m->cap * MAX_LOAD_FACTOR) {
        mapExpand(m, key_len);
    }
    
    unsigned long h = hash_str(key, key_len) % m->cap;
    Node* cur = m->buckets[h];
    while (cur) {
        if (memcmp(cur->key, key, key_len) == 0) {
            cur->count += cnt;
            return;
        }
        cur = cur->next;
    }
    
    // 创建新节点
    Node* n = (Node*)malloc(sizeof(Node));
    if (!n) return;
    memcpy(n->key, key, key_len);
    n->key[key_len] = '\0';
    n->count = cnt;
    n->next = m->buckets[h];
    m->buckets[h] = n;
    m->size++;
}

void freeHashMap(HashMap* m) {
    if (!m) return;
    for (size_t i = 0; i < m->cap; i++) {
        Node* cur = m->buckets[i];
        while (cur) {
            Node* next = cur->next;
            free(cur);
            cur = next;
        }
    }
    free(m->buckets);
    free(m);
}

void swapEntry(Entry &a, Entry &b) {
    Entry tmp = a;
    a = b;
    b = tmp;
}

int cmpEntry(const Entry &a, const Entry &b) {
    if (a.count != b.count) return (b.count - a.count);
    return strcmp(a.key, b.key);
}

int cmpKey(const void* a, const void* b) {
    const Entry* ea = (const Entry*)a;
    const Entry* eb = (const Entry*)b;
    return strcmp(ea->key, eb->key);
}

void quickSortForEntry(Entry *arr, int left, int right) {
    if (left >= right) return;
    int i = left, j = right;
    Entry pivot = arr[(left + right) / 2];
    while (i <= j) {
        while (cmpEntry(arr[i], pivot) < 0) i++;
        while (cmpEntry(arr[j], pivot) > 0) j--;
        if (i <= j) {
            swapEntry(arr[i], arr[j]);
            i++; j--;
        }
    }
    if (left < j) quickSortForEntry(arr, left, j);
    if (i < right) quickSortForEntry(arr, i, right);
}

bool parseDatasetName(const char *fname, int &key_len, char *scale_out) {
    return (sscanf(fname, "data_%d_%s.txt", &key_len, scale_out) == 2);
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (rank == 0) printf("MPI processes: %d\n", nprocs);

    DIR *d = opendir("dataset");
    if (!d) {
        if (rank == 0) perror("opendir");
        MPI_Finalize();
        return 1;
    }

    struct dirent *ent;
    while ((ent = readdir(d)) != NULL) {
        int key_len;
        char scale[16];
        if (!parseDatasetName(ent->d_name, key_len, scale)) continue;

        char *dot = strrchr(scale, '.');
        if (dot && strcmp(dot, ".txt") == 0) {
            *dot = '\0';
        }

        char filepath[PATH_MAX];
        snprintf(filepath, sizeof(filepath), "dataset/%s", ent->d_name);
        if (rank == 0) printf("\n=== Processing %s (key_len=%d scale=%s) ===\n",
                              filepath, key_len, scale);

        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();

        // 打开文件
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
        MPI_Offset chunk = file_size / nprocs;
        MPI_Offset my_off = rank * chunk;
        MPI_Offset my_end = (rank == nprocs - 1) ? file_size : (my_off + chunk);

        // 边界处理：确保从行首开始
        if (rank > 0 && my_off > 0) {
            const size_t probe_size = 4096;  // 4KB探测块
            char* probe_buf = (char*)malloc(probe_size);
            if (!probe_buf) {
                fprintf(stderr, "[%d] malloc failed\n", rank);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            
            // 读取探测块查找换行符
            MPI_File_read_at(fh, my_off, probe_buf, probe_size, MPI_CHAR, MPI_STATUS_IGNORE);
            char* nl_pos = (char*)memchr(probe_buf, '\n', probe_size);
            
            if (nl_pos) {
                my_off += (nl_pos - probe_buf) + 1;  // 移动到行首
            }
            
            free(probe_buf);
        }

        // 创建大缓冲区（16MB）
        size_t buf_size = 16 * 1024 * 1024;
        char *buf = (char*)malloc(buf_size);
        if (!buf) {
            fprintf(stderr, "[%d] malloc failed\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // 创建哈希表
        HashMap *map = createHashMap(INIT_HASH_CAP);
        if (!map) {
            fprintf(stderr, "[%d] createHashMap failed\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        MPI_Offset cur = my_off;
        size_t buf_used = 0;
        
        while (cur < my_end) {
            // 计算本次读取大小
            MPI_Offset remaining = my_end - cur;
            size_t read_size = (remaining < buf_size - buf_used) ? 
                               (size_t)remaining : (buf_size - buf_used);
            
            if (read_size == 0) break;
            
            // 读取数据到缓冲区
            MPI_File_read_at(fh, cur, buf + buf_used, read_size, MPI_CHAR, MPI_STATUS_IGNORE);
            buf_used += read_size;
            cur += read_size;
            
            // 处理完整行
            char *start = buf;
            char *end = buf + buf_used;
            
            while (start < end) {
                char *line_end = (char*)memchr(start, '\n', end - start);
                
                if (line_end) {
                    size_t line_len = line_end - start;
                    
                    // 确保行长度在有效范围内
                    if (line_len > 0 && line_len <= MAX_KEYLEN) {
                        // 添加到哈希表
                        mapAdd(map, start, 1, key_len);
                    }
                    
                    // 移动到下一行
                    start = line_end + 1;
                } else {
                    // 移动剩余数据到缓冲区开头
                    size_t remaining = end - start;
                    if (remaining > 0 && remaining < buf_size) {
                        memmove(buf, start, remaining);
                        buf_used = remaining;
                    } else {
                        buf_used = 0;
                    }
                    break;
                }
            }
        }
        
        // 关闭文件并释放缓冲区
        MPI_File_close(&fh);
        free(buf);

        // 准备发送本地数据
        size_t local_n = map->size;
        Entry *loc_arr = NULL;
        
        if (local_n > 0) {
            loc_arr = (Entry*)malloc(local_n * sizeof(Entry));
            if (!loc_arr) {
                fprintf(stderr, "[%d] malloc failed\n", rank);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            
            // 复制哈希表数据到数组
            size_t idx = 0;
            for (size_t i = 0; i < map->cap; i++) {
                for (Node* p = map->buckets[i]; p; p = p->next) {
                    if (idx < local_n) {
                        memcpy(loc_arr[idx].key, p->key, MAX_KEYLEN + 1);
                        loc_arr[idx].count = p->count;
                        idx++;
                    }
                }
            }
        }
        
        // 释放本地哈希表
        freeHashMap(map);
        
        // 创建MPI数据类型
        MPI_Datatype entry_type;
        MPI_Type_contiguous(sizeof(Entry), MPI_BYTE, &entry_type);
        MPI_Type_commit(&entry_type);
        
        // 收集各进程数据量
        int local_int = (int)local_n;
        int *counts = NULL;
        int *displs = NULL;
        int total = 0;
        
        if (rank == 0) {
            counts = (int*)malloc(nprocs * sizeof(int));
            displs = (int*)malloc(nprocs * sizeof(int));
            if (!counts || !displs) {
                fprintf(stderr, "[0] malloc failed\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
        
        MPI_Gather(&local_int, 1, MPI_INT, counts, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        if (rank == 0) {
            displs[0] = 0;
            for (int i = 1; i < nprocs; i++) {
                displs[i] = displs[i-1] + counts[i-1];
            }
            total = displs[nprocs-1] + counts[nprocs-1];
        }
        
        // 广播总数据量
        MPI_Bcast(&total, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        // 分配接收缓冲区
        Entry *all = NULL;
        if (rank == 0 && total > 0) {
            all = (Entry*)malloc(total * sizeof(Entry));
            if (!all) {
                fprintf(stderr, "[0] malloc failed\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
        
        // 一次性收集所有数据（不再分批）
        if (loc_arr) {
            MPI_Gatherv(
                loc_arr, 
                local_int, 
                entry_type,
                (rank == 0) ? all : NULL, 
                (rank == 0) ? counts : NULL, 
                (rank == 0) ? displs : NULL, 
                entry_type,
                0, 
                MPI_COMM_WORLD
            );
        } else {
            MPI_Gatherv(
                NULL, 
                0, 
                entry_type,
                (rank == 0) ? all : NULL, 
                (rank == 0) ? counts : NULL, 
                (rank == 0) ? displs : NULL, 
                entry_type,
                0, 
                MPI_COMM_WORLD
            );
        }
        
        // 释放数据类型和本地数组
        MPI_Type_free(&entry_type);
        if (loc_arr) free(loc_arr);
        
        // Rank 0处理汇总结果
        if (rank == 0 && total > 0) {
            // 按key排序以便归并
            qsort(all, total, sizeof(Entry), cmpKey);
            
            // 合并相同项
            size_t j = 0;
            for (size_t i = 1; i < total; i++) {
                if (strcmp(all[i].key, all[j].key) == 0) {
                    all[j].count += all[i].count;
                } else {
                    j++;
                    all[j] = all[i];
                }
            }
            size_t unique_count = j + 1;
            
            // 按频率排序
            quickSortForEntry(all, 0, unique_count - 1);
            
            // 写入输出文件
            char outpath[PATH_MAX];
            snprintf(outpath, sizeof(outpath), "output/result%d_%s.txt", key_len, scale);
            FILE *fo = fopen(outpath, "w");
            if (!fo) {
                perror("fopen output");
                MPI_Abort(MPI_COMM_WORLD, 1);
            } 
            
            fprintf(fo, "%zu\n", unique_count);
            for (size_t i = 0; i < unique_count; i++) {
                fprintf(fo, "%.*s %d\n", key_len, all[i].key, all[i].count);
            }
            fclose(fo);
            
            double elapsed = MPI_Wtime() - t0;
            printf("  ✅ Done %s in %.3f seconds, unique keys = %zu\n", ent->d_name, elapsed, unique_count);
            
            // 清理内存
            free(all);
            free(counts);
            free(displs);
        } else if (rank == 0) {
            // 处理没有数据的情况
            if (counts) free(counts);
            if (displs) free(displs);
        }
        
        MPI_Barrier(MPI_COMM_WORLD);
    }

    closedir(d);
    if (rank == 0) printf("\nAll datasets processed.\n");
    MPI_Finalize();
    return 0;
}
