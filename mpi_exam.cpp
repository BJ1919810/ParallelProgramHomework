#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <climits>

#define MAX_KEYLEN 64

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
    m->buckets = (Node**)calloc(cap, sizeof(Node*));
    if (!m->buckets) {
        free(m);
        return NULL;
    }
    return m;
}

void mapAdd(HashMap* m, const char* key, int cnt, int key_len) {
    unsigned long h = hash_str(key, key_len) % m->cap;
    Node* cur = m->buckets[h];
    while (cur) {
        if (memcmp(cur->key, key, key_len) == 0) {
            cur->count += cnt;
            return;
        }
        cur = cur->next;
    }
    Node* n = (Node*)malloc(sizeof(Node));
    if (!n) return;
    memcpy(n->key, key, key_len);
    n->key[key_len] = '\0';
    n->count = cnt;
    n->next = m->buckets[h];
    m->buckets[h] = n;
}

void freeHashMap(HashMap* m) {
    if (!m) return;
    for (size_t i = 0; i < m->cap; i++) {
        Node* cur = m->buckets[i];
        while (cur) {
            Node* nxt = cur->next;
            free(cur);
            cur = nxt;
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

        // 打开文件并统计每个进程应读区间
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
        
        MPI_Offset file_size;
        MPI_File_get_size(fh, &file_size);
        MPI_Offset chunk = file_size / nprocs;
        MPI_Offset my_off = rank * chunk;
        MPI_Offset my_end = (rank == nprocs - 1) ? file_size : (my_off + chunk);

        // 边界处理：确保从行首开始读取
        if (rank > 0) {
            if (my_off > 0) {
                char c;
                MPI_File_seek(fh, my_off - 1, MPI_SEEK_SET);
                MPI_File_read(fh, &c, 1, MPI_CHAR, MPI_STATUS_IGNORE);
                if (c != '\n') {
                    // 跳过不完整的行
                    while (my_off < file_size) {
                        MPI_File_read(fh, &c, 1, MPI_CHAR, MPI_STATUS_IGNORE);
                        my_off++;
                        if (c == '\n') break;
                    }
                } else {
                    my_off--; // 回退到换行符后
                }
            }
        }

        // 创建动态缓冲区
        int buf_size = (key_len + 2) * 1024; // 每个行缓冲区大小
        char *buf = (char*)malloc(buf_size);
        if (!buf) {
            fprintf(stderr, "[%d] malloc failed\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // 修复1: 使用MPI_File_read而不是fdopen
        HashMap *map = createHashMap(100003);
        MPI_Offset cur = my_off;
        
        while (cur < my_end) {
            // 计算本次读取的大小
            MPI_Offset remaining = my_end - cur;
            size_t read_size = (remaining < buf_size) ? remaining : buf_size;
            
            if (read_size == 0) break;
            
            // 读取一块数据
            MPI_File_read_at(fh, cur, buf, read_size, MPI_CHAR, MPI_STATUS_IGNORE);
            
            // 处理缓冲区中的数据
            char *start = buf;
            char *end = buf + read_size;
            
            while (start < end) {
                // 查找行结束符
                char *line_end = (char*)memchr(start, '\n', end - start);
                
                if (line_end) {
                    // 计算行长度
                    size_t line_len = line_end - start;
                    
                    // 确保行长度正确
                    if (line_len == (size_t)key_len) {
                        // 复制键值并添加到哈希表
                        char key[MAX_KEYLEN + 1];
                        memcpy(key, start, key_len);
                        key[key_len] = '\0';
                        mapAdd(map, key, 1, key_len);
                    }
                    
                    // 移动到下一行
                    start = line_end + 1;
                    cur += line_len + 1;
                } else {
                    // 没有找到完整的行，移动剩余数据到缓冲区开头
                    size_t remaining = end - start;
                    if (remaining > 0) {
                        memmove(buf, start, remaining);
                    }
                    break;
                }
            }
            
            // 如果缓冲区有剩余数据，调整起始位置
            if (start < end) {
                size_t remaining = end - start;
                memmove(buf, start, remaining);
                start = buf + remaining;
            } else {
                start = buf;
            }
        }
        
        // 关闭MPI文件
        MPI_File_close(&fh);
        free(buf);

        // 汇总 map entry counts 到 rank 0
        size_t local_n = 0;
        for (size_t i = 0; i < map->cap; i++)
            for (Node* p = map->buckets[i]; p; p = p->next)
                local_n++;

        Entry *loc_arr = (Entry*)malloc(local_n * sizeof(Entry));
        if (!loc_arr) {
            fprintf(stderr, "[%d] malloc failed\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        size_t idx = 0;
        for (size_t i = 0; i < map->cap; i++)
            for (Node* p = map->buckets[i]; p; p = p->next) {
                memcpy(loc_arr[idx].key, p->key, key_len + 1);
                loc_arr[idx].count = p->count;
                idx++;
            }

        int *counts = NULL;
        int *displs = NULL;
        Entry *all = NULL;
        int total = 0;

        int local_int = (int)local_n;

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
            for (int i = 1; i < nprocs; i++)
                displs[i] = displs[i-1] + counts[i-1];
            total = displs[nprocs-1] + counts[nprocs-1];
            all = (Entry*)malloc(total * sizeof(Entry));
            if (!all) {
                fprintf(stderr, "[0] malloc failed\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }

        MPI_Datatype entry_type;
        MPI_Type_contiguous(sizeof(Entry), MPI_BYTE, &entry_type);
        MPI_Type_commit(&entry_type);

        MPI_Gatherv(loc_arr, local_int, entry_type,
                    all, counts, displs, entry_type,
                    0, MPI_COMM_WORLD);

        MPI_Type_free(&entry_type);

        if (rank == 0) {
            // 归并计数
            HashMap *agg = createHashMap(200003);
            for (size_t i = 0; i < total; i++) {
                mapAdd(agg, all[i].key, all[i].count, key_len);
            }

            size_t m = 0;
            for (size_t i = 0; i < agg->cap; i++)
                for (Node* p = agg->buckets[i]; p; p = p->next)
                    m++;

            Entry *agg_arr = (Entry*)malloc(m * sizeof(Entry));
            size_t id2 = 0;
            for (size_t i = 0; i < agg->cap; i++)
                for (Node* p = agg->buckets[i]; p; p = p->next) {
                    memcpy(agg_arr[id2].key, p->key, key_len + 1);
                    agg_arr[id2].count = p->count;
                    id2++;
                }
            quickSortForEntry(agg_arr, 0, m - 1);

            // 写入输出文件
            char outpath[PATH_MAX];
            snprintf(outpath, sizeof(outpath), "output/result%d_%s_mpi.txt", key_len, scale);
            FILE *fo = fopen(outpath, "w");
            if (!fo) {
                perror("fopen output");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            fprintf(fo, "%zu\n", m);
            for (size_t i = 0; i < m; i++) {
                fprintf(fo, "%.*s %d\n", key_len, agg_arr[i].key, agg_arr[i].count);
            }
            fclose(fo);

            double elapsed = MPI_Wtime() - t0;
            printf("  ✅ Done %s in %.3f seconds, unique keys = %zu\n", ent->d_name, elapsed, m);

            // 清理内存
            free(agg_arr);
            freeHashMap(agg);
            free(all);
            free(counts);
            free(displs);
        }

        free(loc_arr);
        freeHashMap(map);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    closedir(d);
    if (rank == 0) printf("\nAll datasets processed.\n");
    MPI_Finalize();
    return 0;
}