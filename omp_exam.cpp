#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <omp.h>
#include <stdbool.h>

// 自定义字符串数组结构体
typedef struct {
    char** data;
    size_t size;
    size_t capacity;
} StringArray;

void initStringArray(StringArray *a) {
    a->size = 0;
    a->capacity = 16;
    a->data = (char**)malloc(a->capacity * sizeof(char*));
}

void pushStringArray(StringArray *a, char* s) {
    if (a->size >= a->capacity) {
        a->capacity *= 2;
        a->data = (char**)realloc(a->data, a->capacity * sizeof(char*));
    }
    a->data[a->size++] = s;
}

void freeStringArray(StringArray *a) {
    for (size_t i = 0; i < a->size; i++) {
        free(a->data[i]);
    }
    free(a->data);
}

// 键值对条目结构
typedef struct {
    char* key;
    int count;
} Entry;

// 条目数组结构
typedef struct {
    Entry* data;
    size_t size;
    size_t capacity;
} EntryArray;

void initEntryArray(EntryArray *a) {
    a->size = 0;
    a->capacity = 16;
    a->data = (Entry*)malloc(a->capacity * sizeof(Entry));
}

void pushEntryArray(EntryArray *a, char* k, int c) {
    if (a->size >= a->capacity) {
        a->capacity *= 2;
        a->data = (Entry*)realloc(a->data, a->capacity * sizeof(Entry));
    }
    a->data[a->size].key = k;
    a->data[a->size].count = c;
    a->size++;
}

void freeEntryArray(EntryArray *a) {
    free(a->data);
}

// 哈希表节点
typedef struct Node {
    char* key;
    int count;
    struct Node* next;
} Node;

// 哈希表结构
typedef struct {
    size_t cap;
    Node** buckets;
    omp_lock_t* locks;
} HashMap;

HashMap* createHashMap(size_t cap) {
    HashMap* m = (HashMap*)malloc(sizeof(HashMap));
    m->cap = cap;
    m->buckets = (Node**)calloc(cap, sizeof(Node*));
    m->locks = (omp_lock_t*)malloc(cap * sizeof(omp_lock_t));
    for (size_t i = 0; i < cap; i++) {
        omp_init_lock(&m->locks[i]);
    }
    return m;
}

void destroyHashMap(HashMap* m) {
    for (size_t i = 0; i < m->cap; i++) {
        Node* n = m->buckets[i];
        while (n) {
            Node* t = n->next;
            free(n->key);
            free(n);
            n = t;
        }
        omp_destroy_lock(&m->locks[i]);
    }
    free(m->locks);
    free(m->buckets);
    free(m);
}

// DJB2哈希算法
unsigned long hash_str(const char* s) {
    unsigned long h = 5381;
    int c;
    while ((c = *s++)) {
        h = ((h << 5) + h) + c;
    }
    return h;
}

void mapAdd(HashMap* m, char* k, int cnt) {
    unsigned long h = hash_str(k) % m->cap;
    omp_set_lock(&m->locks[h]);
    
    Node* cur = m->buckets[h];
    while (cur) {
        if (strcmp(cur->key, k) == 0) {
            cur->count += cnt;
            omp_unset_lock(&m->locks[h]);
            return;
        }
        cur = cur->next;
    }
    
    Node* n = (Node*)malloc(sizeof(Node));
    n->key = strdup(k);
    n->count = cnt;
    n->next = m->buckets[h];
    m->buckets[h] = n;
    
    omp_unset_lock(&m->locks[h]);
}

void mapCollect(HashMap* m, EntryArray *a) {
    for (size_t i = 0; i < m->cap; i++) {
        Node* n = m->buckets[i];
        while (n) {
            pushEntryArray(a, n->key, n->count);
            n = n->next;
        }
    }
}

// 并行归并哈希表
void mergeMaps(HashMap** locals, int threads, HashMap* global) {
    #pragma omp parallel for num_threads(threads) schedule(dynamic)
    for (int t = 0; t < threads; t++) {
        EntryArray tmp;
        initEntryArray(&tmp);
        mapCollect(locals[t], &tmp);
        
        for (size_t i = 0; i < tmp.size; i++) {
            mapAdd(global, tmp.data[i].key, tmp.data[i].count);
        }
        
        freeEntryArray(&tmp);
    }
}

// 归并排序辅助函数
void merge(Entry* A, int l, int m, int r, Entry* tmp) {
    int i = l, j = m+1, k = l;
    while (i <= m && j <= r) {
        if (A[i].count > A[j].count || 
            (A[i].count == A[j].count && strcmp(A[i].key, A[j].key) < 0)) {
            tmp[k++] = A[i++];
        } else {
            tmp[k++] = A[j++];
        }
    }
    while (i <= m) tmp[k++] = A[i++];
    while (j <= r) tmp[k++] = A[j++];
    
    for (i = l; i <= r; i++) {
        A[i] = tmp[i];
    }
}

// 并行归并排序
void parallelSort(Entry* A, int l, int r, Entry* tmp) {
    if (l < r) {
        int m = (l+r)/2;
        #pragma omp task if (r-l > 1000)
        parallelSort(A, l, m, tmp);
        #pragma omp task if (r-l > 1000)
        parallelSort(A, m+1, r, tmp);
        #pragma omp taskwait
        merge(A, l, m, r, tmp);
    }
}

int main() {
    DIR* dir = opendir("dataset");
    if (!dir) {
        perror("opendir failed");
        return 1;
    }
    
    struct dirent* ent;
    while ((ent = readdir(dir))) {
        if (strncmp(ent->d_name, "data_", 5) != 0) continue;
        
        int len = 0, countM = 0;
        sscanf(ent->d_name, "data_%d_%dM.txt", &len, &countM);
        
        char inpath[300], outpath[300], anspath[300];
        snprintf(inpath, sizeof(inpath), "dataset/%s", ent->d_name);
        snprintf(outpath, sizeof(outpath), "output/result%d_%dM.txt", len, countM);
        snprintf(anspath, sizeof(anspath), "answer/result%d_%dM.txt", len, countM);
        
        // 读取文件数据
        FILE* fin = fopen(inpath, "r");
        if (!fin) {
            fprintf(stderr, "Cannot open: %s\n", inpath);
            continue;
        }
        
        StringArray data;
        initStringArray(&data);
        char buf[128];
        
        while (fgets(buf, sizeof(buf), fin)) {
            // 移除换行符
            char* nl = strchr(buf, '\n');
            if (nl) *nl = '\0';
            pushStringArray(&data, strdup(buf));
        }
        fclose(fin);
        int n = data.size;
        
        // 设置并行线程数
        int tcount = omp_get_max_threads();
        omp_set_num_threads(tcount);
        double t0 = omp_get_wtime();
        
        // 每个线程创建本地哈希表
        HashMap** locals = (HashMap**)malloc(tcount * sizeof(HashMap*));
        #pragma omp parallel num_threads(tcount)
        {
            int tid = omp_get_thread_num();
            locals[tid] = createHashMap(1 << 20); // 1M buckets
        }
        
        // 并行处理数据
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < n; i++) {
            int tid = omp_get_thread_num();
            mapAdd(locals[tid], data.data[i], 1);
        }
        
        // 创建全局哈希表并合并结果
        HashMap* global = createHashMap(1 << 22); // 4M buckets
        mergeMaps(locals, tcount, global);
        
        // 清理本地哈希表
        for (int t = 0; t < tcount; t++) {
            destroyHashMap(locals[t]);
        }
        free(locals);
        
        // 收集结果并排序
        EntryArray res;
        initEntryArray(&res);
        mapCollect(global, &res);
        
        Entry* tmp = (Entry*)malloc(res.size * sizeof(Entry));
        #pragma omp parallel
        {
            #pragma omp single
            parallelSort(res.data, 0, res.size-1, tmp);
        }
        free(tmp);
        
        // 写入结果文件
        FILE* fout = fopen(outpath, "w");
        if (!fout) {
            fprintf(stderr, "Cannot write: %s\n", outpath);
        } else {
            fprintf(fout, "%zu\n", res.size);
            for (size_t i = 0; i < res.size; i++) {
                fprintf(fout, "%s %d\n", res.data[i].key, res.data[i].count);
            }
            fclose(fout);
        }
        
        // 验证结果
        bool ok = true;
        FILE* fans = fopen(anspath, "r");
        if (fans) {
            FILE* fcheck = fopen(outpath, "r");
            if (fcheck) {
                char line1[256], line2[256];
                while (fgets(line1, sizeof(line1), fcheck)) {
                    if (!fgets(line2, sizeof(line2), fans) || strcmp(line1, line2)) {
                        ok = false;
                        break;
                    }
                }
                if (fgets(line2, sizeof(line2), fans)) ok = false;
                fclose(fcheck);
            }
            fclose(fans);
        }
        
        double t1 = omp_get_wtime();
        printf("Processed %s (%d lines) in %.3f s | Threads: %d | %s\n", 
               ent->d_name, n, t1-t0, tcount, ok ? "OK" : "ERROR");
        
        // 清理资源
        destroyHashMap(global);
        freeEntryArray(&res);
        freeStringArray(&data);
    }
    
    closedir(dir);
    return 0;
}