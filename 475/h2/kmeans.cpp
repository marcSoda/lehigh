#include <iostream>
#include <tbb/tbb.h>

using namespace std;

struct Point {
    int pid, cid;
    std::vector<double> vals;

    Point(int pid, std::vector<double>& vals) {
        this->pid = pid;
        this->vals = vals;
        cid = -1;
    }
};

class Cluster {
public:
    int cid;
    vector<double> centroid;
    vector<double> sum_vals;
    int count;

    Cluster(int cid, Point point, int num_dims) {
        this->cid = cid;
        centroid.resize(num_dims);
        sum_vals.resize(num_dims);
        count = 1;
        for (int i = 0; i < num_dims; i++) {
            centroid[i] = point.vals[i];
            sum_vals[i] = point.vals[i];
        }
    }

    void add_point(Point point) {
        for (int i = 0; i < centroid.size(); i++)
            sum_vals[i] += point.vals[i];
        count++;
    }

    void remove_point(Point point) {
        for (int i = 0; i < centroid.size(); i++)
            sum_vals[i] -= point.vals[i];
        count--;
    }

    void calculate_centroid() {
        for (int i = 0; i < centroid.size(); i++)
            centroid[i] = sum_vals[i] / count;
    }

    double squared_euclidean(Point point) {
        double dist = 0;
        for (int i = 0; i < centroid.size(); i++) {
            dist += (centroid[i] - point.vals[i]) * (centroid[i] - point.vals[i]);
        }
        return dist;
    }
};

class KMeans {
public:
    int K, num_dims, num_points, max_iterations;
    vector<Point> points;
    tbb::concurrent_vector<Cluster> clusters;

    KMeans(int K, vector<Point> points, int num_points, int num_dims, int max_iterations) {
        this->K = K;
        this->points = points;
        this->num_points = num_points;
        this->num_dims = num_dims;
        this->max_iterations = max_iterations;
    }

    int get_id_nearest_center(Point point) {
        double min = clusters[0].squared_euclidean(point);
        int cid = 0;
        for (int i = 1; i < K; i++) {
            double dist = clusters[i].squared_euclidean(point);
            if (dist < min) {
                min = dist;
                cid = i;
            }
        }
        return cid;
    }

    void rand_cluster_centers() {
        if (K > num_points) return;
        tbb::concurrent_unordered_set<int> selected_indexes;
        tbb::parallel_for(
            tbb::blocked_range<int>(0, K),
            [&](const tbb::blocked_range<int>& range) {
                for (int i = range.begin(); i < range.end(); i++) {
                    int pid;
                    do pid = rand() % num_points;
                    while (!selected_indexes.insert(pid).second);
                    points[pid].cid = i;
                    clusters.push_back(Cluster(i, points[pid], num_dims));
                }
            }
        );
    }

    bool assign_points() {
        return tbb::parallel_reduce(
            tbb::blocked_range<int>(0, num_points),
            true,
            [&](const tbb::blocked_range<int>& r, bool done) -> bool {
                for (int i = r.begin(); i != r.end(); ++i) {
                    int old_cid = points[i].cid;
                    int nearest_cid = get_id_nearest_center(points[i]);
                    if (old_cid != nearest_cid) {
                        if (old_cid != -1) {
                            clusters[old_cid].remove_point(points[i]);
                        }
                        points[i].cid = nearest_cid;
                        clusters[nearest_cid].add_point(points[i]);
                        done = false;
                    }
                }
                return done;
            },
            [](bool x, bool y) -> bool { return x && y; }
        );
    }

    void run() {
        auto begin = chrono::high_resolution_clock::now();
        if (K > num_points) return;
        rand_cluster_centers();
        auto end_phase1 = chrono::high_resolution_clock::now();
        int iter = 0;
        bool done = false;
        while (iter < max_iterations && !done) {
            done = assign_points();
            tbb::parallel_for(0, K, [&](int i) {
                clusters[i].calculate_centroid();
            });
            iter++;
        }
        cout << "Break in iteration " << iter << "\n\n";
        auto end = chrono::high_resolution_clock::now();
        for(int i = 0; i < K; i++) {
            cout << "\nCluster " << clusters[i].cid + 1 << endl;
            cout << "Cluster values: ";
            for(int j = 0; j < num_dims; j++)
                cout << clusters[i].centroid[j] << " ";
        }
        cout << "\n\n";
        cout << "TOTAL EXECUTION TIME = "<<chrono::duration_cast<chrono::microseconds>(end-begin).count()<<"\n";
        cout << "TIME PHASE 1 = "<<chrono::duration_cast<chrono::microseconds>(end_phase1-begin).count()<<"\n";
        cout << "TIME PHASE 2 = "<<chrono::duration_cast<chrono::microseconds>(end-end_phase1).count()<<"\n";
    }
};

int main(int argc, char *argv[]) {
    srand (time(NULL));

    int num_points, num_dims, K, max_iterations;

    cin >> num_points >> num_dims >> K >> max_iterations;

    vector<Point> points;

    for(int i = 0; i < num_points; i++) {
        vector<double> values;

        for(int j = 0; j < num_dims; j++) {
            double value;
            cin >> value;
            values.push_back(value);
        }

        Point p(i, values);
        points.push_back(p);
    }

    KMeans kmeans(K, points, num_points, num_dims, max_iterations);
    kmeans.run();

    return 0;
}

// // simd_attempt
// void add_point(Point point) {
//     const int vec_size = 4; // Size of AVX vector
//     int num_vecs = centroid.size() / vec_size; // Number of complete AVX vectors
//     __m256d sum_vec = _mm256_loadu_pd(&sum_vals[0]); // Load sum_vals into an AVX vector
//     __m256d point_vec = _mm256_loadu_pd(&point.vals[0]); // Load point into an AVX vector
//     for (int i = 1; i < num_vecs; i++) {
//         __m256d sum_vec_i = _mm256_loadu_pd(&sum_vals[i*vec_size]); // Load i-th vector of sum_vals
//         __m256d point_vec_i = _mm256_loadu_pd(&point.vals[i*vec_size]); // Load i-th vector of point
//         sum_vec_i = _mm256_add_pd(sum_vec_i, point_vec_i); // Add i-th vectors
//         _mm256_storeu_pd(&sum_vals[i*vec_size], sum_vec_i); // Store i-th vector back to sum_vals
//     }
//     for (int i = num_vecs*vec_size; i < centroid.size(); i++) {
//         sum_vals[i] += point.vals[i]; // Process remaining elements with scalar operations
//     }
//     count++;
// }
