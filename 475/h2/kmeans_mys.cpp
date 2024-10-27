#include <iostream>
#include <vector>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <chrono>

using namespace std;

class Point {
public:
	int pid, cid;
	vector<double> vals;
	string name;

	Point(int pid, vector<double> &vals, string name = "") {
		this->pid = pid;
		this->vals = vals;
		this->name = name;
		this->cid = -1;
	}
};

class Cluster {
public:
    int cid;
    vector<double> centroid;
    vector<double> sum_vals;
    int count;

    Cluster(int id_cluster, Point point, int num_dims) {
        this->cid = id_cluster;
        this->centroid.resize(num_dims);
        this->sum_vals.resize(num_dims);
        this->count = 1;
        for (int i = 0; i < num_dims; i++) {
            this->centroid[i] = point.vals[i];
            this->sum_vals[i] = point.vals[i];
        }
    }

    void add_point(Point point) {
        for (int i = 0; i < centroid.size(); i++) {
            sum_vals[i] += point.vals[i];
            centroid[i] = sum_vals[i];
        }
        this->calculate_centroid();
        count++;
    }

    void remove_point(Point point) {
        for (int i = 0; i < centroid.size(); i++) {
            sum_vals[i] -= point.vals[i];
            centroid[i] = sum_vals[i];
        }
        this->calculate_centroid();
        count--;
    }

    void calculate_centroid() {
        for (int i = 0; i < centroid.size(); i++) {
            centroid[i] = sum_vals[i] / count;
        }
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
    vector<Cluster> clusters;

    KMeans(int K, vector<Point> points, int total_points, int num_dims, int max_iterations) {
        this->K = K;
		this->points = points;
        this->num_points = total_points;
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
        vector<int> selected_indexes;
        for (int i = 0; i < K; i++) {
            int index_point = rand() % num_points;
            if (find(selected_indexes.begin(), selected_indexes.end(), index_point) != selected_indexes.end()) {
                i--;
                continue;
            }
            selected_indexes.push_back(index_point);
            points[index_point].cid = i;
            Cluster cluster(i, points[index_point], num_dims);
            clusters.push_back(cluster);
        }
    }

    bool assign_points() {
        bool done = true;
        for (int i = 0; i < num_points; i++) {
            int id_old_cluster = points[i].cid;
            int id_nearest_center = get_id_nearest_center(points[i]);
            if (id_old_cluster != id_nearest_center) {
                if (id_old_cluster != -1) {
                    clusters[id_old_cluster].remove_point(points[i]);
                }
                points[i].cid = id_nearest_center;
                clusters[id_nearest_center].add_point(points[i]);
                done = false;
            }
        }
        return done;
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
			for (int i = 0; i < K; i++) clusters[i].calculate_centroid();
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
		cout << "TOTAL EXECUTION TIME = "<<std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count()<<"\n";
		cout << "TIME PHASE 1 = "<<std::chrono::duration_cast<std::chrono::microseconds>(end_phase1-begin).count()<<"\n";
		cout << "TIME PHASE 2 = "<<std::chrono::duration_cast<std::chrono::microseconds>(end-end_phase1).count()<<"\n";
	}
};

int main(int argc, char *argv[]) {
	srand(time(NULL));

	int num_points, num_dims, K, max_iterations, has_name;

	cin >> num_points >> num_dims >> K >> max_iterations >> has_name;

	vector<Point> points;
	string point_name;

	for(int i = 0; i < num_points; i++) {
		vector<double> values;

		for(int j = 0; j < num_dims; j++) {
			double value;
			cin >> value;
			values.push_back(value);
		}

		if(has_name) {
			cin >> point_name;
			Point p(i, values, point_name);
			points.push_back(p);
		}
		else {
			Point p(i, values);
			points.push_back(p);
		}
	}

	KMeans kmeans(K, points, num_points, num_dims, max_iterations);
	kmeans.run();

	return 0;
}
